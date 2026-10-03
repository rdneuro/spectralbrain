"""Triangle mesh representation and mesh-specific geometric analysis.

Provides :class:`BrainMesh`, the primary mesh container, which
implements :class:`~spectralbrain.core.base.GeometricObject` and
owns the Laplacian construction + eigendecomposition pipeline.

Also provides standalone functions for mesh-specific geometry that
cannot be computed on unstructured point clouds: cotangent Laplacian,
angle-defect Gaussian curvature, heat-method geodesics, Euler
characteristic, etc.

Dependencies
------------
- **robust_laplacian** — optional, for the Sharp–Crane tufted
  Laplacian (recommended on non-manifold meshes).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from spectralbrain.backends.cpu import NumpyBackend
from spectralbrain.core.base import (
    SpectralDecomposition,
    mesh_surface_area,
    triangle_areas,
)
from spectralbrain.runtime import (
    Faces,
    MassMatrix,
    Normals,
    ScalarMap,
    SparseMatrix,
    Vertices,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  BRAIN MESH CLASS
# ======================================================================


class BrainMesh:
    """Triangle surface mesh for brain structures.

    Implements the :class:`GeometricObject` protocol and provides
    the Laplacian construction → eigendecomposition → descriptor
    pipeline.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates in mm.
    faces : ndarray, shape (F, 3)
        Triangle indices, 0-indexed.
    metadata : dict, optional
        Provenance (subject, hemisphere, structure, …).

    Examples
    --------
    >>> verts, faces = sb.io.load_freesurfer_surface("lh.white")
    >>> mesh = BrainMesh(verts, faces, metadata={"structure": "cortex"})
    >>> decomp = mesh.decompose(k=100)
    """

    def __init__(
        self,
        vertices: Vertices,
        faces: Faces,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialise from vertices and faces arrays."""
        self._invalidate()
        self.vertices = vertices
        self.faces = faces
        self.metadata = metadata or {}

    # ── Geometry (assignment invalidates every cached quantity) ───────

    def _invalidate(self) -> None:
        """Drop all cached geometry-derived quantities."""
        self._normals: Normals | None = None
        self._L: SparseMatrix | None = None
        self._M: MassMatrix | None = None
        self._lap_method: str | None = None
        self._area: float | None = None

    @property
    def vertices(self) -> Vertices:
        """Vertex coordinates, shape ``(N, 3)``.

        Re-assigning ``mesh.vertices = ...`` clears the cached Laplacian,
        mass matrix, normals and area.  After mutating the array *in place*
        call :meth:`clear_cache` explicitly.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, value: Vertices) -> None:
        v = np.asarray(value, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {v.shape}")
        self._vertices = v
        self._invalidate()

    @property
    def faces(self) -> Faces:
        """Triangle indices, shape ``(F, 3)``; re-assignment clears caches."""
        return self._faces

    @faces.setter
    def faces(self, value: Faces) -> None:
        f = np.asarray(value, dtype=np.int64)
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3), got {f.shape}")
        self._faces = f
        self._invalidate()

    def clear_cache(self) -> None:
        """Clear cached Laplacian/mass/normals/area (after in-place edits)."""
        self._invalidate()

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_volume(
        cls,
        volume: np.ndarray,
        affine: np.ndarray,
        *,
        raw: bool = False,
        label: int | Sequence[int] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BrainMesh:
        """Build a mesh from a (label) volume, improved by default for the LBO.

        Thin wrapper over :func:`spectralbrain.io.meshing.volume_to_mesh`. With
        ``raw=False`` (default) the volume is turned into a well-conditioned,
        LBO-ready surface (sub-voxel iso-surface, topology cleanup, Taubin
        smoothing, isotropic remeshing). With ``raw=True`` it reproduces the
        legacy plain-marching-cubes path exactly.

        Parameters
        ----------
        volume : ndarray, shape (X, Y, Z)
            Intensity or integer-label volume.
        affine : ndarray, shape (4, 4)
            Voxel-to-world affine.
        raw : bool
            Skip the improvement pipeline (legacy behaviour) if True.
        label : int or sequence of int, optional
            Restrict to these label id(s) before meshing.
        metadata : dict, optional
            Provenance stored on the returned mesh.
        **kwargs
            Forwarded to :func:`~spectralbrain.io.meshing.volume_to_mesh`
            (e.g. ``closed``, ``level``, ``field_mode``, ``taubin_iterations``,
            ``remesh``, ``target_edge_mm``).

        Returns
        -------
        BrainMesh

        Examples
        --------
        >>> seg, affine = sb.io.load_nifti("aseg.mgz")
        >>> hippo = BrainMesh.from_volume(seg, affine, label=17)  # improved
        >>> raw = BrainMesh.from_volume(seg, affine, label=17, raw=True)
        """
        # Local import avoids the io <-> core import cycle at module load.
        from spectralbrain.io.meshing import volume_to_mesh

        verts, faces, info = volume_to_mesh(
            volume, affine, raw=raw, label=label, return_info=True, **kwargs
        )
        meta = {"source": "volume", **info}
        if metadata:
            meta.update(metadata)
        return cls(verts, faces, metadata=meta)

    # ── GeometricObject protocol ──────────────────────────────────────

    @property
    def coordinates(self) -> Vertices:
        """Return the vertex coordinates array."""
        return self.vertices

    @property
    def n_points(self) -> int:
        """Return the number of vertices (alias for n_vertices)."""
        return self.vertices.shape[0]

    @property
    def n_vertices(self) -> int:
        """Return the number of vertices in the mesh."""
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        """Return the number of triangular faces."""
        return self.faces.shape[0]

    def surface_area(self) -> float:
        """Total surface area in mm²."""
        if self._area is None:
            self._area = mesh_surface_area(self.vertices, self.faces)
        return self._area

    # ── Laplacian construction ────────────────────────────────────────

    def compute_laplacian(
        self,
        method: Literal["cotangent", "robust"] = "cotangent",
        *,
        robust_mollify_factor: float = 1e-5,
    ) -> tuple[SparseMatrix, MassMatrix]:
        """Construct the Laplacian and mass matrix.

        Parameters
        ----------
        method : str
            ``"cotangent"`` — FEM cotangent-weighted Laplacian
            (Pinkall–Polthier / Meyer–Desbrun–Schröder–Barr weights) with
            the barycentric lumped mass matrix (area/3 per incident
            triangle).  Degenerate (zero-area) triangles are excluded with
            a warning.
            ``"robust"`` — Sharp & Crane tufted Laplacian via
            ``robust_laplacian`` (handles non-manifold edges).
        robust_mollify_factor : float
            Mollification parameter for the robust method.

        Returns
        -------
        L : SparseMatrix, shape (N, N)
            Stiffness (Laplacian) matrix — symmetric positive
            semi-definite.
        M : MassMatrix, shape (N, N)
            Diagonal mass matrix.
        """
        if method == "cotangent":
            L, M = _cotangent_laplacian(self.vertices, self.faces)
        elif method == "robust":
            L, M = _robust_laplacian_mesh(
                self.vertices,
                self.faces,
                mollify_factor=robust_mollify_factor,
            )
        else:
            raise ValueError(f"Unknown Laplacian method: {method!r}")

        self._L = L
        self._M = M
        self._lap_method = method
        logger.info(
            "Laplacian (%s): N=%d, nnz=%d",
            method,
            L.shape[0],
            L.nnz,
        )
        return L, M

    # ── Spectral decomposition ────────────────────────────────────────

    def decompose(
        self,
        k: int = 100,
        *,
        laplacian_method: Literal["cotangent", "robust"] | None = None,
        backend: Any | None = None,
        **eigsh_kwargs: Any,
    ) -> SpectralDecomposition:
        """Compute the spectral decomposition.

        Parameters
        ----------
        k : int
            Number of eigenpairs to compute.
        laplacian_method : str, optional
            Passed to :meth:`compute_laplacian`.  ``None`` (default) reuses
            a Laplacian already built by :meth:`compute_laplacian`, or
            builds the ``"cotangent"`` one.  An explicit method that differs
            from the cached one triggers a rebuild.
        backend : Backend, optional
            Compute backend.  Defaults to :class:`NumpyBackend`.
        **eigsh_kwargs
            Extra arguments to ``backend.eigsh()``.

        Returns
        -------
        SpectralDecomposition
        """
        method = laplacian_method or self._lap_method or "cotangent"
        if self._L is None or self._M is None or self._lap_method != method:
            self.compute_laplacian(method=method)

        be = backend or NumpyBackend()
        evals, evecs = be.eigsh(self._L, self._M, k=k, **eigsh_kwargs)

        return SpectralDecomposition(
            eigenvalues=evals,
            eigenvectors=evecs,
            stiffness=self._L,
            mass=self._M,
            surface_area=self.surface_area(),
            metadata={
                **self.metadata,
                "laplacian_method": self._lap_method,
                "backend": be.name,
                "n_vertices": self.n_vertices,
                "n_faces": self.n_faces,
            },
        )

    # ── Normals ───────────────────────────────────────────────────────

    def compute_normals(self) -> Normals:
        """Compute area-weighted per-vertex normals from face topology.

        Returns
        -------
        normals : ndarray, shape (N, 3)
            Unit normals.
        """
        if self._normals is not None:
            return self._normals
        self._normals = _vertex_normals(self.vertices, self.faces)
        return self._normals

    # ── Curvature ─────────────────────────────────────────────────────

    def gaussian_curvature(self) -> ScalarMap:
        """Gaussian curvature via angle defect (Descartes–Euler).

        K(v) = (2π − Σ θ_j) / A(v)   (interior vertices)
        K(v) = (π − Σ θ_j) / A(v)    (boundary vertices; geodesic curvature
        defect, so a flat open patch has K = 0 everywhere)

        where θ_j are the face angles at vertex v and A(v) is the
        mixed Voronoi area (Meyer et al. 2003).

        Returns
        -------
        ndarray, shape (N,)
            Gaussian curvature in 1/mm².
        """
        return _gaussian_curvature(self.vertices, self.faces)

    def mean_curvature(self) -> ScalarMap:
        """Mean curvature via the Laplacian (Hn = ΔX / 2).

        Uses the cotangent Laplacian (built, or reused if already cached)
        and the mass matrix.  The sign is taken relative to the
        area-weighted vertex normals implied by the face winding: with
        outward (counter-clockwise) winding, convex regions (gyral crowns,
        a sphere) have H > 0 and concave regions (sulcal fundi) H < 0.

        Returns
        -------
        ndarray, shape (N,)
            Mean curvature (signed) in 1/mm.
        """
        if self._lap_method == "cotangent" and self._L is not None and self._M is not None:
            L, M = self._L, self._M
        elif self._L is None:
            L, M = self.compute_laplacian(method="cotangent")
        else:
            # A non-cotangent Laplacian is cached for decompose(); do not
            # overwrite it — build a private cotangent operator instead.
            L, M = _cotangent_laplacian(self.vertices, self.faces)
        return _mean_curvature_laplacian(
            self.vertices,
            L,
            M,
            faces=self.faces,
        )

    def principal_curvatures(self) -> tuple[ScalarMap, ScalarMap]:
        """Principal curvatures κ₁ ≥ κ₂ from H and K.

        κ₁ = H + √(H² − K),   κ₂ = H − √(H² − K)

        Returns
        -------
        kappa1 : ndarray, shape (N,)
        kappa2 : ndarray, shape (N,)
        """
        H = self.mean_curvature()
        K = self.gaussian_curvature()
        disc = np.clip(H**2 - K, 0.0, None)
        sqrt_disc = np.sqrt(disc)
        return H + sqrt_disc, H - sqrt_disc

    def shape_index(self) -> ScalarMap:
        """Koenderink Shape Index S ∈ [−1, +1].

        S = (2/π) · arctan((κ₁ + κ₂) / (κ₁ − κ₂)),  κ₁ ≥ κ₂

        Convex caps → +1, concave cups → −1, saddles → 0 (with outward
        face winding).

        Returns
        -------
        ndarray, shape (N,)
        """
        k1, k2 = self.principal_curvatures()
        # κ₁ ≥ κ₂ ⇒ κ₁ − κ₂ ≥ 0, so arctan2 stays in [−π/2, π/2] (S ∈ [−1, 1])
        # and umbilics (κ₁ = κ₂) map to ±1 by the sign of κ₁ + κ₂.
        return (2.0 / np.pi) * np.arctan2(k1 + k2, k1 - k2)

    def curvedness(self) -> ScalarMap:
        """Koenderink Curvedness C ≥ 0.

        C = √((κ₁² + κ₂²) / 2)

        Returns
        -------
        ndarray, shape (N,)
        """
        k1, k2 = self.principal_curvatures()
        return np.sqrt((k1**2 + k2**2) / 2.0)

    def casorati_curvature(self) -> ScalarMap:
        """Casorati curvature — identical to curvedness.

        K_C = √((κ₁² + κ₂²) / 2)

        Preferred over mean/Gaussian curvature for complexity
        quantification (Matsuyama et al. 2023).

        Returns
        -------
        ndarray, shape (N,)
        """
        return self.curvedness()

    def willmore_density(self) -> ScalarMap:
        """Local Willmore energy density H²(v).

        The integral ∫ H² dA is the Willmore energy — conformally
        invariant, measures deviation from a sphere.

        Returns
        -------
        ndarray, shape (N,)
        """
        H = self.mean_curvature()
        return H**2

    def willmore_energy(self) -> float:
        """Total Willmore energy ∫ H² dA.

        Returns
        -------
        float
        """
        w = self.willmore_density()
        areas = _voronoi_areas(self.vertices, self.faces)
        return float(np.sum(w * areas))

    # ── Geodesic distances ────────────────────────────────────────────

    def geodesic_distance(
        self,
        source_indices: np.ndarray,
        *,
        method: Literal["heat", "dijkstra"] = "heat",
        t_factor: float = 1.0,
    ) -> ScalarMap:
        """Geodesic distance from source vertices to all others.

        Parameters
        ----------
        source_indices : ndarray
            Indices of source vertices.
        method : str
            ``"heat"`` — Crane et al. heat method (smooth, O(N)).
            ``"dijkstra"`` — graph shortest path on edge graph
            (exact on graph, O(N log N)).
        t_factor : float
            For the heat method: time-step multiplier relative to
            the squared mean edge length.

        Returns
        -------
        distances : ndarray, shape (N,)
            Geodesic distance from the closest source vertex.
        """
        if method == "heat":
            return _geodesic_heat(
                self.vertices,
                self.faces,
                source_indices,
                L=self._L,
                M=self._M,
                t_factor=t_factor,
            )
        elif method == "dijkstra":
            return _geodesic_dijkstra(
                self.vertices,
                self.faces,
                source_indices,
            )
        else:
            raise ValueError(f"Unknown geodesic method: {method!r}")

    # ── Topology ──────────────────────────────────────────────────────

    def euler_characteristic(self) -> int:
        """Euler characteristic χ = V − E + F.

        Returns
        -------
        int
        """
        V = self.n_vertices
        F = self.n_faces
        E = _count_edges(self.faces)
        return V - E + F

    def genus(self) -> int:
        """Genus g from χ = 2 − 2g (closed surfaces).

        Returns
        -------
        int
        """
        chi = self.euler_characteristic()
        return (2 - chi) // 2

    def boundary_vertices(self) -> np.ndarray:
        """Indices of boundary (non-manifold edge) vertices.

        Returns
        -------
        ndarray of int
            Vertex indices on the mesh boundary.
        """
        return _boundary_vertices(self.faces, self.n_vertices)

    def is_closed(self) -> bool:
        """True if the mesh has no boundary."""
        return len(self.boundary_vertices()) == 0

    # ── Quality metrics ───────────────────────────────────────────────

    def edge_lengths(self) -> np.ndarray:
        """All edge lengths.

        Returns
        -------
        ndarray, shape (E,)
        """
        return _edge_lengths(self.vertices, self.faces)

    def vertex_valence(self) -> np.ndarray:
        """Number of edges incident to each vertex.

        Returns
        -------
        ndarray, shape (N,), int
        """
        return _vertex_valence(self.faces, self.n_vertices)

    def quality_report(self) -> dict[str, Any]:
        """Mesh quality summary.

        Returns
        -------
        dict
            Keys: n_vertices, n_faces, n_edges, euler, genus,
            is_closed, area, edge_length_{min,mean,max,std},
            valence_{min,mean,max}, n_boundary_vertices.
        """
        el = self.edge_lengths()
        vv = self.vertex_valence()
        bv = self.boundary_vertices()
        return {
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_edges": _count_edges(self.faces),
            "euler_characteristic": self.euler_characteristic(),
            "genus": self.genus(),
            "is_closed": self.is_closed(),
            "surface_area": self.surface_area(),
            "edge_length_min": float(el.min()),
            "edge_length_mean": float(el.mean()),
            "edge_length_max": float(el.max()),
            "edge_length_std": float(el.std()),
            "valence_min": int(vv.min()),
            "valence_mean": float(vv.mean()),
            "valence_max": int(vv.max()),
            "n_boundary_vertices": len(bv),
        }

    # ── Smoothing ─────────────────────────────────────────────────────

    def laplacian_smooth(
        self,
        n_iterations: int = 10,
        step_size: float = 0.5,
    ) -> BrainMesh:
        """Laplacian smoothing (uniform weights).

        Parameters
        ----------
        n_iterations : int
        step_size : float
            Damping factor (0–1).

        Returns
        -------
        BrainMesh
            New mesh with smoothed vertices.
        """
        verts = _laplacian_smooth(
            self.vertices,
            self.faces,
            n_iterations,
            step_size,
        )
        return BrainMesh(verts, self.faces.copy(), metadata=dict(self.metadata))

    def taubin_smooth(
        self,
        n_iterations: int = 10,
        lambda_: float = 0.5,
        mu: float = -0.53,
    ) -> BrainMesh:
        """Taubin smoothing (shrinkage-free).

        Alternates positive and negative smoothing steps to avoid
        the volume shrinkage of pure Laplacian smoothing.

        Parameters
        ----------
        n_iterations : int
            Number of (λ, μ) cycles.
        lambda_ : float
            Positive smoothing factor.
        mu : float
            Negative (inflation) factor. Must satisfy
            ``mu < -lambda_`` for shrinkage-free behaviour.

        Returns
        -------
        BrainMesh
        """
        verts = self.vertices.copy()
        with progress_simple("Taubin smoothing", total=n_iterations) as tick:
            for _ in range(n_iterations):
                verts = _laplacian_step(verts, self.faces, lambda_)
                verts = _laplacian_step(verts, self.faces, mu)
                tick(1)
        return BrainMesh(verts, self.faces.copy(), metadata=dict(self.metadata))

    # ── Vertex areas ──────────────────────────────────────────────────

    def vertex_areas(
        self,
        method: Literal["voronoi", "barycentric"] = "barycentric",
    ) -> ScalarMap:
        """Per-vertex area (Voronoi or barycentric lumped).

        Parameters
        ----------
        method : str
            ``"barycentric"`` — A(v) = Σ_{t∈star(v)} area(t) / 3.
            ``"voronoi"`` — mixed Voronoi–barycentric (Meyer et al.).

        Returns
        -------
        ndarray, shape (N,)
        """
        if method == "barycentric":
            return _barycentric_vertex_areas(self.vertices, self.faces)
        elif method == "voronoi":
            return _voronoi_areas(self.vertices, self.faces)
        raise ValueError(f"Unknown method: {method!r}")

    # ── repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a compact mesh summary string."""
        s = self.metadata.get("structure", "")
        label = f", structure='{s}'" if s else ""
        return (
            f"BrainMesh(n_vertices={self.n_vertices}, "
            f"n_faces={self.n_faces}, "
            f"area={self.surface_area():.1f}{label})"
            """Return a compact mesh summary."""
        )


# ======================================================================
# §2  COTANGENT LAPLACIAN
# ======================================================================


def _cotangent_laplacian(
    vertices: Vertices,
    faces: Faces,
) -> tuple[SparseMatrix, MassMatrix]:
    """Build the FEM cotangent-weighted Laplacian and lumped mass matrix.

    Implementation follows Meyer, Desbrun, Schröder & Barr (2003).
    Uses vectorised operations — no Python loops over faces.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    L : csc_matrix, shape (N, N)
        Stiffness matrix (positive semi-definite, zero row-sum).
    M : csc_matrix, shape (N, N)
        Diagonal mass matrix (barycentric lumped).
    """
    N = vertices.shape[0]
    faces = _drop_degenerate_faces(vertices, faces)

    # Three edges per triangle: opposite to vertex 0, 1, 2.
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

    # Edge vectors.
    e0 = v2 - v1  # opposite vertex 0
    e1 = v0 - v2  # opposite vertex 1
    e2 = v1 - v0  # opposite vertex 2

    # Cotangent of angle at vertex i = dot(e_j, e_k) / |cross(e_j, e_k)|
    # where e_j, e_k are the two edges meeting at vertex i.

    # Areas via cross product (also needed for mass matrix).
    cross_01 = np.cross(e0, e1)
    area2 = np.linalg.norm(cross_01, axis=1)  # 2 × area (> 0 after filtering)

    # Cotangent weights for each edge.
    # Edge (i1, i2) is opposite vertex 0 → cot(angle at v0)
    #   cot(α₀) = dot(e1, e2) / |cross(e1, e2)|
    #   but cross(e1, e2) = cross(v0-v2, v1-v0) which has same magnitude
    #   as cross(e0, e1) = cross(v2-v1, v0-v2).
    # Actually, use: cot(angle_at_vi) = dot(ej, ek) / area2
    #   where ej, ek are edges meeting at vi with appropriate signs.

    # Angle at vertex 0: edges e2 = v1-v0 and -e1 = v2-v0
    cot0 = np.sum((-e1) * e2, axis=1) / area2
    # Angle at vertex 1: edges e0 = v2-v1 and -e2 = v0-v1
    cot1 = np.sum((-e2) * e0, axis=1) / area2
    # Angle at vertex 2: edges e1 = v0-v2 and -e0 = v1-v2
    cot2 = np.sum((-e0) * e1, axis=1) / area2

    # Build L as COO: edge (i,j) gets weight (cot_opposite / 2).
    # Edge (i1, i2) → weight cot0 / 2
    # Edge (i2, i0) → weight cot1 / 2
    # Edge (i0, i1) → weight cot2 / 2
    rows = np.concatenate([i1, i2, i2, i0, i0, i1])
    cols = np.concatenate([i2, i1, i0, i2, i1, i0])
    vals = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]) * 0.5

    # Off-diagonal entries (negative in the convention L = D - W).
    L_off = sp.coo_matrix((-vals, (rows, cols)), shape=(N, N))

    # Diagonal: sum of each row → degree.
    L = L_off.tocsc()
    diag_vals = -np.asarray(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag_vals, 0, shape=(N, N), format="csc")

    # Mass matrix: barycentric lumped (A_triangle / 3 per vertex).
    tri_areas = area2 / 2.0  # true area
    mass_vals = np.zeros(N, dtype=np.float64)
    np.add.at(mass_vals, i0, tri_areas / 3.0)
    np.add.at(mass_vals, i1, tri_areas / 3.0)
    np.add.at(mass_vals, i2, tri_areas / 3.0)
    n_isolated = int(np.sum(mass_vals <= 0))
    if n_isolated:
        logger.warning(
            "%d vertices are not referenced by any (non-degenerate) face; they "
            "decouple from the mesh and produce spurious zero eigenvalues. "
            "Remove unreferenced vertices before decomposing.",
            n_isolated,
        )
    mass_vals = np.clip(mass_vals, 1e-20, None)

    M = sp.diags(mass_vals, 0, shape=(N, N), format="csc")

    return L, M


def _drop_degenerate_faces(
    vertices: Vertices,
    faces: Faces,
    *,
    rel_tol: float = 1e-12,
) -> Faces:
    """Remove zero-area / repeated-index triangles (with a warning).

    Degenerate triangles make the cotangent weights blow up (≈1e21 after
    clipping), which silently corrupts the spectrum.  A triangle is
    degenerate if it repeats a vertex index or if twice its area is below
    ``rel_tol × (longest edge)²``.
    """
    f = np.asarray(faces, dtype=np.int64)
    if f.shape[0] == 0:
        return f
    v0, v1, v2 = vertices[f[:, 0]], vertices[f[:, 1]], vertices[f[:, 2]]
    area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    lmax2 = np.max(
        np.stack(
            [
                np.sum((v1 - v0) ** 2, axis=1),
                np.sum((v2 - v1) ** 2, axis=1),
                np.sum((v0 - v2) ** 2, axis=1),
            ]
        ),
        axis=0,
    )
    repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    bad = repeated | (area2 <= rel_tol * lmax2) | ~np.isfinite(area2)
    n_bad = int(bad.sum())
    if n_bad:
        logger.warning(
            "Ignoring %d degenerate (zero-area) triangle(s) of %d when building "
            "the cotangent Laplacian; consider cleaning the mesh or using "
            "laplacian_method='robust'.",
            n_bad,
            f.shape[0],
        )
        f = f[~bad]
    return f


def weld_mesh(
    vertices: Vertices,
    faces: Faces,
    *,
    decimals: int = 8,
) -> tuple[Vertices, Faces]:
    """Merge duplicate vertices and drop degenerate / duplicate faces.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    decimals : int
        Coordinates are compared after rounding to this many decimals.

    Returns
    -------
    vertices, faces
        Cleaned mesh; unreferenced vertices are removed.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    _, first, inv = np.unique(np.round(v, decimals), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    f = inv[f]
    v = v[first]
    repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    f = f[~repeated]
    # Drop duplicate faces (same vertex set), keeping the first occurrence.
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(keep)]
    # Drop zero-area faces and unreferenced vertices.
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    a2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    f = f[a2 > 1e-12 * max(float(np.max(a2)) if a2.size else 0.0, 1e-300)]
    used = np.unique(f)
    remap = np.full(v.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return v[used], remap[f]


# ======================================================================
# §3  ROBUST LAPLACIAN (Sharp & Crane, SGP 2020)
# ======================================================================


def _robust_laplacian_mesh(
    vertices: Vertices,
    faces: Faces,
    *,
    mollify_factor: float = 1e-5,
) -> tuple[SparseMatrix, MassMatrix]:
    """Tufted Laplacian via ``robust_laplacian`` (Sharp & Crane).

    Handles non-manifold edges, degenerate triangles, and
    unreferenced vertices gracefully.

    Parameters
    ----------
    vertices, faces : arrays
    mollify_factor : float
        Regularisation strength.

    Returns
    -------
    L, M : SparseMatrix
    """
    try:
        import robust_laplacian as rl
    except ImportError as exc:
        raise ImportError(
            "robust_laplacian is required for method='robust'.\n  pip install robust_laplacian"
        ) from exc

    L, M = rl.mesh_laplacian(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        mollify_factor=mollify_factor,
    )
    return sp.csc_matrix(L), sp.csc_matrix(M)


# ======================================================================
# §4  NORMALS
# ======================================================================


def _vertex_normals(vertices: Vertices, faces: Faces) -> Normals:
    """Area-weighted vertex normals from face topology."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)

    # Accumulate onto vertices (area-weighted: |cross| ∝ area).
    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)

    # Normalise.
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vertex_normals / norms


# ======================================================================
# §5  CURVATURE
# ======================================================================


def _gaussian_curvature(vertices: Vertices, faces: Faces) -> ScalarMap:
    """Angle-defect Gaussian curvature: K(v) = (2π - Σθ) / A(v)."""
    N = vertices.shape[0]
    angle_sum = np.zeros(N, dtype=np.float64)

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

    def _angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Angle between vectors a and b (per row)."""
        cos = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30
        )
        return np.arccos(np.clip(cos, -1.0, 1.0))

    ang0 = _angles(v1 - v0, v2 - v0)
    ang1 = _angles(v0 - v1, v2 - v1)
    ang2 = _angles(v0 - v2, v1 - v2)

    np.add.at(angle_sum, i0, ang0)
    np.add.at(angle_sum, i1, ang1)
    np.add.at(angle_sum, i2, ang2)

    areas = _voronoi_areas(vertices, faces)
    areas = np.clip(areas, 1e-20, None)

    total_angle = np.full(N, 2.0 * np.pi)
    total_angle[_boundary_vertices(faces, N)] = np.pi
    return (total_angle - angle_sum) / areas


def _mean_curvature_laplacian(
    vertices: Vertices,
    L: SparseMatrix,
    M: MassMatrix,
    faces: Faces | None = None,
) -> ScalarMap:
    """Signed mean curvature from the Laplacian: Hn = (M⁻¹ L X) / 2.

    With the PSD stiffness convention (L = −Δ), ``M⁻¹ L X = 2 H n`` where
    ``n`` is the outward normal.  The sign is ``sign(⟨M⁻¹LX, n_v⟩)`` with
    ``n_v`` the area-weighted vertex normal from *faces*; if *faces* is
    ``None`` the unsigned magnitude is returned.
    """
    M_inv_diag = 1.0 / np.clip(np.asarray(M.diagonal(), dtype=np.float64), 1e-20, None)
    Hn = M_inv_diag[:, None] * np.asarray(L @ vertices)  # (N, 3)

    H_mag = np.linalg.norm(Hn, axis=1) / 2.0
    if faces is None or len(faces) == 0:
        return H_mag

    normals = _vertex_normals(vertices, faces)
    sign = np.sign(np.sum(Hn * normals, axis=1))
    sign[sign == 0] = 1.0
    return sign * H_mag


# ======================================================================
# §6  VERTEX AREAS
# ======================================================================


def _barycentric_vertex_areas(vertices: Vertices, faces: Faces) -> ScalarMap:
    """Barycentric vertex area: A(v) = Σ area(t)/3 for t in star(v)."""
    N = vertices.shape[0]
    tri_areas = triangle_areas(vertices, faces)
    vert_areas = np.zeros(N, dtype=np.float64)
    np.add.at(vert_areas, faces[:, 0], tri_areas / 3.0)
    np.add.at(vert_areas, faces[:, 1], tri_areas / 3.0)
    np.add.at(vert_areas, faces[:, 2], tri_areas / 3.0)
    return vert_areas


def _voronoi_areas(vertices: Vertices, faces: Faces) -> ScalarMap:
    """Mixed Voronoi–barycentric vertex areas (Meyer et al. 2003).

    For each non-obtuse triangle, vertex *i* receives its Voronoi share
    ``(|e_ij|² cot θ_k + |e_ik|² cot θ_j) / 8``.  For an obtuse triangle the
    obtuse vertex receives ``area/2`` and the other two ``area/4``.  The
    areas sum exactly to the total surface area.
    """
    N = vertices.shape[0]
    f = np.asarray(faces, dtype=np.int64)
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]

    # Edge vectors opposite each vertex and squared lengths.
    e0, e1, e2 = p2 - p1, p0 - p2, p1 - p0
    l0, l1, l2 = (np.sum(e * e, axis=1) for e in (e0, e1, e2))
    area2 = np.linalg.norm(np.cross(e2, -e1), axis=1)
    area = area2 / 2.0
    safe2 = np.where(area2 > 0, area2, 1.0)

    # Cotangents of the angles at each vertex.
    cot0 = np.sum(e2 * -e1, axis=1) / safe2
    cot1 = np.sum(e0 * -e2, axis=1) / safe2
    cot2 = np.sum(e1 * -e0, axis=1) / safe2

    # Voronoi shares (edge i-j is opposite vertex k).
    vor0 = (l2 * cot2 + l1 * cot1) / 8.0
    vor1 = (l0 * cot0 + l2 * cot2) / 8.0
    vor2 = (l1 * cot1 + l0 * cot0) / 8.0

    # Obtuse checks: angle at vertex i obtuse ⇔ dot of its two edges < 0.
    obt0 = np.sum(e2 * -e1, axis=1) < 0
    obt1 = np.sum(e0 * -e2, axis=1) < 0
    obt2 = np.sum(e1 * -e0, axis=1) < 0
    any_obt = obt0 | obt1 | obt2

    a0 = np.where(any_obt, np.where(obt0, area / 2.0, area / 4.0), vor0)
    a1 = np.where(any_obt, np.where(obt1, area / 2.0, area / 4.0), vor1)
    a2 = np.where(any_obt, np.where(obt2, area / 2.0, area / 4.0), vor2)
    degenerate = area2 <= 0
    a0[degenerate] = a1[degenerate] = a2[degenerate] = 0.0

    out = np.zeros(N, dtype=np.float64)
    np.add.at(out, i0, a0)
    np.add.at(out, i1, a1)
    np.add.at(out, i2, a2)
    return out


# ======================================================================
# §7  GEODESIC DISTANCE
# ======================================================================


def _geodesic_heat(
    vertices: Vertices,
    faces: Faces,
    source_indices: np.ndarray,
    *,
    L: SparseMatrix | None = None,
    M: MassMatrix | None = None,
    t_factor: float = 1.0,
) -> ScalarMap:
    """Geodesic distance via the heat method (Crane, Weischedel, Wardetzky 2013).

    Steps: (1) diffuse a delta at sources, (2) normalise gradients,
    (3) solve Poisson for the distance function.
    """
    if L is None or M is None:
        L, M = _cotangent_laplacian(vertices, faces)

    N = vertices.shape[0]
    el = _edge_lengths(vertices, faces)
    h = float(el.mean())
    t = t_factor * h**2

    # Step 1: solve (M + t·L) u = δ_sources
    A = sp.csc_matrix(M + t * L)
    rhs = np.zeros(N, dtype=np.float64)
    source_indices = np.atleast_1d(source_indices)
    rhs[source_indices] = 1.0

    from scipy.sparse.linalg import spsolve

    u = spsolve(A, rhs)

    # Step 2: compute normalised gradient of u.
    # For each face, grad_u = Σ_i u_i (N × e_opp_i) / (2·area)
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

    # Face normals.
    fn = np.cross(v1 - v0, v2 - v0)  # (F, 3)
    area2 = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_unit = fn / np.clip(area2, 1e-20, None)

    # Gradient per face.
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    grad_u = (
        u[i0, None] * np.cross(fn_unit, e0)
        + u[i1, None] * np.cross(fn_unit, e1)
        + u[i2, None] * np.cross(fn_unit, e2)
    ) / np.clip(area2, 1e-20, None)

    # Normalise to unit length (negate for descent direction).
    grad_norm = np.linalg.norm(grad_u, axis=1, keepdims=True)
    X = -grad_u / np.clip(grad_norm, 1e-20, None)

    # Step 3: integrated divergence → solve L φ = div(X).
    # div_i = Σ_t A_t ⟨∇φ_i, X_t⟩ = ½ Σ_t ⟨X_t, n_t × e_opp(i)⟩.
    div = np.zeros(N, dtype=np.float64)
    np.add.at(div, i0, 0.5 * np.sum(X * np.cross(fn_unit, e0), axis=1))
    np.add.at(div, i1, 0.5 * np.sum(X * np.cross(fn_unit, e1), axis=1))
    np.add.at(div, i2, 0.5 * np.sum(X * np.cross(fn_unit, e2), axis=1))

    # L is singular (constant null space); a tiny mass shift makes the
    # Poisson solve well-posed without changing the solution's gradient.
    eps = 1e-10 * float(abs(L).max()) / max(float(M.diagonal().max()), 1e-300)
    phi = spsolve(sp.csc_matrix(L + eps * M), div)
    phi -= phi[source_indices].mean()  # shift so source = 0

    return np.abs(phi)


def _geodesic_dijkstra(
    vertices: Vertices,
    faces: Faces,
    source_indices: np.ndarray,
) -> ScalarMap:
    """Geodesic distance via Dijkstra on the edge graph."""
    from scipy.sparse.csgraph import dijkstra

    # Build weighted adjacency.
    edges = _edge_list(faces)
    N = vertices.shape[0]
    lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        axis=1,
    )
    W = sp.csr_matrix(
        (lengths, (edges[:, 0], edges[:, 1])),
        shape=(N, N),
    )
    W = W.maximum(W.T)  # symmetrise

    dists = dijkstra(W, indices=source_indices, min_only=True)
    return dists


# ======================================================================
# §8  EDGE AND TOPOLOGY UTILITIES
# ======================================================================


def _edge_list(faces: Faces) -> np.ndarray:
    """Unique undirected edge list, shape (E, 2)."""
    all_edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    # Sort each edge so (min, max).
    sorted_edges = np.sort(all_edges, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)
    return unique_edges


def _count_edges(faces: Faces) -> int:
    """Count the number of unique edges in the mesh."""
    return _edge_list(faces).shape[0]


def _edge_lengths(vertices: Vertices, faces: Faces) -> np.ndarray:
    """Compute lengths of all edges in the mesh."""
    edges = _edge_list(faces)
    return np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        axis=1,
    )


def _vertex_valence(faces: Faces, n_vertices: int) -> np.ndarray:
    """Count edges incident to each vertex."""
    edges = _edge_list(faces)
    valence = np.zeros(n_vertices, dtype=np.int64)
    np.add.at(valence, edges[:, 0], 1)
    np.add.at(valence, edges[:, 1], 1)
    return valence


def _boundary_vertices(faces: Faces, n_vertices: int) -> np.ndarray:
    """Find vertices on boundary (non-manifold) edges."""
    all_edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    sorted_edges = np.sort(all_edges, axis=1)
    # Boundary edges appear exactly once (interior edges twice).
    _, _counts = np.unique(sorted_edges, axis=0, return_counts=True)
    np.zeros(sorted_edges.shape[0], dtype=bool)

    # Re-derive the unique mapping.
    _, inv, cnts = np.unique(
        sorted_edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    boundary_edge_ids = np.where(cnts == 1)[0]
    boundary_edge_mask = np.isin(inv, boundary_edge_ids)
    boundary_edges = sorted_edges[boundary_edge_mask]
    return np.unique(boundary_edges.ravel())


# ======================================================================
# §9  LAPLACIAN SMOOTHING
# ======================================================================


def _laplacian_step(
    vertices: Vertices,
    faces: Faces,
    factor: float,
) -> Vertices:
    """One step of uniform Laplacian smoothing."""
    N = vertices.shape[0]
    adj = sp.lil_matrix((N, N), dtype=np.float64)

    edges = _edge_list(faces)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    adj = adj.tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    degree = np.clip(degree, 1, None)

    # Laplacian displacement: L(v) = (1/deg) Σ_j (v_j - v_i)
    avg = (adj @ vertices) / degree[:, None]
    displacement = avg - vertices
    return vertices + factor * displacement


def _laplacian_smooth(
    vertices: Vertices,
    faces: Faces,
    n_iterations: int,
    step_size: float,
) -> Vertices:
    """Iterative Laplacian smoothing with progress."""
    verts = vertices.copy()
    with progress_simple("Laplacian smoothing", total=n_iterations) as tick:
        for _ in range(n_iterations):
            verts = _laplacian_step(verts, faces, step_size)
            tick(1)
    return verts


# ======================================================================
# §10  __all__
# ======================================================================

__all__: list[str] = [
    "BrainMesh",
    "weld_mesh",
]
