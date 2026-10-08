"""Core geometric objects and shared processing functions.

This module defines:

- :class:`SpectralDecomposition` -- the central object of the library,
  holding eigenvalues/eigenvectors and providing access to all
  spectral descriptors.
- :class:`GeometricObject` -- the structural protocol that
  :class:`~spectralbrain.core.meshes.BrainMesh` implements. Any external
  geometry type that satisfies it (e.g. the geometry objects of the sibling
  library ``pointsbrain``) can feed the same spectral pipeline.
- Shared geometric functions that operate on raw ``(N, 3)`` vertex
  coordinate arrays.

Design principle
----------------
SpectralBrain's data flow is:

    io.loaders -> core.meshes -> **core.base.SpectralDecomposition** -> spectral.*

The SpectralDecomposition is what ``core/`` produces and ``spectral/``
consumes.  It is the single handoff point between geometry and
analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import directed_hausdorff

from spectralbrain.runtime import (
    Eigenvalues,
    Eigenvectors,
    Faces,
    GlobalDescriptor,
    MassMatrix,
    Normals,
    PathLike,
    Points,
    SparseMatrix,
    Vertices,
    get_logger,
)

logger = get_logger(__name__)


# ======================================================================
# S1  SPECTRAL DECOMPOSITION -- the central object
# ======================================================================


class SpectralDecomposition:
    """Eigenvalues and eigenvectors of a Laplace-Beltrami operator.

    This is the **central object** of SpectralBrain.  It is produced
    by ``core.meshes.BrainMesh.decompose()`` (or by any operator in
    :mod:`spectralbrain.spectral.operators`) and consumed by every
    function in ``spectral/``.

    Parameters
    ----------
    eigenvalues : ndarray, shape (k,)
        Non-negative LBO eigenvalues, sorted ascending.
    eigenvectors : ndarray, shape (N, k)
        Corresponding eigenvectors, M-orthonormal.
    stiffness : SparseMatrix, optional
        The Laplacian matrix L.
    mass : MassMatrix, optional
        The mass matrix M.
    surface_area : float, optional
        Total surface area (for eigenvalue normalisation).
    metadata : dict, optional
        Provenance info (subject, hemisphere, structure, backend, ...).

    Attributes
    ----------
    eigenvalues : ndarray
    eigenvectors : ndarray
    stiffness : SparseMatrix or None
    mass : MassMatrix or None
    surface_area : float or None
    metadata : dict

    Examples
    --------
    >>> mesh = sb.core.BrainMesh(vertices, faces)
    >>> decomp = mesh.decompose(k=100)
    >>> decomp.n_eigenvalues
    100
    >>> decomp.fiedler_value
    0.0023
    >>> decomp.truncate(50).n_eigenvalues
    50

    >>> # Pass to spectral descriptors:
    >>> hks = sb.spectral.compute_hks(decomp, t_values)
    """

    def __init__(
        self,
        eigenvalues: Eigenvalues,
        eigenvectors: Eigenvectors,
        *,
        stiffness: SparseMatrix | None = None,
        mass: MassMatrix | None = None,
        surface_area: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialise from eigenvalues, eigenvectors, and optional matrices."""
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors, dtype=np.float64)

        if eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues must be 1-D, got shape {eigenvalues.shape}")
        if eigenvectors.ndim != 2:
            raise ValueError(f"eigenvectors must be 2-D, got shape {eigenvectors.shape}")
        if eigenvectors.shape[1] != eigenvalues.shape[0]:
            raise ValueError(
                f"eigenvectors columns ({eigenvectors.shape[1]}) != "
                f"eigenvalues length ({eigenvalues.shape[0]})"
            )

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.stiffness = stiffness
        self.mass = mass
        self.surface_area = surface_area
        self.metadata = metadata or {}

    # -- properties ----------------------------------------------------

    @property
    def n_vertices(self) -> int:
        """Number of vertices/points."""
        return self.eigenvectors.shape[0]

    @property
    def n_eigenvalues(self) -> int:
        """Number of stored eigenpairs."""
        return self.eigenvalues.shape[0]

    @property
    def k(self) -> int:
        """Alias for :attr:`n_eigenvalues`."""
        return self.n_eigenvalues

    @property
    def fiedler_value(self) -> float:
        """First non-trivial eigenvalue lambda_1 (the Fiedler value).

        Encodes global connectivity of the shape.  Larger values
        indicate tighter geometry.
        """
        if self.n_eigenvalues < 2:
            return 0.0
        return float(self.eigenvalues[1])

    @property
    def spectral_gap(self) -> float:
        """Gap between lambda_1 and lambda_2."""
        if self.n_eigenvalues < 3:
            return 0.0
        return float(self.eigenvalues[2] - self.eigenvalues[1])

    @property
    def shape_dna(self) -> GlobalDescriptor:
        """ShapeDNA -- the raw eigenvalue sequence (excluding lambda_0 ~ 0).

        Returns
        -------
        ndarray, shape (k-1,)
        """
        return self.eigenvalues[1:]

    @property
    def shape_dna_normalized(self) -> GlobalDescriptor:
        """Area-normalised ShapeDNA.

        Eigenvalues scaled by surface area so that shapes of
        different sizes are comparable.

        Returns
        -------
        ndarray, shape (k-1,)

        Raises
        ------
        ValueError
            If surface_area is not set.
        """
        if self.surface_area is None or self.surface_area <= 0:
            raise ValueError(
                "surface_area must be set for normalised ShapeDNA.  "
                "Pass it to SpectralDecomposition() or call "
                "normalize_eigenvalues()."
            )
        return self.eigenvalues[1:] * self.surface_area

    # -- manipulation --------------------------------------------------

    def truncate(self, k: int) -> SpectralDecomposition:
        """Return a copy with only the first *k* eigenpairs.

        Parameters
        ----------
        k : int
            Number of eigenpairs to keep (must be <= current k).

        Returns
        -------
        SpectralDecomposition
        """
        if k > self.n_eigenvalues:
            raise ValueError(
                f"Cannot truncate to k={k}, only have {self.n_eigenvalues} eigenpairs."
            )
        return SpectralDecomposition(
            eigenvalues=self.eigenvalues[:k].copy(),
            eigenvectors=self.eigenvectors[:, :k].copy(),
            stiffness=self.stiffness,
            mass=self.mass,
            surface_area=self.surface_area,
            metadata={**self.metadata, "truncated_from": self.n_eigenvalues},
        )

    def normalize_eigenvalues(
        self,
        method: Literal["area", "volume", "fiedler"] = "area",
        area: float | None = None,
        volume: float | None = None,
    ) -> SpectralDecomposition:
        """Return a copy with normalised eigenvalues.

        Parameters
        ----------
        method : str
            ``"area"`` -- multiply by surface area (Reuter convention).
            ``"volume"`` -- multiply by volume^{2/3}.
            ``"fiedler"`` -- divide by lambda_1.
        area : float, optional
            Override surface area.
        volume : float, optional
            Override volume.

        Returns
        -------
        SpectralDecomposition
        """
        evals = self.eigenvalues.copy()
        sa = area or self.surface_area

        if method == "area":
            if sa is None or sa <= 0:
                raise ValueError("Surface area required for area normalisation.")
            evals *= sa
        elif method == "volume":
            if volume is None or volume <= 0:
                raise ValueError("Volume required for volume normalisation.")
            evals *= volume ** (2 / 3)
        elif method == "fiedler":
            if evals[1] <= 0:
                raise ValueError("Fiedler value is zero -- cannot normalise.")
            evals /= evals[1]
        else:
            raise ValueError(f"Unknown normalisation method: {method!r}")

        return SpectralDecomposition(
            eigenvalues=evals,
            eigenvectors=self.eigenvectors,
            stiffness=self.stiffness,
            mass=self.mass,
            surface_area=sa,
            metadata={**self.metadata, "eigenvalue_normalisation": method},
        )

    # -- persistence ---------------------------------------------------

    def save(self, path: PathLike, **kwargs: Any) -> Path:
        """Save to HDF5 via :func:`spectralbrain.io.export.save_hdf5`.

        Parameters
        ----------
        path : PathLike
            Output ``.h5`` file.

        Returns
        -------
        Path
        """
        from spectralbrain.io.export import save_hdf5

        return save_hdf5(
            path,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            metadata={
                **(self.metadata or {}),
                "surface_area": self.surface_area or 0.0,
            },
            **kwargs,
        )

    @classmethod
    def load(cls, path: PathLike) -> SpectralDecomposition:
        """Load from an HDF5 cache file.

        Parameters
        ----------
        path : PathLike

        Returns
        -------
        SpectralDecomposition
        """
        from spectralbrain.io.export import load_hdf5

        data = load_hdf5(path)
        meta = data.get("metadata", {})
        sa = meta.pop("surface_area", None)
        if sa == 0.0:
            sa = None
        return cls(
            eigenvalues=data["eigenvalues"],
            eigenvectors=data["eigenvectors"],
            surface_area=sa,
            metadata=meta,
        )

    # -- repr ----------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact summary of the decomposition."""
        parts = [f"SpectralDecomposition(n_vertices={self.n_vertices}, k={self.n_eigenvalues}"]
        if self.surface_area is not None:
            """Return a compact representation of the decomposition."""
            parts.append(f", area={self.surface_area:.1f}")
        struct = self.metadata.get("structure")
        if struct:
            parts.append(f", structure='{struct}'")
        parts.append(")")
        return "".join(parts)


# ======================================================================
# S2  GEOMETRIC OBJECT PROTOCOL
# ======================================================================


@runtime_checkable
class GeometricObject(Protocol):
    """Structural protocol implemented by :class:`BrainMesh`.

    SpectralBrain itself handles triangle meshes only; the protocol is
    kept public so that external geometry types (for instance those of the
    sibling library ``pointsbrain``) can plug into the same
    ``decompose() -> SpectralDecomposition`` pipeline.
    """

    @property
    def coordinates(self) -> np.ndarray:
        """Vertex/point coordinates, shape ``(N, 3)``."""
        ...

    @property
    def n_points(self) -> int:
        """Number of vertices/points."""
        ...

    def decompose(self, k: int = 100, **kwargs: Any) -> SpectralDecomposition:
        """Compute the spectral decomposition."""
        ...

    def compute_normals(self) -> Normals:
        """Estimate per-vertex/point normals."""
        ...

    def surface_area(self) -> float:
        """Total surface area."""
        ...


# ======================================================================
# S3  SHARED GEOMETRIC FUNCTIONS
# ======================================================================
# All functions operate on (N, 3) vertex-coordinate arrays.


# -- Basic geometry ----------------------------------------------------


def compute_centroid(points: Points) -> np.ndarray:
    """Compute the centroid (center of mass) of a point set.

    Parameters
    ----------
    points : ndarray, shape (N, 3)

    Returns
    -------
    ndarray, shape (3,)
    """
    return np.mean(points, axis=0)


def compute_bounding_box(
    points: Points,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned bounding box.

    Parameters
    ----------
    points : ndarray, shape (N, 3)

    Returns
    -------
    bb_min : ndarray, shape (3,)
    bb_max : ndarray, shape (3,)
    extent : ndarray, shape (3,)
        ``bb_max - bb_min``.
    """
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)
    return bb_min, bb_max, bb_max - bb_min


def compute_pca_axes(
    points: Points,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of a point set -- principal axes and explained variance.

    Parameters
    ----------
    points : ndarray, shape (N, 3)

    Returns
    -------
    components : ndarray, shape (3, 3)
        Rows are the principal directions, sorted by decreasing
        variance.
    explained_variance : ndarray, shape (3,)
        Eigenvalues of the covariance matrix (variance along
        each axis).
    centroid : ndarray, shape (3,)
    """
    centroid = compute_centroid(points)
    centered = points - centroid  # (N, 3)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)  # (3, 3)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]  # descending
    return eigvecs[:, order].T, eigvals[order], centroid


# -- Normalisation / alignment -----------------------------------------


def center_points(points: Points) -> Points:
    """Translate points so the centroid is at the origin.

    Parameters
    ----------
    points : ndarray, shape (N, 3)

    Returns
    -------
    ndarray, shape (N, 3)
    """
    return points - compute_centroid(points)


def normalize_scale(
    points: Points,
    method: Literal["bbox", "rms", "area"] = "bbox",
    *,
    area: float | None = None,
) -> tuple[Points, float]:
    """Scale points to unit bounding-box diagonal, unit RMS, or unit area.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    method : str
        ``"bbox"`` -- divide by bounding-box diagonal.
        ``"rms"`` -- divide by RMS distance from centroid.
        ``"area"`` -- divide by sqrt(surface_area).
    area : float, optional
        Surface area (required for ``method="area"``).

    Returns
    -------
    scaled : ndarray, shape (N, 3)
    scale_factor : float
        The divisor applied.
    """
    centered = center_points(points)

    if method == "bbox":
        _, _, extent = compute_bounding_box(centered)
        sf = np.linalg.norm(extent)
    elif method == "rms":
        sf = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    elif method == "area":
        if area is None or area <= 0:
            raise ValueError("Surface area required for area normalisation.")
        sf = np.sqrt(area)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if sf < 1e-12:
        logger.warning("Scale factor near zero (%.2e); returning unscaled.", sf)
        return centered, 1.0

    return centered / sf, float(sf)


def align_to_pca(points: Points) -> tuple[Points, np.ndarray]:
    """Align points so principal axes coincide with coordinate axes.

    Parameters
    ----------
    points : ndarray, shape (N, 3)

    Returns
    -------
    aligned : ndarray, shape (N, 3)
    rotation : ndarray, shape (3, 3)
        Rotation matrix applied (rows = new axes).
    """
    components, _, centroid = compute_pca_axes(points)
    centered = points - centroid
    aligned = centered @ components.T  # (N, 3)
    return aligned, components


def procrustes_align(
    source: Points,
    target: Points,
) -> tuple[Points, np.ndarray, float]:
    """Rigid + uniform scale alignment (Procrustes).

    Finds the rotation R, translation t, and scale s that minimise
    ||s*R*source + t - target||^2.

    Parameters
    ----------
    source : ndarray, shape (N, 3)
        Points to align.
    target : ndarray, shape (N, 3)
        Reference points (must have the same N).

    Returns
    -------
    aligned : ndarray, shape (N, 3)
        Transformed *source*.
    rotation : ndarray, shape (3, 3)
        Rotation matrix.
    scale : float
        Uniform scale factor.

    Raises
    ------
    ValueError
        If source and target have different shapes.
    """
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: source {source.shape} vs target {target.shape}")

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src = source - mu_s
    tgt = target - mu_t

    # Optimal rotation via SVD of cross-covariance.
    H = src.T @ tgt  # (3, 3)
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])  # fix reflection
    R = Vt.T @ D @ U.T  # (3, 3)

    # Optimal scale.
    scale = float(np.trace(R @ H) / np.trace(src.T @ src))

    aligned = scale * (src @ R.T) + mu_t
    return aligned, R, scale


# -- Neighbourhood queries ---------------------------------------------


def knn_search(
    points: Points,
    k: int = 20,
    *,
    query_points: Points | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """k-nearest-neighbour search using a KD-tree.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Reference point set.
    k : int
        Number of neighbours.
    query_points : ndarray, shape (M, 3), optional
        Query points.  Defaults to *points* (self-query).

    Returns
    -------
    distances : ndarray, shape (M, k)
    indices : ndarray, shape (M, k)
    """
    tree = cKDTree(points)
    q = query_points if query_points is not None else points
    distances, indices = tree.query(q, k=k, workers=-1)
    return distances, indices


def radius_search(
    points: Points,
    radius: float,
    *,
    query_points: Points | None = None,
) -> list[np.ndarray]:
    """Fixed-radius neighbour search.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    radius : float
        Search radius.
    query_points : ndarray, shape (M, 3), optional

    Returns
    -------
    list of ndarray
        For each query point, an array of neighbour indices.
    """
    tree = cKDTree(points)
    q = query_points if query_points is not None else points
    return tree.query_ball_point(q, r=radius, workers=-1)


# -- Shape distances ---------------------------------------------------


def hausdorff_distance(
    A: Points,
    B: Points,
    *,
    symmetric: bool = True,
) -> float:
    """Hausdorff distance between two point sets.

    Parameters
    ----------
    A, B : ndarray, shape (N, 3) and (M, 3)
    symmetric : bool
        If True, return max(d(A->B), d(B->A)).

    Returns
    -------
    float
    """
    d_ab = directed_hausdorff(A, B)[0]
    if not symmetric:
        return float(d_ab)
    d_ba = directed_hausdorff(B, A)[0]
    return float(max(d_ab, d_ba))


def chamfer_distance(A: Points, B: Points) -> float:
    """L^2 Chamfer distance between two point sets.

    Chamfer = (1/|A|) Sigma_{a in A} min_{b in B} ||a-b||^2
            + (1/|B|) Sigma_{b in B} min_{a in A} ||b-a||^2

    Parameters
    ----------
    A, B : ndarray, shape (N, 3) and (M, 3)

    Returns
    -------
    float
    """
    tree_b = cKDTree(B)
    tree_a = cKDTree(A)
    d_ab, _ = tree_b.query(A, k=1)
    d_ba, _ = tree_a.query(B, k=1)
    return float(np.mean(d_ab**2) + np.mean(d_ba**2))


# -- Volume/surface conversion -----------------------------------------


def marching_cubes(
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    level: float | None = None,
    step_size: int = 1,
) -> tuple[Vertices, Faces]:
    """Extract a mesh from a volumetric label/mask via marching cubes.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z)
        Binary mask or label volume (non-zero = inside).
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine.
    level : float, optional
        Iso-surface level.  If ``None`` (default) the volume is first
        **binarised** (``volume != 0``) and the 0.5 iso-surface of the mask
        is extracted, so label values (e.g. 17) do not shift the surface.
        If given explicitly, the iso-surface of the raw scalar volume at
        that level is extracted (intensity / probability maps).
    step_size : int
        Subsampling step for speed.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        World-space coordinates.
    faces : ndarray, shape (F, 3)
        Triangle indices, 0-indexed, wound counter-clockwise when viewed
        from outside (outward normals, positive signed volume for closed
        surfaces) for any affine, including reflecting ones (det < 0).
    """
    try:
        from skimage.measure import marching_cubes as _mc
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for marching cubes.\n  pip install scikit-image"
        ) from exc

    affine = np.asarray(affine, dtype=np.float64)
    if level is None:
        vol = (np.asarray(volume) != 0).astype(np.float64)
        level = 0.5
    else:
        vol = np.asarray(volume, dtype=np.float64)

    verts_vox, faces, _normals, _values = _mc(
        vol,
        level=level,
        step_size=step_size,
    )

    # skimage's default ``gradient_direction="descent"`` (inside = higher
    # values) yields triangles wound *inward* in voxel index space; reverse
    # them so normals point outward.  A reflecting affine (det < 0, e.g.
    # FreeSurfer LIA) flips handedness again.
    faces = faces[:, ::-1]
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]

    # Transform to world coordinates.
    ones = np.ones((verts_vox.shape[0], 1))
    verts_h = np.hstack([verts_vox, ones])  # (N, 4)
    verts_world = (affine @ verts_h.T).T[:, :3]  # (N, 3)

    return (
        np.asarray(verts_world, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )


# -- Convex hull -------------------------------------------------------


def convex_hull_volume(points: Points) -> float:
    """Convex hull volume of a point set.

    Parameters
    ----------
    points : ndarray, shape (N, 3)

    Returns
    -------
    float
        Volume in cubic mm.
    """
    hull = ConvexHull(points)
    return float(hull.volume)


def convex_hull_area(points: Points) -> float:
    """Convex hull surface area.

    Parameters
    ----------
    points : ndarray, shape (N, 3)

    Returns
    -------
    float
        Area in mm^2.
    """
    hull = ConvexHull(points)
    return float(hull.area)


# -- Triangle area helper (used by meshes.py and here) -----------------


def triangle_areas(
    vertices: Vertices,
    faces: Faces,
) -> np.ndarray:
    """Compute per-triangle areas.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    areas : ndarray, shape (F,)
        Area of each triangle in mm^2.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)  # (F, 3)
    return 0.5 * np.linalg.norm(cross, axis=1)  # (F,)


def mesh_surface_area(vertices: Vertices, faces: Faces) -> float:
    """Total surface area of a triangle mesh.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    float
        Total area in mm^2.
    """
    return float(triangle_areas(vertices, faces).sum())


# ======================================================================
# S4  __all__
# ======================================================================

__all__: list[str] = [
    "GeometricObject",
    # Central object
    "SpectralDecomposition",
    "align_to_pca",
    # Normalisation
    "center_points",
    "chamfer_distance",
    "compute_bounding_box",
    # Basic geometry
    "compute_centroid",
    "compute_pca_axes",
    "convex_hull_area",
    # Convex hull
    "convex_hull_volume",
    # Shape distances
    "hausdorff_distance",
    # Neighbourhood
    "knn_search",
    # Volume/surface conversion
    "marching_cubes",
    "mesh_surface_area",
    "normalize_scale",
    "procrustes_align",
    "radius_search",
    # Triangle helpers
    "triangle_areas",
]
