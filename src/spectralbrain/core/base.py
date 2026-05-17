"""Core geometric objects and shared processing functions.

This module defines:

- :class:`SpectralDecomposition` — the central object of the library,
  holding eigenvalues/eigenvectors and providing access to all
  spectral descriptors.
- :class:`GeometricObject` — abstract protocol that both
  :class:`BrainMesh` and :class:`BrainPointCloud` implement.
- Shared geometric functions that operate on raw ``(N, 3)``
  coordinate arrays regardless of mesh or point-cloud origin.

Design principle
----------------
SpectralBrain's data flow is:

    io.loaders → core.meshes / core.pointclouds → **core.base.SpectralDecomposition** → spectral.*

The SpectralDecomposition is what ``core/`` produces and ``spectral/``
consumes.  It is the single handoff point between geometry and
analysis.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import directed_hausdorff

from spectralbrain.runtime import (
    DescriptorMatrix,
    DistanceMatrix,
    Eigenvalues,
    Eigenvectors,
    Faces,
    GlobalDescriptor,
    MassMatrix,
    Normals,
    PathLike,
    Points,
    ScalarMap,
    SparseMatrix,
    Vertices,
    get_logger,
)

logger = get_logger(__name__)


# ======================================================================
# §1  SPECTRAL DECOMPOSITION — the central object
# ======================================================================

class SpectralDecomposition:
    """Eigenvalues and eigenvectors of a Laplace–Beltrami operator.

    This is the **central object** of SpectralBrain.  It is produced
    by ``core.meshes.BrainMesh.decompose()`` or
    ``core.pointclouds.BrainPointCloud.decompose()`` and consumed
    by every function in ``spectral/``.

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
        Provenance info (subject, hemisphere, structure, backend, …).

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
        stiffness: Optional[SparseMatrix] = None,
        mass: Optional[MassMatrix] = None,
        surface_area: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise from eigenvalues, eigenvectors, and optional matrices."""
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors, dtype=np.float64)

        if eigenvalues.ndim != 1:
            raise ValueError(
                f"eigenvalues must be 1-D, got shape {eigenvalues.shape}"
            )
        if eigenvectors.ndim != 2:
            raise ValueError(
                f"eigenvectors must be 2-D, got shape {eigenvectors.shape}"
            )
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

    # ── properties ────────────────────────────────────────────────────

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
        """First non-trivial eigenvalue λ₁ (the Fiedler value).

        Encodes global connectivity of the shape.  Larger values
        indicate tighter geometry.
        """
        if self.n_eigenvalues < 2:
            return 0.0
        return float(self.eigenvalues[1])

    @property
    def spectral_gap(self) -> float:
        """Gap between λ₁ and λ₂."""
        if self.n_eigenvalues < 3:
            return 0.0
        return float(self.eigenvalues[2] - self.eigenvalues[1])

    @property
    def shape_dna(self) -> GlobalDescriptor:
        """ShapeDNA — the raw eigenvalue sequence (excluding λ₀ ≈ 0).

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

    # ── manipulation ──────────────────────────────────────────────────

    def truncate(self, k: int) -> SpectralDecomposition:
        """Return a copy with only the first *k* eigenpairs.

        Parameters
        ----------
        k : int
            Number of eigenpairs to keep (must be ≤ current k).

        Returns
        -------
        SpectralDecomposition
        """
        if k > self.n_eigenvalues:
            raise ValueError(
                f"Cannot truncate to k={k}, only have "
                f"{self.n_eigenvalues} eigenpairs."
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
        area: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SpectralDecomposition:
        """Return a copy with normalised eigenvalues.

        Parameters
        ----------
        method : str
            ``"area"`` — multiply by surface area (Reuter convention).
            ``"volume"`` — multiply by volume^{2/3}.
            ``"fiedler"`` — divide by λ₁.
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
                raise ValueError("Fiedler value is zero — cannot normalise.")
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

    # ── persistence ───────────────────────────────────────────────────

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

    # ── repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a compact summary of the decomposition."""
        parts = [
            f"SpectralDecomposition("
            f"n_vertices={self.n_vertices}, "
            f"k={self.n_eigenvalues}"
        ]
        if self.surface_area is not None:
            """Return a compact representation of the decomposition."""
            parts.append(f", area={self.surface_area:.1f}")
        struct = self.metadata.get("structure")
        if struct:
            parts.append(f", structure='{struct}'")
        parts.append(")")
        return "".join(parts)


# ======================================================================
# §2  GEOMETRIC OBJECT PROTOCOL
# ======================================================================

@runtime_checkable
class GeometricObject(Protocol):
    """Protocol that :class:`BrainMesh` and :class:`BrainPointCloud`
    both implement.

    Any function that accepts a ``GeometricObject`` can work with
    either representation.
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
# §3  SHARED GEOMETRIC FUNCTIONS
# ======================================================================
# All functions operate on (N, 3) coordinate arrays and are agnostic
# to whether the input is a mesh or a point cloud.


# ── Basic geometry ────────────────────────────────────────────────────

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of a point set — principal axes and explained variance.

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
    centered = points - centroid                          # (N, 3)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)  # (3, 3)
    eigvals, eigvecs = np.linalg.eigh(cov)               # ascending
    order = np.argsort(eigvals)[::-1]                     # descending
    return eigvecs[:, order].T, eigvals[order], centroid


# ── Normalisation / alignment ─────────────────────────────────────────

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
    area: Optional[float] = None,
) -> Tuple[Points, float]:
    """Scale points to unit bounding-box diagonal, unit RMS, or unit area.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    method : str
        ``"bbox"`` — divide by bounding-box diagonal.
        ``"rms"`` — divide by RMS distance from centroid.
        ``"area"`` — divide by sqrt(surface_area).
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
        sf = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
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


def align_to_pca(points: Points) -> Tuple[Points, np.ndarray]:
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
    aligned = centered @ components.T                     # (N, 3)
    return aligned, components


def procrustes_align(
    source: Points,
    target: Points,
) -> Tuple[Points, np.ndarray, float]:
    """Rigid + uniform scale alignment (Procrustes).

    Finds the rotation R, translation t, and scale s that minimise
    ||s·R·source + t − target||².

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
        raise ValueError(
            f"Shape mismatch: source {source.shape} vs target {target.shape}"
        )

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src = source - mu_s
    tgt = target - mu_t

    # Optimal rotation via SVD of cross-covariance.
    H = src.T @ tgt                                       # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])                       # fix reflection
    R = Vt.T @ D @ U.T                                    # (3, 3)

    # Optimal scale.
    scale = float(
        np.trace(R @ H) / np.trace(src.T @ src)
    )

    aligned = scale * (src @ R.T) + mu_t
    return aligned, R, scale


# ── Subsampling ───────────────────────────────────────────────────────

def farthest_point_sampling(
    points: Points,
    n_samples: int,
    *,
    seed: Optional[int] = None,
) -> Tuple[Points, np.ndarray]:
    """Farthest-point sampling (FPS) for uniform subsampling.

    Iteratively selects the point farthest from the current set,
    producing an approximately uniform subset.  Essential for
    standardising point-cloud density from volumetric segmentations.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    n_samples : int
        Number of points to select.
    seed : int, optional
        RNG seed for the initial point.

    Returns
    -------
    sampled : ndarray, shape (n_samples, 3)
        Selected points.
    indices : ndarray, shape (n_samples,)
        Indices into *points*.

    Notes
    -----
    Complexity: O(N · n_samples).  For N > 100 k, consider the
    approximate FPS in Open3D or PyTorch3D.
    """
    N = points.shape[0]
    if n_samples >= N:
        return points.copy(), np.arange(N)

    rng = np.random.default_rng(seed)
    indices = np.zeros(n_samples, dtype=np.int64)
    indices[0] = rng.integers(N)

    # min_dist[i] = distance from point i to the closest selected point.
    min_dist = np.full(N, np.inf, dtype=np.float64)

    for j in range(1, n_samples):
        last = points[indices[j - 1]]                     # (3,)
        dist = np.sum((points - last) ** 2, axis=1)       # (N,)
        min_dist = np.minimum(min_dist, dist)
        indices[j] = np.argmax(min_dist)

    return points[indices], indices


# ── Neighbourhood queries ─────────────────────────────────────────────

def knn_search(
    points: Points,
    k: int = 20,
    *,
    query_points: Optional[Points] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    query_points: Optional[Points] = None,
) -> List[np.ndarray]:
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


def compute_adjacency_from_knn(
    points: Points,
    k: int = 20,
    *,
    symmetric: bool = True,
) -> SparseMatrix:
    """Build a kNN adjacency matrix (binary or weighted).

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        Number of neighbours.
    symmetric : bool
        Symmetrise the adjacency (``A = max(A, A.T)``).

    Returns
    -------
    SparseMatrix, shape (N, N)
        Binary adjacency.
    """
    import scipy.sparse as sp
    distances, indices = knn_search(points, k=k)
    N = points.shape[0]

    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()
    data = np.ones(N * k, dtype=np.float64)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    if symmetric:
        A = A.maximum(A.T)
    return A


# ── Shape distances ───────────────────────────────────────────────────

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
        If True, return max(d(A→B), d(B→A)).

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
    """L² Chamfer distance between two point sets.

    Chamfer = (1/|A|) Σ_{a∈A} min_{b∈B} ||a−b||²
            + (1/|B|) Σ_{b∈B} min_{a∈A} ||b−a||²

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
    return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))


# ── Volume/surface conversion ─────────────────────────────────────────

def marching_cubes(
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    level: Optional[float] = None,
    step_size: int = 1,
) -> Tuple[Vertices, Faces]:
    """Extract a mesh from a volumetric label/mask via marching cubes.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z)
        Binary mask or label volume (non-zero = inside).
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine.
    level : float, optional
        Iso-surface level.  Default 0.5 (for binary masks).
    step_size : int
        Subsampling step for speed.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        World-space coordinates.
    faces : ndarray, shape (F, 3)
        Triangle indices, 0-indexed.
    """
    try:
        from skimage.measure import marching_cubes as _mc
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for marching cubes.\n"
            "  pip install scikit-image"
        ) from exc

    if level is None:
        level = 0.5

    vol = volume.astype(np.float64)
    verts_vox, faces, normals, values = _mc(
        vol, level=level, step_size=step_size,
    )

    # Transform to world coordinates.
    ones = np.ones((verts_vox.shape[0], 1))
    verts_h = np.hstack([verts_vox, ones])                # (N, 4)
    verts_world = (affine @ verts_h.T).T[:, :3]           # (N, 3)

    return (
        np.asarray(verts_world, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )


# ── Convex hull ───────────────────────────────────────────────────────

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
        Area in mm².
    """
    hull = ConvexHull(points)
    return float(hull.area)


# ── Point density ─────────────────────────────────────────────────────

def estimate_point_density(
    points: Points,
    k: int = 6,
) -> ScalarMap:
    """Estimate local point density via k-NN distance.

    Density at each point is approximated as ``k / V_k`` where
    ``V_k = (4/3)π r_k³`` and ``r_k`` is the distance to the k-th
    nearest neighbour.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        Neighbour count for density estimation.

    Returns
    -------
    density : ndarray, shape (N,)
        Points per mm³ (approximate).
    """
    distances, _ = knn_search(points, k=k)
    r_k = distances[:, -1]                                # (N,) distance to k-th NN
    r_k = np.clip(r_k, 1e-10, None)                      # avoid div by zero
    volume_k = (4 / 3) * np.pi * r_k ** 3
    return k / volume_k


def detect_density_outliers(
    points: Points,
    k: int = 6,
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """Flag points with anomalously low or high local density.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        Neighbour count.
    threshold_sigma : float
        Number of standard deviations from the mean log-density
        to flag as outlier.

    Returns
    -------
    outlier_mask : ndarray, shape (N,), bool
        True for outlier points.
    """
    density = estimate_point_density(points, k=k)
    log_d = np.log(density + 1e-30)
    z = (log_d - log_d.mean()) / (log_d.std() + 1e-30)
    return np.abs(z) > threshold_sigma


# ── Triangle area helper (used by meshes.py and here) ─────────────────

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
        Area of each triangle in mm².
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)                   # (F, 3)
    return 0.5 * np.linalg.norm(cross, axis=1)            # (F,)


def mesh_surface_area(vertices: Vertices, faces: Faces) -> float:
    """Total surface area of a triangle mesh.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    float
        Total area in mm².
    """
    return float(triangle_areas(vertices, faces).sum())


# ======================================================================
# §4  __all__
# ======================================================================

__all__: List[str] = [
    # Central object
    "SpectralDecomposition",
    "GeometricObject",
    # Basic geometry
    "compute_centroid",
    "compute_bounding_box",
    "compute_pca_axes",
    # Normalisation
    "center_points",
    "normalize_scale",
    "align_to_pca",
    "procrustes_align",
    # Subsampling
    "farthest_point_sampling",
    # Neighbourhood
    "knn_search",
    "radius_search",
    "compute_adjacency_from_knn",
    # Shape distances
    "hausdorff_distance",
    "chamfer_distance",
    # Volume/surface conversion
    "marching_cubes",
    # Convex hull
    "convex_hull_volume",
    "convex_hull_area",
    # Point density
    "estimate_point_density",
    "detect_density_outliers",
    # Triangle helpers
    "triangle_areas",
    "mesh_surface_area",
]
