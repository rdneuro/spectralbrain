"""Point-cloud representation and point-cloud-specific analysis.

Provides :class:`BrainPointCloud`, the mesh-free geometric container
for brain structures extracted from volumetric segmentations,
tractography, or surface vertex sets.

Design philosophy
-----------------
Point clouds are the **atlas-free, mesh-free** pathway.  No marching
cubes, no surface reconstruction.  A volumetric segmentation label
becomes a set of (x, y, z) coordinates in world space — full stop.
The Laplace–Beltrami operator is approximated via graph Laplacians
(kNN, Belkin–Niyogi, or Sharp–Crane robust), and spectral descriptors
are computed identically to the mesh pathway.

Acquisition pathways
--------------------
.. code-block:: text

   ┌─ FreeSurfer segmentation (aseg.mgz, thalamic_nuclei) ──► voxel → RAS
   ├─ FreeSurfer surface (.white, .pial) ──────────────────► vertices as points
   ├─ HippUnfold surface (.surf.gii, 8k/18k) ─────────────► vertices as points
   ├─ Raw T1w/T2w/FLAIR (via containers) ──────────────────► segment → voxel → RAS
   ├─ Atlas label volume (Schaefer, Julich, any .nii.gz) ──► voxel → RAS
   ├─ TractSeg tract masks (.nii.gz per tract) ────────────► voxel → RAS
   └─ Tractography streamlines (.trk, .tck) ───────────────► streamline points

All converge to a single ``BrainPointCloud(points)`` that enters the
spectral pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from spectralbrain.backends.cpu import NumpyBackend
from spectralbrain.core.base import (
    GeometricObject,
    SpectralDecomposition,
    compute_centroid,
    convex_hull_area,
    estimate_point_density,
    farthest_point_sampling,
    knn_search,
)
from spectralbrain.runtime import (
    Eigenvalues,
    Eigenvectors,
    MassMatrix,
    Normals,
    PathLike,
    Points,
    ScalarMap,
    SparseMatrix,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  BRAIN POINT CLOUD CLASS
# ======================================================================

class BrainPointCloud:
    """Unstructured 3D point cloud for brain structures.

    Implements :class:`~spectralbrain.core.base.GeometricObject`.
    No face connectivity — the Laplacian is built from a neighbourhood
    graph (kNN, ε-ball, or Belkin–Niyogi).

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Coordinates in world space (mm, RAS).
    metadata : dict, optional
        Provenance (subject, hemisphere, structure, source, …).

    Examples
    --------
    >>> pc = BrainPointCloud.from_freesurfer_seg(
    ...     "aseg.mgz", label_id=17, jitter=True)
    >>> decomp = pc.decompose(k=50, method="robust")
    """

    def __init__(
        self,
        points: Points,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise from a (N, 3) coordinate array."""
        self.points = np.asarray(points, dtype=np.float64)
        self.metadata = metadata or {}

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"points must be (N, 3), got {self.points.shape}"
            )

        # Cached properties.
        self._normals: Optional[Normals] = None
        self._L: Optional[SparseMatrix] = None
        self._M: Optional[MassMatrix] = None
        self._kd_tree: Optional[cKDTree] = None

    # ── GeometricObject protocol ──────────────────────────────────────

    @property
    def coordinates(self) -> Points:
        """Return the point coordinates array."""
        return self.points

    @property
    def n_points(self) -> int:
        """Return the number of points in the cloud."""
        return self.points.shape[0]

    def surface_area(self) -> float:
        """Approximate surface area via convex hull.

        For point clouds without face topology, the convex hull area
        is a rough proxy.  Used for eigenvalue normalisation.
        """
        return convex_hull_area(self.points)

    # ── KD-tree (cached) ──────────────────────────────────────────────

    @property
    def kd_tree(self) -> cKDTree:
        """Cached spatial index."""
        if self._kd_tree is None:
            self._kd_tree = cKDTree(self.points)
        return self._kd_tree

    # ══════════════════════════════════════════════════════════════════
    # §2  FACTORY CLASSMETHODS — multi-source acquisition
    # ══════════════════════════════════════════════════════════════════

    # ── From volumetric segmentation (THE core pathway) ───────────────

    @classmethod
    def from_volume(
        cls,
        label_volume: np.ndarray,
        affine: np.ndarray,
        label_id: int,
        *,
        jitter: bool = True,
        jitter_scale: float = 0.25,
        seed: Optional[int] = None,
        subsample: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainPointCloud:
        """Extract a point cloud from a volumetric label map.

        The most direct pathway: ``mgz → np.array → point cloud``.
        No marching cubes, no mesh reconstruction.  Voxel centroids
        in world space become points.

        Parameters
        ----------
        label_volume : ndarray, shape (X, Y, Z)
            Integer label volume (e.g. from ``aseg.mgz``).
        affine : ndarray, shape (4, 4)
            Voxel-to-world affine.
        label_id : int
            Target label (e.g. 17 = left hippocampus in aseg).
        jitter : bool
            Add sub-voxel Gaussian noise to break grid artefacts.
        jitter_scale : float
            Jitter σ in voxel units.
        seed : int, optional
            RNG seed for jitter.
        subsample : int, optional
            If set, apply FPS to reduce to this many points.
        metadata : dict, optional

        Returns
        -------
        BrainPointCloud

        Examples
        --------
        >>> data, affine = sb.io.load_nifti("aseg.mgz")
        >>> pc = BrainPointCloud.from_volume(data, affine, label_id=17)
        """
        from spectralbrain.io.loaders import labels_to_pointcloud

        points = labels_to_pointcloud(
            label_volume, affine, label_id,
            jitter=jitter, jitter_scale=jitter_scale, seed=seed,
        )

        meta = {
            "source": "volume",
            "label_id": label_id,
            "n_raw_points": points.shape[0],
            **(metadata or {}),
        }

        if subsample is not None and points.shape[0] > subsample:
            points, _ = farthest_point_sampling(points, subsample, seed=seed)
            meta["subsampled_to"] = subsample

        return cls(points, metadata=meta)

    # ── From FreeSurfer segmentation file (load + extract) ────────────

    @classmethod
    def from_freesurfer_seg(
        cls,
        seg_path: PathLike,
        label_id: int,
        *,
        jitter: bool = True,
        jitter_scale: float = 0.25,
        seed: Optional[int] = None,
        subsample: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainPointCloud:
        """Load a FreeSurfer segmentation and extract a structure.

        Handles ``aseg.mgz``, ``aparc+aseg.mgz``,
        ``ThalamicNuclei.v13.T1.mgz``, ``amygNucVolumes.v22.mgz``,
        ``hippoSfVolumes-T1.v22.mgz``, or any integer label volume.

        Parameters
        ----------
        seg_path : PathLike
            Path to ``.mgz`` / ``.nii.gz`` segmentation.
        label_id : int
            Target label code.
        jitter, jitter_scale, seed, subsample : as in :meth:`from_volume`.
        metadata : dict, optional

        Returns
        -------
        BrainPointCloud
        """
        from spectralbrain.io.loaders import load_nifti

        data, affine = load_nifti(seg_path)
        meta = {
            "source": "freesurfer_seg",
            "seg_file": str(Path(seg_path).name),
            **(metadata or {}),
        }
        return cls.from_volume(
            data, affine, label_id,
            jitter=jitter, jitter_scale=jitter_scale,
            seed=seed, subsample=subsample, metadata=meta,
        )

    # ── From any surface file (vertices as points) ────────────────────

    @classmethod
    def from_surface(
        cls,
        surface_path: PathLike,
        *,
        subsample: Optional[int] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainPointCloud:
        """Load a surface file and use its vertices as a point cloud.

        Works with FreeSurfer surfaces (.white, .pial, .inflated),
        GIfTI (.surf.gii), HippUnfold outputs, or any mesh format.

        Parameters
        ----------
        surface_path : PathLike
        subsample : int, optional
            FPS subsampling target.
        seed : int, optional
        metadata : dict, optional

        Returns
        -------
        BrainPointCloud
        """
        from spectralbrain.io.loaders import load

        result = load(surface_path)
        points = result["vertices"]

        meta = {
            "source": "surface",
            "surface_file": str(Path(surface_path).name),
            "n_original_vertices": points.shape[0],
            **(metadata or {}),
        }

        if subsample is not None and points.shape[0] > subsample:
            points, _ = farthest_point_sampling(points, subsample, seed=seed)
            meta["subsampled_to"] = subsample

        return cls(points, metadata=meta)

    # ── From atlas label volume (generic: Schaefer, Julich, etc.) ─────

    @classmethod
    def from_atlas_volume(
        cls,
        atlas_path: PathLike,
        label_id: int,
        *,
        jitter: bool = True,
        seed: Optional[int] = None,
        subsample: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainPointCloud:
        """Extract a point cloud from an arbitrary atlas NIfTI.

        The atlas volume must be in the same space as the subject
        (template space or registered native).

        Parameters
        ----------
        atlas_path : PathLike
            NIfTI label volume (e.g. Schaefer in MNI).
        label_id : int
            Atlas region ID.
        jitter, seed, subsample : as in :meth:`from_volume`.
        metadata : dict, optional

        Returns
        -------
        BrainPointCloud
        """
        from spectralbrain.io.loaders import load_nifti

        data, affine = load_nifti(atlas_path)
        meta = {
            "source": "atlas_volume",
            "atlas_file": str(Path(atlas_path).name),
            **(metadata or {}),
        }
        return cls.from_volume(
            data, affine, label_id,
            jitter=jitter, seed=seed, subsample=subsample,
            metadata=meta,
        )

    # ── From TractSeg tract mask ──────────────────────────────────────

    @classmethod
    def from_tract_mask(
        cls,
        mask_path: PathLike,
        *,
        threshold: float = 0.5,
        jitter: bool = True,
        seed: Optional[int] = None,
        subsample: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainPointCloud:
        """Extract a point cloud from a TractSeg binary tract mask.

        TractSeg outputs one NIfTI per tract (72 tracts).  Each
        non-zero voxel becomes a point.

        Parameters
        ----------
        mask_path : PathLike
            Binary tract mask NIfTI (from TractSeg).
        threshold : float
            Binarisation threshold for probabilistic masks.
        jitter, seed, subsample : as in :meth:`from_volume`.
        metadata : dict, optional

        Returns
        -------
        BrainPointCloud
        """
        from spectralbrain.io.loaders import load_nifti

        data, affine = load_nifti(mask_path)
        binary = (data > threshold).astype(np.int32)
        meta = {
            "source": "tract_mask",
            "tract_file": str(Path(mask_path).name),
            **(metadata or {}),
        }
        # Use label_id=1 on the binarised mask.
        return cls.from_volume(
            binary, affine, label_id=1,
            jitter=jitter, seed=seed, subsample=subsample,
            metadata=meta,
        )

    # ── From tractography streamlines (.trk, .tck) ───────────────────

    @classmethod
    def from_tractography(
        cls,
        trk_path: PathLike,
        *,
        sampling: Literal["all", "endpoints", "midpoints", "uniform"] = "uniform",
        points_per_streamline: int = 10,
        subsample: Optional[int] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainPointCloud:
        """Extract a point cloud from tractography streamlines.

        Parameters
        ----------
        trk_path : PathLike
            ``.trk`` (TrackVis) or ``.tck`` (MRtrix) file.
        sampling : str
            ``"all"`` — every point from every streamline.
            ``"endpoints"`` — first + last point per streamline.
            ``"midpoints"`` — middle point per streamline.
            ``"uniform"`` — uniformly resample each streamline
            to *points_per_streamline* points.
        points_per_streamline : int
            For ``sampling="uniform"``: points per streamline.
        subsample : int, optional
            FPS subsampling after extraction.
        seed : int, optional
        metadata : dict, optional

        Returns
        -------
        BrainPointCloud
        """
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "nibabel is required for tractography loading.\n"
                "  pip install nibabel"
            ) from exc

        trk = nib.streamlines.load(str(trk_path))
        streamlines = trk.streamlines

        all_points: List[np.ndarray] = []

        if sampling == "all":
            for sl in streamlines:
                all_points.append(np.asarray(sl, dtype=np.float64))

        elif sampling == "endpoints":
            for sl in streamlines:
                sl = np.asarray(sl, dtype=np.float64)
                if len(sl) >= 2:
                    all_points.append(sl[[0, -1]])
                elif len(sl) == 1:
                    all_points.append(sl[[0]])

        elif sampling == "midpoints":
            for sl in streamlines:
                sl = np.asarray(sl, dtype=np.float64)
                mid = len(sl) // 2
                all_points.append(sl[[mid]])

        elif sampling == "uniform":
            for sl in streamlines:
                sl = np.asarray(sl, dtype=np.float64)
                if len(sl) < 2:
                    all_points.append(sl)
                    continue
                # Resample by arc-length interpolation.
                cumlen = np.concatenate([
                    [0],
                    np.cumsum(np.linalg.norm(np.diff(sl, axis=0), axis=1)),
                ])
                total = cumlen[-1]
                if total < 1e-10:
                    all_points.append(sl[:1])
                    continue
                target_dists = np.linspace(0, total, points_per_streamline)
                resampled = np.zeros((points_per_streamline, 3))
                for dim in range(3):
                    resampled[:, dim] = np.interp(target_dists, cumlen, sl[:, dim])
                all_points.append(resampled)
        else:
            raise ValueError(f"Unknown sampling: {sampling!r}")

        if not all_points:
            raise ValueError(f"No streamlines found in {trk_path}")

        points = np.vstack(all_points)

        meta = {
            "source": "tractography",
            "trk_file": str(Path(trk_path).name),
            "sampling": sampling,
            "n_streamlines": len(streamlines),
            "n_raw_points": points.shape[0],
            **(metadata or {}),
        }

        if subsample is not None and points.shape[0] > subsample:
            points, _ = farthest_point_sampling(points, subsample, seed=seed)
            meta["subsampled_to"] = subsample

        logger.info(
            "Extracted %d points from %d streamlines (%s sampling)",
            points.shape[0], len(streamlines), sampling,
        )
        return cls(points, metadata=meta)

    # ── From raw anatomical via containers ────────────────────────────

    @classmethod
    def from_raw(
        cls,
        raw_path: PathLike,
        label_id: int,
        *,
        output_dir: Optional[PathLike] = None,
        gpu: Optional[bool] = None,
        jitter: bool = True,
        seed: Optional[int] = None,
        subsample: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BrainPointCloud:
        """End-to-end: raw T1w/T2w/FLAIR → segment → point cloud.

        Chains skull-stripping and segmentation via Singularity
        containers, then extracts the target structure as a point
        cloud.  **No DL dependencies on host.**

        Parameters
        ----------
        raw_path : PathLike
            Raw anatomical NIfTI (any contrast).
        label_id : int
            Target structure label.
        output_dir : PathLike, optional
            Working directory for intermediate files.
        gpu : bool, optional
        jitter, seed, subsample : as in :meth:`from_volume`.
        metadata : dict, optional

        Returns
        -------
        BrainPointCloud
        """
        from spectralbrain.io.preprocess import raw_to_pointcloud

        points = raw_to_pointcloud(
            raw_path, label_id,
            output_dir=output_dir, gpu=gpu,
            jitter=jitter, jitter_scale=0.25, seed=seed,
        )

        meta = {
            "source": "raw_pipeline",
            "raw_file": str(Path(raw_path).name),
            "label_id": label_id,
            **(metadata or {}),
        }

        if subsample is not None and points.shape[0] > subsample:
            points, _ = farthest_point_sampling(points, subsample, seed=seed)
            meta["subsampled_to"] = subsample

        return cls(points, metadata=meta)

    # ══════════════════════════════════════════════════════════════════
    # §3  LAPLACIAN CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════

    def compute_laplacian(
        self,
        method: Literal["knn", "belkin_niyogi", "robust"] = "robust",
        *,
        k: int = 30,
        epsilon: Optional[float] = None,
        sigma: Optional[float] = None,
        robust_mollify: float = 1e-5,
    ) -> Tuple[SparseMatrix, MassMatrix]:
        """Construct a graph Laplacian on the point cloud.

        Parameters
        ----------
        method : str
            ``"knn"`` — Gaussian-weighted kNN graph Laplacian.
            ``"belkin_niyogi"`` — heat-kernel weighted Laplacian
            with provable convergence to the continuous LBO
            (Belkin & Niyogi, JCSS 2008).
            ``"robust"`` — Sharp & Crane tufted Laplacian
            (robust to noise and non-uniform density).
        k : int
            Number of neighbours for kNN and Belkin–Niyogi.
        epsilon : float, optional
            Bandwidth for ε-ball graph.  ``None`` = auto from
            mean kNN distance.
        sigma : float, optional
            Gaussian kernel bandwidth.  ``None`` = auto (median
            distance heuristic).
        robust_mollify : float
            Mollification for the robust method.

        Returns
        -------
        L : SparseMatrix, shape (N, N)
        M : MassMatrix, shape (N, N)
        """
        if method == "knn":
            L, M = _knn_laplacian(
                self.points, k=k, sigma=sigma,
            )
        elif method == "belkin_niyogi":
            L, M = _belkin_niyogi_laplacian(
                self.points, k=k, epsilon=epsilon,
            )
        elif method == "robust":
            L, M = _robust_laplacian_pc(
                self.points, mollify_factor=robust_mollify,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        self._L = L
        self._M = M
        logger.info(
            "Point cloud Laplacian (%s): N=%d, nnz=%d",
            method, L.shape[0], L.nnz,
        )
        return L, M

    # ── Spectral decomposition ────────────────────────────────────────

    def decompose(
        self,
        k: int = 50,
        *,
        laplacian_method: Literal["knn", "belkin_niyogi", "robust"] = "robust",
        backend: Optional[Any] = None,
        **kwargs: Any,
    ) -> SpectralDecomposition:
        """Compute the spectral decomposition.

        Parameters
        ----------
        k : int
            Number of eigenpairs.
        laplacian_method : str
            Passed to :meth:`compute_laplacian`.
        backend : Backend, optional
        **kwargs
            Extra args for :meth:`compute_laplacian` and
            ``backend.eigsh()``.

        Returns
        -------
        SpectralDecomposition
        """
        if self._L is None or self._M is None:
            lap_kwargs = {
                key: kwargs.pop(key)
                for key in ("sigma", "epsilon", "robust_mollify")
                if key in kwargs
            }
            self.compute_laplacian(
                method=laplacian_method,
                k=kwargs.pop("knn_k", 30),
                **lap_kwargs,
            )

        be = backend or NumpyBackend()
        evals, evecs = be.eigsh(self._L, self._M, k=k, **kwargs)

        return SpectralDecomposition(
            eigenvalues=evals,
            eigenvectors=evecs,
            stiffness=self._L,
            mass=self._M,
            surface_area=self.surface_area(),
            metadata={
                **self.metadata,
                "laplacian_method": laplacian_method,
                "backend": be.name,
                "n_points": self.n_points,
            },
        )

    # ══════════════════════════════════════════════════════════════════
    # §4  NORMALS AND CURVATURE (from local PCA, no faces needed)
    # ══════════════════════════════════════════════════════════════════

    def compute_normals(
        self,
        k: int = 15,
        *,
        orient_to_centroid: bool = True,
    ) -> Normals:
        """Estimate per-point normals via local PCA.

        The normal at each point is the eigenvector of the local
        covariance matrix corresponding to the **smallest**
        eigenvalue (the direction of least variance in the
        neighbourhood).

        Parameters
        ----------
        k : int
            Neighbours for local PCA.
        orient_to_centroid : bool
            Flip normals to point away from the cloud centroid
            (heuristic for outward orientation).

        Returns
        -------
        normals : ndarray, shape (N, 3)
        """
        if self._normals is not None:
            return self._normals

        N = self.n_points
        normals = np.zeros((N, 3), dtype=np.float64)
        _, indices = knn_search(self.points, k=k)
        centroid = compute_centroid(self.points)

        with progress_simple("Estimating normals", total=N) as tick:
            for i in range(N):
                nbrs = self.points[indices[i]]             # (k, 3)
                cov = np.cov(nbrs, rowvar=False)           # (3, 3)
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Smallest eigenvalue → normal direction.
                normals[i] = eigvecs[:, 0]

                if orient_to_centroid:
                    outward = self.points[i] - centroid
                    if np.dot(normals[i], outward) < 0:
                        normals[i] *= -1

                if (i + 1) % 500 == 0 or i == N - 1:
                    tick(min(500, N - (i + 1 - 500)))

        # Normalise (should already be unit, but ensure).
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.clip(norms, 1e-12, None)

        self._normals = normals
        return normals

    def local_curvature(
        self,
        k: int = 15,
    ) -> Tuple[ScalarMap, ScalarMap]:
        """Estimate per-point curvature from local PCA eigenvalues.

        The ratio λ₀ / (λ₀ + λ₁ + λ₂) of the smallest eigenvalue
        to the sum is a measure of local planarity — smaller values
        mean flatter (lower curvature), larger values mean more curved.

        Parameters
        ----------
        k : int
            Neighbours for local PCA.

        Returns
        -------
        curvature : ndarray, shape (N,)
            Normalised planarity measure (0 = flat, 1 = isotropic).
        linearity : ndarray, shape (N,)
            (λ₀ - λ₁) / λ₀ — high for tube-like structures.
        """
        N = self.n_points
        curvature = np.zeros(N, dtype=np.float64)
        linearity = np.zeros(N, dtype=np.float64)
        _, indices = knn_search(self.points, k=k)

        with progress_simple("Estimating curvature", total=N) as tick:
            for i in range(N):
                nbrs = self.points[indices[i]]
                cov = np.cov(nbrs, rowvar=False)
                eigvals = np.linalg.eigvalsh(cov)          # ascending
                total = eigvals.sum()
                if total > 1e-20:
                    curvature[i] = eigvals[0] / total
                    if eigvals[2] > 1e-20:
                        linearity[i] = (eigvals[2] - eigvals[1]) / eigvals[2]
                if (i + 1) % 500 == 0 or i == N - 1:
                    tick(min(500, N - (i + 1 - 500)))

        return curvature, linearity

    # ══════════════════════════════════════════════════════════════════
    # §5  ATLAS-FREE CLUSTERING
    # ══════════════════════════════════════════════════════════════════

    def cluster(
        self,
        method: Literal[
            "hdbscan", "spectral", "kmeans", "dbscan",
        ] = "hdbscan",
        *,
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 50,
        eps: Optional[float] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Atlas-free spatial clustering of the point cloud.

        Assigns each point to a cluster without relying on any atlas
        or parcellation.  Useful for data-driven subregion discovery.

        Parameters
        ----------
        method : str
            ``"hdbscan"`` — density-based, handles irregular shapes,
            auto-detects number of clusters (recommended).
            ``"spectral"`` — spectral clustering on kNN graph.
            ``"kmeans"`` — simple, fast, spherical assumption.
            ``"dbscan"`` — density-based, fixed epsilon.
        n_clusters : int, optional
            For methods that require it (spectral, kmeans).
        min_cluster_size : int
            For HDBSCAN.
        eps : float, optional
            For DBSCAN.
        **kwargs
            Passed to the underlying clustering algorithm.

        Returns
        -------
        labels : ndarray, shape (N,), int
            Cluster assignment per point.  ``-1`` = noise (HDBSCAN /
            DBSCAN).
        """
        if method == "hdbscan":
            return _cluster_hdbscan(
                self.points,
                min_cluster_size=min_cluster_size,
                **kwargs,
            )
        elif method == "spectral":
            if n_clusters is None:
                raise ValueError("n_clusters required for spectral clustering.")
            return _cluster_spectral(
                self.points, n_clusters=n_clusters, **kwargs,
            )
        elif method == "kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters required for kmeans.")
            return _cluster_kmeans(
                self.points, n_clusters=n_clusters, **kwargs,
            )
        elif method == "dbscan":
            return _cluster_dbscan(
                self.points, eps=eps, **kwargs,
            )
        else:
            raise ValueError(f"Unknown clustering method: {method!r}")

    # ── Subsample ─────────────────────────────────────────────────────

    def subsample(
        self,
        n: int,
        *,
        seed: Optional[int] = None,
    ) -> BrainPointCloud:
        """Return a subsampled copy via farthest-point sampling.

        Parameters
        ----------
        n : int
            Target number of points.
        seed : int, optional

        Returns
        -------
        BrainPointCloud
        """
        sampled, idx = farthest_point_sampling(self.points, n, seed=seed)
        meta = {
            **self.metadata,
            "subsampled_to": n,
            "subsampled_from": self.n_points,
        }
        return BrainPointCloud(sampled, metadata=meta)

    # ── Denoise ───────────────────────────────────────────────────────

    def denoise(
        self,
        k: int = 10,
        threshold_sigma: float = 2.5,
    ) -> BrainPointCloud:
        """Remove density outliers (statistical outlier removal).

        Parameters
        ----------
        k : int
            Neighbours for density estimation.
        threshold_sigma : float
            Points beyond this many σ from mean log-density
            are removed.

        Returns
        -------
        BrainPointCloud
            Cleaned copy.
        """
        from spectralbrain.core.base import detect_density_outliers

        outliers = detect_density_outliers(
            self.points, k=k, threshold_sigma=threshold_sigma,
        )
        clean = self.points[~outliers]
        n_removed = outliers.sum()
        logger.info(
            "Denoised: removed %d / %d points (%.1f%%)",
            n_removed, self.n_points, 100 * n_removed / self.n_points,
        )
        meta = {
            **self.metadata,
            "denoised": True,
            "n_removed": int(n_removed),
        }
        return BrainPointCloud(clean, metadata=meta)

    # ── repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a compact point cloud summary string."""
        src = self.metadata.get("source", "")
        struct = self.metadata.get("structure", "")
        parts = [f"BrainPointCloud(n_points={self.n_points}"]
        if src:
            """Return a compact point cloud summary."""
            parts.append(f", source='{src}'")
        if struct:
            parts.append(f", structure='{struct}'")
        parts.append(")")
        return "".join(parts)


# ======================================================================
# §6  LAPLACIAN IMPLEMENTATIONS
# ======================================================================

def _knn_laplacian(
    points: Points,
    *,
    k: int = 30,
    sigma: Optional[float] = None,
) -> Tuple[SparseMatrix, MassMatrix]:
    """Gaussian-weighted kNN graph Laplacian.

    W_ij = exp(-||x_i - x_j||² / (2σ²))  if j ∈ kNN(i)
    L = D - W
    M = D (diagonal degree as mass)

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
    sigma : float, optional
        Kernel bandwidth.  ``None`` = median distance heuristic.

    Returns
    -------
    L, M : SparseMatrix
    """
    N = points.shape[0]
    dists, indices = knn_search(points, k=k)

    if sigma is None:
        # Median heuristic: σ = median of all kNN distances.
        sigma = float(np.median(dists[:, 1:]))
        if sigma < 1e-10:
            sigma = 1.0
    sigma2 = 2.0 * sigma ** 2

    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()
    weights = np.exp(-dists.ravel() ** 2 / sigma2)

    W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    W = (W + W.T) / 2                                     # symmetrise

    D_vals = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(D_vals, 0, shape=(N, N), format="csc")
    L = sp.csc_matrix(D - W)

    # Mass = diagonal degree.
    M = sp.diags(D_vals, 0, shape=(N, N), format="csc")

    return L, M


def _belkin_niyogi_laplacian(
    points: Points,
    *,
    k: int = 30,
    epsilon: Optional[float] = None,
) -> Tuple[SparseMatrix, MassMatrix]:
    """Belkin–Niyogi heat-kernel Laplacian with convergence guarantees.

    The graph Laplacian L_ε converges to the continuous Laplace–
    Beltrami operator as N → ∞ and ε → 0 at appropriate rate
    (Belkin & Niyogi, JCSS 2008).

    W_ij = (1 / (4πε)) · exp(-||x_i - x_j||² / (4ε))
    L_ε = (1/N) · (D - W)

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        kNN for sparsification (exact B-N uses all pairs;
        kNN approximation is standard for scalability).
    epsilon : float, optional
        Bandwidth ε.  ``None`` = auto from mean kNN distance squared.

    Returns
    -------
    L, M : SparseMatrix
    """
    N = points.shape[0]
    dists, indices = knn_search(points, k=k)

    if epsilon is None:
        mean_dist = float(np.mean(dists[:, 1:]))
        epsilon = mean_dist ** 2
        if epsilon < 1e-10:
            epsilon = 1.0

    coeff = 1.0 / (4.0 * np.pi * epsilon)
    exp_coeff = -1.0 / (4.0 * epsilon)

    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()
    weights = coeff * np.exp(exp_coeff * dists.ravel() ** 2)

    W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    W = (W + W.T) / 2

    D_vals = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(D_vals, 0, shape=(N, N), format="csc")
    L = sp.csc_matrix((1.0 / N) * (D - W))

    # Uniform mass for point clouds (1/N per point).
    M = sp.diags(
        np.full(N, 1.0 / N, dtype=np.float64),
        0, shape=(N, N), format="csc",
    )

    return L, M


def _robust_laplacian_pc(
    points: Points,
    *,
    mollify_factor: float = 1e-5,
) -> Tuple[SparseMatrix, MassMatrix]:
    """Sharp–Crane tufted Laplacian for point clouds (SGP 2020).

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    mollify_factor : float

    Returns
    -------
    L, M : SparseMatrix
    """
    try:
        import robust_laplacian as rl
    except ImportError as exc:
        raise ImportError(
            "robust_laplacian is required for method='robust'.\n"
            "  pip install robust_laplacian"
        ) from exc

    L, M = rl.point_cloud_laplacian(
        np.asarray(points, dtype=np.float64),
        mollify_factor=mollify_factor,
    )
    return sp.csc_matrix(L), sp.csc_matrix(M)


# ======================================================================
# §7  CLUSTERING IMPLEMENTATIONS
# ======================================================================

def _cluster_hdbscan(
    points: Points,
    *,
    min_cluster_size: int = 50,
    **kwargs: Any,
) -> np.ndarray:
    """HDBSCAN clustering."""
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size, **kwargs,
        )
    except ImportError:
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size, **kwargs,
            )
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN requires scikit-learn >= 1.3 or the hdbscan package.\n"
                "  pip install scikit-learn  # or: pip install hdbscan"
            ) from exc

    labels = clusterer.fit_predict(points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(
        "HDBSCAN: %d clusters, %d noise points (%.1f%%)",
        n_clusters, n_noise, 100 * n_noise / len(labels),
    )
    return labels


def _cluster_spectral(
    points: Points,
    *,
    n_clusters: int,
    k: int = 20,
    **kwargs: Any,
) -> np.ndarray:
    """Spectral clustering on kNN affinity graph."""
    from sklearn.cluster import SpectralClustering

    # Build kNN affinity.
    dists, indices = knn_search(points, k=k)
    N = points.shape[0]
    sigma = float(np.median(dists[:, 1:]))
    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()
    weights = np.exp(-dists.ravel() ** 2 / (2 * sigma ** 2))
    W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    W = (W + W.T) / 2

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        **kwargs,
    )
    labels = sc.fit_predict(W.toarray())
    logger.info("Spectral clustering: %d clusters", n_clusters)
    return labels


def _cluster_kmeans(
    points: Points,
    *,
    n_clusters: int,
    **kwargs: Any,
) -> np.ndarray:
    """K-means clustering."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, **kwargs)
    labels = km.fit_predict(points)
    logger.info("K-means: %d clusters", n_clusters)
    return labels


def _cluster_dbscan(
    points: Points,
    *,
    eps: Optional[float] = None,
    min_samples: int = 10,
    **kwargs: Any,
) -> np.ndarray:
    """DBSCAN clustering."""
    from sklearn.cluster import DBSCAN

    if eps is None:
        dists, _ = knn_search(points, k=min_samples)
        eps = float(np.percentile(dists[:, -1], 90))

    db = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = db.fit_predict(points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("DBSCAN (ε=%.2f): %d clusters", eps, n_clusters)
    return labels


# ======================================================================
# §8  __all__
# ======================================================================

__all__ = [
    "BrainPointCloud",
]
