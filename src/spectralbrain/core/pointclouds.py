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

import warnings
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from spectralbrain.backends.cpu import NumpyBackend
from spectralbrain.core.base import (
    SpectralDecomposition,
    _knn_excluding_self,
    compute_centroid,
    convex_hull_area,
    farthest_point_sampling,
    knn_search,
)
from spectralbrain.runtime import (
    MassMatrix,
    Normals,
    PathLike,
    Points,
    ScalarMap,
    SparseMatrix,
    get_logger,
    progress_simple,
    resolve_seed,
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialise from a (N, 3) coordinate array."""
        self.points = np.asarray(points, dtype=np.float64)
        self.metadata = metadata or {}

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {self.points.shape}")

        # Cached properties.
        self._normals: Normals | None = None
        self._L: SparseMatrix | None = None
        self._M: MassMatrix | None = None
        self._lap_key: tuple | None = None
        self._kd_tree: cKDTree | None = None

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
        seed: int | None = None,
        subsample: int | None = None,
        metadata: dict[str, Any] | None = None,
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
            label_volume,
            affine,
            label_id,
            jitter=jitter,
            jitter_scale=jitter_scale,
            seed=seed,
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
        seed: int | None = None,
        subsample: int | None = None,
        metadata: dict[str, Any] | None = None,
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
            data,
            affine,
            label_id,
            jitter=jitter,
            jitter_scale=jitter_scale,
            seed=seed,
            subsample=subsample,
            metadata=meta,
        )

    # ── From any surface file (vertices as points) ────────────────────

    @classmethod
    def from_surface(
        cls,
        surface_path: PathLike,
        *,
        subsample: int | None = None,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
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
        seed: int | None = None,
        subsample: int | None = None,
        metadata: dict[str, Any] | None = None,
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
            data,
            affine,
            label_id,
            jitter=jitter,
            seed=seed,
            subsample=subsample,
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
        seed: int | None = None,
        subsample: int | None = None,
        metadata: dict[str, Any] | None = None,
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
            binary,
            affine,
            label_id=1,
            jitter=jitter,
            seed=seed,
            subsample=subsample,
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
        subsample: int | None = None,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
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
                "nibabel is required for tractography loading.\n  pip install nibabel"
            ) from exc

        trk = nib.streamlines.load(str(trk_path))
        streamlines = trk.streamlines

        all_points: list[np.ndarray] = []

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
                cumlen = np.concatenate(
                    [
                        [0],
                        np.cumsum(np.linalg.norm(np.diff(sl, axis=0), axis=1)),
                    ]
                )
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
            points.shape[0],
            len(streamlines),
            sampling,
        )
        return cls(points, metadata=meta)

    # ── From raw anatomical via containers ────────────────────────────

    @classmethod
    def from_raw(
        cls,
        raw_path: PathLike,
        label_id: int,
        *,
        output_dir: PathLike | None = None,
        gpu: bool | None = None,
        jitter: bool = True,
        seed: int | None = None,
        subsample: int | None = None,
        metadata: dict[str, Any] | None = None,
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
            raw_path,
            label_id,
            output_dir=output_dir,
            gpu=gpu,
            jitter=jitter,
            jitter_scale=0.25,
            seed=seed,
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
        epsilon: float | None = None,
        sigma: float | None = None,
        robust_mollify: float = 1e-5,
        kernel_sigma: float | None = None,
        area: float | None = None,
    ) -> tuple[SparseMatrix, MassMatrix]:
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
            Number of neighbours for kNN and Belkin–Niyogi (the point
            itself is not counted).
        epsilon : float, optional
            Heat-kernel bandwidth ε for ``"belkin_niyogi"``.  ``None`` =
            auto from the kNN radius.
        sigma : float, optional
            Gaussian kernel bandwidth for ``"knn"``.  ``None`` = auto
            (median distance heuristic).  Same as *kernel_sigma*.
        robust_mollify : float
            Mollification for the robust method.
        kernel_sigma : float, optional
            Preferred name for the Gaussian kernel bandwidth (takes
            precedence over *sigma*).
        area : float, optional
            Surface area used by ``"belkin_niyogi"`` to scale the operator
            to the continuous LBO.  ``None`` = convex-hull area.

        Returns
        -------
        L : SparseMatrix, shape (N, N)
        M : MassMatrix, shape (N, N)
        """
        bw = kernel_sigma if kernel_sigma is not None else sigma
        if method == "knn":
            L, M = _knn_laplacian(
                self.points,
                k=k,
                sigma=bw,
            )
        elif method == "belkin_niyogi":
            L, M = _belkin_niyogi_laplacian(
                self.points,
                k=k,
                epsilon=epsilon,
                area=area,
            )
        elif method == "robust":
            L, M = _robust_laplacian_pc(
                self.points,
                mollify_factor=robust_mollify,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        self._L = L
        self._M = M
        self._lap_key = _laplacian_key(
            method,
            k=k,
            epsilon=epsilon,
            kernel_sigma=bw,
            robust_mollify=robust_mollify,
            area=area,
        )
        logger.info(
            "Point cloud Laplacian (%s): N=%d, nnz=%d",
            method,
            L.shape[0],
            L.nnz,
        )
        return L, M

    # ── Spectral decomposition ────────────────────────────────────────

    def decompose(
        self,
        k: int = 50,
        *,
        laplacian_method: Literal["knn", "belkin_niyogi", "robust"] | None = None,
        backend: Any | None = None,
        **kwargs: Any,
    ) -> SpectralDecomposition:
        """Compute the spectral decomposition.

        Parameters
        ----------
        k : int
            Number of eigenpairs.
        laplacian_method : str, optional
            Passed to :meth:`compute_laplacian`.  ``None`` (default) reuses
            the Laplacian already built by :meth:`compute_laplacian` (if no
            Laplacian keyword is given), else ``"robust"``.
        backend : Backend, optional
        **kwargs
            Laplacian keywords — ``knn_k``, ``kernel_sigma``, ``epsilon``,
            ``robust_mollify``, ``area`` — are forwarded to
            :meth:`compute_laplacian`; the Laplacian is cached and rebuilt
            whenever the method or any of these change.  ``sigma`` is a
            **deprecated** alias of ``kernel_sigma`` (Gaussian bandwidth);
            to set the eigensolver's shift-invert value use
            ``eigsh_sigma``.  Every other keyword goes to
            ``backend.eigsh()``.

        Returns
        -------
        SpectralDecomposition
        """
        if "sigma" in kwargs:
            warnings.warn(
                "decompose(sigma=...) is deprecated: it sets the kNN Gaussian "
                "kernel bandwidth. Use kernel_sigma=... (bandwidth) or "
                "eigsh_sigma=... (eigensolver shift) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            bw = kwargs.pop("sigma")
            kwargs.setdefault("kernel_sigma", bw)

        lap_names = ("knn_k", "kernel_sigma", "epsilon", "robust_mollify", "area")
        lap_kwargs = {name: kwargs.pop(name) for name in lap_names if name in kwargs}
        if "eigsh_sigma" in kwargs:
            kwargs["sigma"] = kwargs.pop("eigsh_sigma")

        cached = self._L is not None and self._M is not None and self._lap_key is not None
        if laplacian_method is None and not lap_kwargs and cached:
            pass  # reuse the user's explicitly built Laplacian
        else:
            method = laplacian_method or (
                self._lap_key[0] if cached and self._lap_key is not None else "robust"
            )
            build = {
                "k": lap_kwargs.get("knn_k", 30),
                "epsilon": lap_kwargs.get("epsilon"),
                "kernel_sigma": lap_kwargs.get("kernel_sigma"),
                "robust_mollify": lap_kwargs.get("robust_mollify", 1e-5),
                "area": lap_kwargs.get("area"),
            }
            key = _laplacian_key(method, **build)
            if not cached or key != self._lap_key:
                self.compute_laplacian(method=method, **build)

        be = backend or NumpyBackend()
        evals, evecs = be.eigsh(self._L, self._M, k=k, **kwargs)

        assert self._lap_key is not None
        return SpectralDecomposition(
            eigenvalues=evals,
            eigenvectors=evecs,
            stiffness=self._L,
            mass=self._M,
            surface_area=self.surface_area(),
            metadata={
                **self.metadata,
                "laplacian_method": self._lap_key[0],
                "laplacian_params": dict(self._lap_key[1]),
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
            done = 0
            for i in range(N):
                nbrs = self.points[indices[i]]  # (k, 3)
                cov = np.cov(nbrs, rowvar=False)  # (3, 3)
                _eigvals, eigvecs = np.linalg.eigh(cov)
                # Smallest eigenvalue → normal direction.
                normals[i] = eigvecs[:, 0]

                if orient_to_centroid:
                    outward = self.points[i] - centroid
                    if np.dot(normals[i], outward) < 0:
                        normals[i] *= -1

                if (i + 1) % 500 == 0 or i == N - 1:
                    tick(i + 1 - done)
                    done = i + 1

        # Normalise (should already be unit, but ensure).
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.clip(norms, 1e-12, None)

        self._normals = normals
        return normals

    def local_curvature(
        self,
        k: int = 15,
    ) -> tuple[ScalarMap, ScalarMap]:
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
            done = 0
            for i in range(N):
                nbrs = self.points[indices[i]]
                cov = np.cov(nbrs, rowvar=False)
                eigvals = np.linalg.eigvalsh(cov)  # ascending
                total = eigvals.sum()
                if total > 1e-20:
                    curvature[i] = eigvals[0] / total
                    if eigvals[2] > 1e-20:
                        linearity[i] = (eigvals[2] - eigvals[1]) / eigvals[2]
                if (i + 1) % 500 == 0 or i == N - 1:
                    tick(i + 1 - done)
                    done = i + 1

        return curvature, linearity

    # ══════════════════════════════════════════════════════════════════
    # §5  ATLAS-FREE CLUSTERING
    # ══════════════════════════════════════════════════════════════════

    def cluster(
        self,
        method: Literal[
            "hdbscan",
            "spectral",
            "kmeans",
            "dbscan",
        ] = "hdbscan",
        *,
        n_clusters: int | None = None,
        min_cluster_size: int = 50,
        eps: float | None = None,
        seed: int | None = None,
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
        seed : int, optional
            ``random_state`` for the stochastic methods (k-means, spectral).
            ``None`` falls back to the library-wide seed set by
            :func:`~spectralbrain.utils.seed_everything`.
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
            kwargs.setdefault("random_state", resolve_seed(seed))
            return _cluster_spectral(
                self.points,
                n_clusters=n_clusters,
                **kwargs,
            )
        elif method == "kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters required for kmeans.")
            kwargs.setdefault("random_state", resolve_seed(seed))
            return _cluster_kmeans(
                self.points,
                n_clusters=n_clusters,
                **kwargs,
            )
        elif method == "dbscan":
            return _cluster_dbscan(
                self.points,
                eps=eps,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown clustering method: {method!r}")

    # ── Subsample ─────────────────────────────────────────────────────

    def subsample(
        self,
        n: int,
        *,
        seed: int | None = None,
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
        sampled, _idx = farthest_point_sampling(self.points, n, seed=seed)
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
            self.points,
            k=k,
            threshold_sigma=threshold_sigma,
        )
        clean = self.points[~outliers]
        n_removed = outliers.sum()
        logger.info(
            "Denoised: removed %d / %d points (%.1f%%)",
            n_removed,
            self.n_points,
            100 * n_removed / self.n_points,
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
    sigma: float | None = None,
) -> tuple[SparseMatrix, MassMatrix]:
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
    dists, indices = _knn_excluding_self(points, k)  # no self-loops
    k = indices.shape[1]

    if sigma is None:
        # Median heuristic: σ = median of all kNN distances.
        sigma = float(np.median(dists))
        if sigma < 1e-10:
            sigma = 1.0
    sigma2 = 2.0 * sigma**2

    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()
    weights = np.exp(-(dists.ravel() ** 2) / sigma2)

    W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    W = (W + W.T) / 2  # symmetrise

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
    epsilon: float | None = None,
    area: float | None = None,
) -> tuple[SparseMatrix, MassMatrix]:
    """Belkin–Niyogi heat-kernel Laplacian with convergence guarantees.

    The point-cloud Laplacian (2-manifold, uniform sampling)

        L_ε f(x_i) = 1/(ε · 4πε) · (1/N) Σ_j (f(x_i) − f(x_j)) e^{−‖x_i−x_j‖²/4ε}

    converges to ``(1/A) Δ f`` as N → ∞, ε → 0 (Belkin & Niyogi, JCSS
    2008), A being the surface area.  We therefore pose the generalised
    problem ``L v = λ M v`` with the lumped mass ``M = (A/N) I`` and
    stiffness ``L = (A/N)² / (4π ε²) · (D − W)``, so that
    ``M⁻¹ L = A · L_ε → Δ`` and the eigenvalues approximate the LBO
    spectrum (same scale as the mesh pathway).

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    k : int
        kNN for sparsification (exact B-N uses all pairs;
        kNN approximation is standard for scalability).  The point itself
        is excluded.
    epsilon : float, optional
        Bandwidth ε.  ``None`` = auto, ``r_k² / 25`` with ``r_k`` the median
        distance to the k-th neighbour (keeps the kernel inside the kNN
        support).  Choose ``k`` large enough for the desired ε.
    area : float, optional
        Surface area A.  ``None`` = convex-hull area of the points.

    Returns
    -------
    L, M : SparseMatrix
    """
    N = points.shape[0]
    dists, indices = _knn_excluding_self(points, k)
    k = indices.shape[1]

    if epsilon is None:
        # The heat kernel exp(−r²/4ε) (σ² = 2ε) must fit inside the kNN
        # radius r_k, otherwise truncation biases the eigenvalues low:
        # ε = r_k² / 25 puts r_k at ≈ 3.5σ.
        r_k = float(np.median(dists[:, -1]))
        epsilon = r_k**2 / 25.0
        if epsilon < 1e-12:
            epsilon = 1.0
    if area is None:
        area = convex_hull_area(points)
    if not area or area <= 0:
        raise ValueError("Belkin–Niyogi Laplacian needs a positive surface area.")

    exp_coeff = -1.0 / (4.0 * epsilon)
    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()
    weights = np.exp(exp_coeff * dists.ravel() ** 2)

    W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    W = (W + W.T) / 2

    D_vals = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(D_vals, 0, shape=(N, N), format="csc")
    a_i = area / N
    scale = a_i**2 / (4.0 * np.pi * epsilon**2)
    L = sp.csc_matrix(scale * (D - W))

    # Uniform lumped mass: each point carries A/N of the surface.
    M = sp.diags(
        np.full(N, a_i, dtype=np.float64),
        0,
        shape=(N, N),
        format="csc",
    )

    return L, M


def _robust_laplacian_pc(
    points: Points,
    *,
    mollify_factor: float = 1e-5,
) -> tuple[SparseMatrix, MassMatrix]:
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
            "robust_laplacian is required for method='robust'.\n  pip install robust_laplacian"
        ) from exc

    L, M = rl.point_cloud_laplacian(
        np.asarray(points, dtype=np.float64),
        mollify_factor=mollify_factor,
    )
    return sp.csc_matrix(L), sp.csc_matrix(M)


def _laplacian_key(
    method: str,
    *,
    k: int = 30,
    epsilon: float | None = None,
    kernel_sigma: float | None = None,
    robust_mollify: float = 1e-5,
    area: float | None = None,
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Cache key: method + only the parameters that method actually uses."""
    if method == "knn":
        params: dict[str, Any] = {"k": int(k), "kernel_sigma": kernel_sigma}
    elif method == "belkin_niyogi":
        params = {"k": int(k), "epsilon": epsilon, "area": area}
    else:
        params = {"robust_mollify": robust_mollify}
    return method, tuple(sorted(params.items()))


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
            min_cluster_size=min_cluster_size,
            **kwargs,
        )
    except ImportError:
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                **kwargs,
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
        n_clusters,
        n_noise,
        100 * n_noise / len(labels),
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
    weights = np.exp(-(dists.ravel() ** 2) / (2 * sigma**2))
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
    eps: float | None = None,
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
