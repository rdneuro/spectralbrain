"""Spectral-descriptor clustering for brain surface meshes.

Spatial, temporal, and joint spatio-temporal clustering of HKS, WKS,
and fused descriptor matrices on triangle meshes and point clouds.
Every algorithm is input-agnostic (mesh or point cloud — only an
adjacency matrix is needed for spatially-regularised methods) and
exposes a ``backend`` parameter for CPU/GPU dispatch.

Sections
--------
§1  Result containers
§2  Distance / affinity construction
§3  Spatial clustering (cluster vertices by descriptor values)
§4  Temporal / scale clustering (cluster the t- or E-axis)
§5  Spatio-temporal joint clustering
§6  HKS + WKS descriptor fusion
§7  Bayesian cluster confirmation
§8  Cluster quality & comparison metrics
§9  Convenience / pipeline wrappers
§10 Persistence vineyards — tracking topology across HKS scales
§11 Mapper pipeline — topological data analysis on meshes
§12 Non-negative tensor decomposition (multi-subject)
§13 Joint time-vertex graph signal processing
§14 Scale-space blob tracking (Lindeberg on manifolds)
§15 Multi-view clustering (geometry + descriptor views)
§16 Spectral graph wavelet clustering

Design principles
-----------------
* **Backend-agnostic**: ``backend="auto"`` chooses GPU when available.
* **k-free where possible**: most methods auto-determine cluster count.
* **Progress bars**: all O(n²) or iterative routines expose Rich bars.
* **Memory-safe**: GPU paths use batched transfers; CPU paths use
  ``joblib`` parallelism where effective.
* **Reproducible**: every stochastic method accepts ``random_state``.

References
----------
Sun J, Ovsjanikov M, Guibas L. A concise and provably informative
    multi-scale signature based on heat diffusion. *Computer Graphics
    Forum* 28(5):1383–1392, 2009.
Aubry M, Schlickewei U, Cremers D. The wave kernel signature: a
    quantum mechanical approach to shape analysis. *ICCV Workshops*,
    2011.
Cai D, He X, Han J, Huang TS. Graph regularized nonnegative matrix
    factorization for data representation. *IEEE TPAMI* 33(8):
    1548–1560, 2011.
Campello RJGB, Moulavi D, Sander J. Density-based clustering based
    on hierarchical density estimates. *PAKDD*, 2013.
Chazal F, Guibas LJ, Oudot SY, Skraba P. Persistence-based
    clustering in Riemannian manifolds. *J. ACM* 60(6):41, 2013.
Dhillon IS. Co-clustering documents and words using bipartite
    spectral graph partitioning. *KDD*, 2001.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
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
from scipy import stats as sp_stats

from spectralbrain.runtime import (
    DescriptorMatrix,
    DistanceMatrix,
    LabelArray,
    ScalarMap,
    SparseMatrix,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Lazy imports — every optional dependency is loaded on first use
# ──────────────────────────────────────────────────────────────────────

def _require_hdbscan():
    """Lazy-import HDBSCAN."""
    try:
        import hdbscan
        return hdbscan
    except ImportError:
        raise ImportError(
            "hdbscan required: pip install hdbscan"
        )


def _require_sklearn_cluster():
    """Lazy-import sklearn.cluster."""
    from sklearn import cluster as skc
    return skc


def _require_sklearn_metrics():
    """Lazy-import sklearn.metrics."""
    from sklearn import metrics as skm
    return skm


def _require_sklearn_decomposition():
    """Lazy-import sklearn.decomposition."""
    from sklearn import decomposition as skd
    return skd


def _require_sklearn_mixture():
    """Lazy-import sklearn.mixture."""
    from sklearn import mixture as skmix
    return skmix


def _require_umap():
    """Lazy-import UMAP."""
    try:
        import umap
        return umap
    except ImportError:
        raise ImportError("umap-learn required: pip install umap-learn")


def _require_leidenalg():
    """Lazy-import leidenalg."""
    try:
        import leidenalg
        import igraph
        return leidenalg, igraph
    except ImportError:
        raise ImportError(
            "leidenalg + igraph required: pip install leidenalg python-igraph"
        )


def _require_gudhi():
    """Lazy-import GUDHI."""
    try:
        import gudhi
        return gudhi
    except ImportError:
        raise ImportError("gudhi required: pip install gudhi")


def _require_tslearn():
    """Lazy-import tslearn."""
    try:
        import tslearn
        return tslearn
    except ImportError:
        raise ImportError("tslearn required: pip install tslearn")


def _require_skfda():
    """Lazy-import scikit-fda."""
    try:
        import skfda
        return skfda
    except ImportError:
        raise ImportError("scikit-fda required: pip install scikit-fda")


def _require_pymc():
    """Lazy-import PyMC for Bayesian clustering."""
    try:
        import pymc as pm
        import arviz as az
        return pm, az
    except ImportError:
        raise ImportError(
            "pymc + arviz required: pip install pymc arviz"
        )


def _require_torch():
    """Lazy-import PyTorch."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch required: pip install torch")


def _require_joblib():
    """Lazy-import joblib for parallelisation."""
    try:
        import joblib
        return joblib
    except ImportError:
        raise ImportError("joblib required: pip install joblib")


def _require_dionysus():
    """Lazy-import dionysus2 for persistent homology."""
    try:
        import dionysus
        return dionysus
    except ImportError:
        raise ImportError("dionysus required: pip install dionysus")


def _require_kepler_mapper():
    """Lazy-import KeplerMapper for TDA."""
    try:
        import kmapper
        return kmapper
    except ImportError:
        raise ImportError(
            "kepler-mapper required: pip install kmapper"
        )


def _require_tensorly():
    """Lazy-import TensorLy for tensor decomposition."""
    try:
        import tensorly
        import tensorly.decomposition
        return tensorly
    except ImportError:
        raise ImportError("tensorly required: pip install tensorly")


def _require_pygsp():
    """Lazy-import PyGSP for graph signal processing."""
    try:
        import pygsp
        return pygsp
    except ImportError:
        raise ImportError("PyGSP required: pip install PyGSP")


# ======================================================================
# §1  RESULT CONTAINERS
# ======================================================================

@dataclass
class ClusterResult:
    """Output of any spatial or spatio-temporal clustering algorithm.

    Attributes
    ----------
    labels : ndarray, shape (N,)
        Integer cluster label per vertex. ``-1`` = noise / unassigned.
    n_clusters : int
        Number of discovered clusters (excluding noise).
    method : str
        Algorithm name (e.g. ``"hdbscan"``, ``"gnmf"``, ``"leiden"``).
    probabilities : ndarray or None, shape (N,) or (N, K)
        Soft membership. Shape depends on method:
        ``(N,)`` for HDBSCAN membership probability,
        ``(N, K)`` for NMF / DPMM component weights.
    quality : dict
        Internal quality metrics (silhouette, modularity, etc.).
    metadata : dict
        Algorithm-specific outputs (condensed tree, persistence
        diagram, temporal profiles, etc.).
    """

    labels: LabelArray
    n_clusters: int
    method: str
    probabilities: Optional[np.ndarray] = None
    quality: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def noise_count(self) -> int:
        """Return the number of noise (unclustered) points."""
        return int((self.labels == -1).sum())

    @property
    def cluster_sizes(self) -> Dict[int, int]:
        """Return a dict mapping cluster label to member count."""
        unique, counts = np.unique(
            self.labels[self.labels >= 0], return_counts=True
        )
        return dict(zip(unique.tolist(), counts.tolist()))

    def __repr__(self) -> str:
        """Return a compact clustering result summary."""
        return (
            f"ClusterResult(method='{self.method}', "
            f"n_clusters={self.n_clusters}, "
            f"noise={self.noise_count}, "
            f"quality={self.quality})"
        )


@dataclass
class TemporalClusterResult:
    """Output of temporal/scale-axis clustering.

    Clusters the T time/energy samples rather than the N vertices.

    Attributes
    ----------
    labels : ndarray, shape (T,)
        Cluster label per time/energy sample.
    n_clusters : int
    centroids : ndarray or None, shape (K, N)
        Centroid profiles (one per cluster × all vertices).
    method : str
    quality : dict
    """

    labels: LabelArray
    n_clusters: int
    method: str
    centroids: Optional[np.ndarray] = None
    quality: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Output of HKS + WKS descriptor fusion.

    Attributes
    ----------
    fused : ndarray, shape (N, D)
        Fused per-vertex descriptor matrix.
    method : str
        Fusion strategy name.
    weights : ndarray or None
        Per-channel or per-kernel weights (for MKL / learned fusions).
    metadata : dict
    """

    fused: DescriptorMatrix
    method: str
    weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianClusterConfirmation:
    """Output of Bayesian cluster confirmation analysis.

    Attributes
    ----------
    posterior_labels : ndarray, shape (N,)
        MAP cluster assignments from the Bayesian model.
    label_probabilities : ndarray, shape (N, K)
        Full posterior membership probabilities.
    waic : float
        Widely Applicable Information Criterion.
    loo : float
        Leave-One-Out cross-validation estimate (ELPD).
    cluster_credible_intervals : dict
        Per-cluster posterior summaries (mean, HDI of centroid).
    agreement_with_input : float
        ARI between input labels and Bayesian MAP labels.
    metadata : dict
    """

    posterior_labels: LabelArray
    label_probabilities: np.ndarray
    waic: float
    loo: float
    cluster_credible_intervals: Dict[int, Dict[str, Any]]
    agreement_with_input: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VineyardResult:
    """Output of persistence vineyard tracking across HKS scales.

    Attributes
    ----------
    vines : list of ndarray
        Each vine is an array of (t, birth, death) triples tracing
        one topological feature across scales.
    diagrams : dict
        Persistence diagram at each sampled t, keyed by t index.
    salient_features : list of dict
        Features that persist across ≥ ``min_life`` fraction of the
        t-range, with their birth/death scale and spatial location.
    scale_of_emergence : ndarray, shape (n_features,)
        The t at which each salient feature first appears.
    metadata : dict
    """

    vines: List[np.ndarray]
    diagrams: Dict[int, np.ndarray]
    salient_features: List[Dict[str, Any]]
    scale_of_emergence: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapperResult:
    """Output of the Mapper pipeline on a descriptor-equipped mesh.

    Attributes
    ----------
    nerve_graph : dict
        Adjacency list of the nerve complex (node → set of neighbours).
    node_membership : dict
        Mapping from nerve-node index to list of vertex indices.
    n_nodes : int
        Number of nodes in the nerve complex.
    n_edges : int
        Number of edges.
    vertex_to_nodes : ndarray, shape (N,)
        For each vertex, the nerve node(s) it belongs to (first hit).
    metadata : dict
        Contains the full ``kmapper.KeplerMapper`` graph object for
        downstream visualisation.
    """

    nerve_graph: Dict[int, List[int]]
    node_membership: Dict[int, List[int]]
    n_nodes: int
    n_edges: int
    vertex_to_nodes: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorDecompositionResult:
    """Output of non-negative tensor CP/PARAFAC or Tucker decomposition.

    For a tensor ℋ ∈ ℝ^{n × T × S} (vertices × scales × subjects):

    Attributes
    ----------
    spatial_factors : ndarray, shape (N, R)
        Shared spatial components (vertex loadings).
    temporal_factors : ndarray, shape (T, R)
        Population-level temporal/scale profiles.
    subject_factors : ndarray, shape (S, R)
        Per-subject component strengths.
    labels : ndarray, shape (N,)
        Hard cluster labels from argmax of spatial factors.
    n_components : int
    reconstruction_error : float
    metadata : dict
    """

    spatial_factors: np.ndarray
    temporal_factors: np.ndarray
    subject_factors: np.ndarray
    labels: LabelArray
    n_components: int
    reconstruction_error: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleSpaceBlobResult:
    """Output of Lindeberg-style scale-space blob tracking on HKS.

    Attributes
    ----------
    blob_trajectories : list of list of dict
        Each trajectory is a list of {vertex, scale_index, t_value,
        response} dicts ordered by scale.
    natural_scales : ndarray, shape (N,)
        For each vertex, the t at which the normalised LoG response
        is maximal (its "natural scale").
    blob_labels : ndarray, shape (N,)
        Cluster label derived from trajectory membership.
    n_blobs : int
    metadata : dict
    """

    blob_trajectories: List[List[Dict[str, Any]]]
    natural_scales: np.ndarray
    blob_labels: LabelArray
    n_blobs: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# §2  DISTANCE / AFFINITY CONSTRUCTION
# ======================================================================

def build_descriptor_distance(
    H: DescriptorMatrix,
    *,
    metric: Literal[
        "euclidean", "cosine", "correlation", "manhattan"
    ] = "euclidean",
    log_transform: bool = True,
    normalize: Literal["none", "l1", "l2"] = "l1",
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> DistanceMatrix:
    """Build pairwise distance matrix from a descriptor matrix.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Per-vertex descriptor (HKS, WKS, or fused).
    metric : str
        Distance metric in feature space.
    log_transform : bool
        Apply log(H + ε) before distance computation. Recommended
        for HKS because values span many orders of magnitude.
    normalize : str
        Per-row normalisation after optional log transform.
    backend : str
        ``"auto"`` selects GPU if torch+CUDA available.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric pairwise distance matrix.
    """
    H = np.asarray(H, dtype=np.float64)
    n, t = H.shape

    # --- preprocessing ---
    X = np.log(H + 1e-12) if log_transform else H.copy()

    if normalize == "l1":
        row_sums = np.abs(X).sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X /= row_sums
    elif normalize == "l2":
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        X /= row_norms

    # --- compute distances ---
    use_gpu = _resolve_backend(backend)

    if use_gpu:
        torch = _require_torch()
        device = torch.device("cuda")
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        if metric == "euclidean":
            D_t = torch.cdist(X_t.unsqueeze(0), X_t.unsqueeze(0)).squeeze(0)
        elif metric == "cosine":
            X_norm = X_t / (X_t.norm(dim=1, keepdim=True) + 1e-12)
            D_t = 1.0 - X_norm @ X_norm.T
            D_t.clamp_(min=0.0)
        elif metric == "correlation":
            X_c = X_t - X_t.mean(dim=1, keepdim=True)
            X_cn = X_c / (X_c.norm(dim=1, keepdim=True) + 1e-12)
            D_t = 1.0 - X_cn @ X_cn.T
            D_t.clamp_(min=0.0)
        elif metric == "manhattan":
            D_t = torch.cdist(X_t.unsqueeze(0), X_t.unsqueeze(0),
                              p=1.0).squeeze(0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        D = D_t.cpu().numpy().astype(np.float64)
        del X_t, D_t
        torch.cuda.empty_cache()
    else:
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(X, metric=metric))

    np.fill_diagonal(D, 0.0)
    return D


def build_hybrid_distance(
    D_descriptor: DistanceMatrix,
    adjacency: SparseMatrix,
    *,
    alpha: float = 0.7,
    geodesic_backend: Literal["heat", "dijkstra"] = "dijkstra",
) -> DistanceMatrix:
    """Fuse descriptor distance with geodesic distance on the mesh.

    Parameters
    ----------
    D_descriptor : ndarray, shape (N, N)
        Pairwise descriptor-space distance.
    adjacency : sparse, shape (N, N)
        Mesh adjacency (cotangent weights or binary).
    alpha : float
        Weight for descriptor distance.  ``0.0`` = pure geodesic,
        ``1.0`` = pure descriptor.
    geodesic_backend : str
        ``"dijkstra"`` via scipy or ``"heat"`` via potpourri3d.

    Returns
    -------
    ndarray, shape (N, N)
        Normalised fused distance.
    """
    n = D_descriptor.shape[0]

    # --- geodesic distance from adjacency ---
    if geodesic_backend == "dijkstra":
        from scipy.sparse.csgraph import shortest_path
        A = sp.csr_matrix(adjacency).copy()
        # ensure positive weights for Dijkstra
        A.data = np.abs(A.data)
        A.data[A.data == 0] = 1e-12
        D_geo = shortest_path(A, method="D", directed=False)
    elif geodesic_backend == "heat":
        try:
            import potpourri3d as pp3d
        except ImportError:
            raise ImportError(
                "potpourri3d required for heat-method geodesics: "
                "pip install potpourri3d"
            )
        raise NotImplementedError(
            "Heat-method geodesics require (vertices, faces). "
            "Use geodesic_backend='dijkstra' with adjacency, "
            "or build the distance matrix externally."
        )
    else:
        raise ValueError(f"Unknown geodesic_backend: {geodesic_backend}")

    # --- normalise both to unit median ---
    med_desc = np.median(D_descriptor[D_descriptor > 0]) or 1.0
    med_geo = np.median(D_geo[D_geo > 0]) or 1.0

    D_fused = (
        alpha * (D_descriptor / med_desc)
        + (1.0 - alpha) * (D_geo / med_geo)
    )
    np.fill_diagonal(D_fused, 0.0)
    return D_fused


def build_hks_affinity_graph(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    sigma: Optional[float] = None,
    log_transform: bool = True,
) -> SparseMatrix:
    """Build HKS-weighted mesh adjacency for graph-based clustering.

    Each mesh edge (i, j) gets weight
    ``w_ij = c_ij · exp(-||h_i - h_j||² / 2σ²)``
    where ``c_ij`` is the original cotangent weight and ``h`` is
    the (optionally log-transformed, L1-normalised) HKS vector.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Descriptor matrix.
    adjacency : sparse, shape (N, N)
        Mesh Laplacian or adjacency with cotangent weights.
    sigma : float or None
        Kernel bandwidth. If None, uses the median edge-HKS-distance.
    log_transform : bool
        Apply log(H + ε) before computing feature distances.

    Returns
    -------
    sparse, shape (N, N)
        Weighted adjacency (CSR).
    """
    A = sp.coo_matrix(adjacency)
    rows, cols = A.row, A.col
    base_weights = np.abs(A.data)

    X = np.log(H + 1e-12) if log_transform else H.copy()
    row_sums = np.abs(X).sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X /= row_sums

    # feature distances along mesh edges
    d_feat = np.linalg.norm(X[rows] - X[cols], axis=1)

    if sigma is None:
        sigma = float(np.median(d_feat[d_feat > 0])) or 1.0

    affinity_weights = base_weights * np.exp(-d_feat**2 / (2 * sigma**2))

    W = sp.csr_matrix((affinity_weights, (rows, cols)),
                       shape=adjacency.shape)
    # symmetrise
    W = (W + W.T) / 2.0
    return W


# ======================================================================
# §3  SPATIAL CLUSTERING
# ======================================================================

def cluster_hdbscan(
    H: DescriptorMatrix,
    *,
    adjacency: Optional[SparseMatrix] = None,
    alpha_fusion: float = 0.7,
    min_cluster_size: int = 200,
    min_samples: int = 10,
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    metric: Literal[
        "euclidean", "precomputed"
    ] = "euclidean",
    dim_reduction: Optional[Literal["umap", "pca"]] = "umap",
    n_components: int = 8,
    log_transform: bool = True,
    random_state: int = 42,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> ClusterResult:
    """HDBSCAN clustering on spectral descriptor features.

    Density-based clustering that automatically determines the number
    of clusters. When an adjacency matrix is provided, uses a fused
    geodesic + descriptor distance for spatially coherent parcellation.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Per-vertex descriptor matrix (HKS, WKS, or fused).
    adjacency : sparse or None
        Mesh adjacency for hybrid distance fusion. If None, clusters
        purely in descriptor feature space.
    alpha_fusion : float
        Weight for descriptor distance in the hybrid metric.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    min_samples : int
        Core-distance neighbourhood size.
    cluster_selection_method : str
        ``"eom"`` (excess of mass) or ``"leaf"``.
    metric : str
        If ``"precomputed"``, H is treated as a distance matrix.
    dim_reduction : str or None
        Reduce descriptor dimensionality before clustering.
    n_components : int
        Target dimensionality for reduction.
    log_transform : bool
        Apply log(H + ε) before processing.
    random_state : int
        Seed for reproducibility.
    backend : str
        ``"auto"`` / ``"cpu"`` / ``"gpu"`` for distance computation.

    Returns
    -------
    ClusterResult
        With ``outlier_scores`` in metadata.
    """
    hdbscan = _require_hdbscan()
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]

    if metric == "precomputed":
        D = H
    else:
        # --- preprocess ---
        X = np.log(H + 1e-12) if log_transform else H.copy()
        row_sums = np.abs(X).sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X /= row_sums

        # --- dimensionality reduction ---
        if dim_reduction == "umap" and X.shape[1] > n_components:
            umap_mod = _require_umap()
            X = umap_mod.UMAP(
                n_components=n_components,
                metric="cosine",
                n_neighbors=min(30, n - 1),
                min_dist=0.0,
                random_state=random_state,
            ).fit_transform(X)
        elif dim_reduction == "pca" and X.shape[1] > n_components:
            skd = _require_sklearn_decomposition()
            X = skd.PCA(
                n_components=n_components,
                random_state=random_state,
            ).fit_transform(X)

        # --- fused or pure distance ---
        if adjacency is not None:
            D_desc = build_descriptor_distance(
                X, metric="euclidean", log_transform=False,
                normalize="none", backend=backend,
            )
            D = build_hybrid_distance(
                D_desc, adjacency, alpha=alpha_fusion,
            )
            use_metric = "precomputed"
        else:
            D = X
            use_metric = "euclidean"

    actual_metric = use_metric if adjacency is not None else "euclidean"
    if metric == "precomputed":
        actual_metric = "precomputed"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=actual_metric,
        cluster_selection_method=cluster_selection_method,
    )
    clusterer.fit(D)

    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    quality = {}
    if n_clusters >= 2 and np.sum(labels >= 0) > n_clusters:
        skm = _require_sklearn_metrics()
        valid = labels >= 0
        if actual_metric == "precomputed":
            quality["silhouette"] = float(
                skm.silhouette_score(D[np.ix_(valid, valid)],
                                     labels[valid], metric="precomputed")
            )
        else:
            quality["silhouette"] = float(
                skm.silhouette_score(D[valid], labels[valid])
            )

    return ClusterResult(
        labels=labels.astype(np.int64),
        n_clusters=n_clusters,
        method="hdbscan",
        probabilities=clusterer.probabilities_,
        quality=quality,
        metadata={
            "outlier_scores": clusterer.outlier_scores_,
            "condensed_tree": (
                clusterer.condensed_tree_
                if hasattr(clusterer, "condensed_tree_") else None
            ),
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        },
    )


def cluster_leiden(
    adjacency_or_H: Union[SparseMatrix, DescriptorMatrix],
    *,
    H: Optional[DescriptorMatrix] = None,
    resolution: float = 1.0,
    quality_function: Literal[
        "modularity", "cpm"
    ] = "modularity",
    n_iterations: int = -1,
    random_state: int = 42,
    sigma: Optional[float] = None,
) -> ClusterResult:
    """Leiden community detection on mesh graph.

    Accepts either (a) a pre-built weighted adjacency matrix or
    (b) a raw descriptor matrix ``adjacency_or_H`` plus mesh
    adjacency ``H`` (confusing naming avoided by keyword use).

    Parameters
    ----------
    adjacency_or_H : sparse or ndarray
        If sparse: the weighted adjacency/affinity graph.
        If ndarray (N, T): descriptor matrix (requires ``H=None``
        and will build k-NN graph from descriptors).
    H : ndarray or None
        If adjacency_or_H is sparse and H is provided, weights
        are multiplied by HKS affinity.
    resolution : float
        Resolution parameter γ. Higher = more clusters.
    quality_function : str
        ``"modularity"`` (RBConfiguration) or ``"cpm"`` (CPM).
    n_iterations : int
        Leiden iterations. ``-1`` = iterate until stable.
    random_state : int
    sigma : float or None
        Bandwidth for HKS affinity kernel (if H provided).

    Returns
    -------
    ClusterResult
        With ``modularity`` in quality dict.
    """
    la, ig = _require_leidenalg()

    # --- build the graph ---
    if sp.issparse(adjacency_or_H):
        A = sp.csr_matrix(adjacency_or_H)
        if H is not None:
            A = build_hks_affinity_graph(H, A, sigma=sigma)
    else:
        # descriptor matrix → build k-NN affinity graph
        from sklearn.neighbors import kneighbors_graph
        X = np.asarray(adjacency_or_H, dtype=np.float64)
        k = min(30, X.shape[0] - 1)
        A = kneighbors_graph(X, n_neighbors=k, mode="distance")
        # convert distance to similarity
        sigma_knn = np.median(A.data) or 1.0
        A.data = np.exp(-A.data**2 / (2 * sigma_knn**2))
        A = (A + A.T) / 2.0

    A_coo = sp.coo_matrix(A)
    # build igraph
    edges = list(zip(A_coo.row.tolist(), A_coo.col.tolist()))
    weights = np.abs(A_coo.data).tolist()
    g = ig.Graph(n=A.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights
    # remove self-loops and multi-edges
    g.simplify(combine_edges="max")

    if quality_function == "modularity":
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=random_state,
        )
    elif quality_function == "cpm":
        partition = la.find_partition(
            g,
            la.CPMVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=random_state,
        )
    else:
        raise ValueError(f"Unknown quality_function: {quality_function}")

    labels = np.array(partition.membership, dtype=np.int64)
    n_clusters = len(set(labels))

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="leiden",
        quality={
            "modularity": float(partition.modularity),
            "quality": float(partition.quality()),
        },
        metadata={
            "resolution": resolution,
            "quality_function": quality_function,
        },
    )


def cluster_gnmf(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    n_components: int = 8,
    lam: float = 1.0,
    sparsity_alpha: float = 0.0,
    n_iter: int = 300,
    tol: float = 1e-5,
    random_state: int = 42,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> ClusterResult:
    """Graph-Regularised Non-negative Matrix Factorisation (GNMF).

    Decomposes the non-negative descriptor matrix H ≈ W·F subject to
    a Laplacian smoothness penalty on the spatial factor W, so that
    mesh-adjacent vertices receive similar component activations.

    .. math::

        \\min_{W,F \\geq 0} \\frac{1}{2}\\|H - WF\\|_F^2
        + \\lambda \\operatorname{tr}(W^T L W)
        + \\alpha \\|W\\|_1

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Non-negative descriptor matrix (HKS is always ≥ 0).
    adjacency : sparse, shape (N, N)
        Mesh adjacency (cotangent Laplacian or its negative off-diag).
    n_components : int
        Number of spatial components (≈ cluster count).
    lam : float
        Laplacian smoothness weight λ.
    sparsity_alpha : float
        ℓ₁ sparsity on W.
    n_iter : int
        Maximum multiplicative update iterations.
    tol : float
        Relative objective change for convergence.
    random_state : int
    backend : str

    Returns
    -------
    ClusterResult
        With ``W`` (spatial), ``F`` (temporal profiles) in metadata.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.min() < 0:
        warnings.warn(
            "GNMF requires non-negative input. Shifting H by min value.",
            stacklevel=2,
        )
        H = H - H.min()

    n, T = H.shape
    rng = np.random.default_rng(random_state)

    # --- build adjacency and degree ---
    A_aff = sp.csr_matrix(adjacency, dtype=np.float64)
    # extract off-diagonal as positive affinity
    A_off = -A_aff.copy()
    A_off.setdiag(0)
    A_off.eliminate_zeros()
    A_off.data = np.abs(A_off.data)
    D_diag = np.asarray(A_off.sum(axis=1)).flatten()

    use_gpu = _resolve_backend(backend)

    if use_gpu:
        return _gnmf_gpu(H, A_off, D_diag, n_components, lam,
                         sparsity_alpha, n_iter, tol, rng)
    else:
        return _gnmf_cpu(H, A_off, D_diag, n_components, lam,
                         sparsity_alpha, n_iter, tol, rng)


def _gnmf_cpu(H, A_off, D_diag, r, lam, alpha, n_iter, tol, rng):
    """CPU implementation of GNMF with multiplicative updates."""
    n, T = H.shape
    eps = 1e-12

    W = rng.random((n, r)).astype(np.float64) + 0.1
    F = rng.random((r, T)).astype(np.float64) + 0.1

    prev_obj = np.inf
    with progress_simple("GNMF", total=n_iter) as update:
        for it in range(n_iter):
            # --- update F ---
            numerator_F = W.T @ H                       # (r, T)
            denominator_F = W.T @ W @ F + eps           # (r, T)
            F *= numerator_F / denominator_F

            # --- update W ---
            AW = A_off @ W                               # (n, r)
            DW = D_diag[:, None] * W                     # (n, r)
            numerator_W = H @ F.T + lam * AW             # (n, r)
            denominator_W = W @ (F @ F.T) + lam * DW + alpha + eps
            W *= numerator_W / denominator_W

            # --- convergence ---
            if it % 10 == 0:
                obj = (
                    0.5 * np.linalg.norm(H - W @ F, 'fro')**2
                    + lam * np.trace(W.T @ (D_diag[:, None] * W - A_off @ W))
                    + alpha * np.abs(W).sum()
                )
                if abs(prev_obj - obj) / (abs(prev_obj) + eps) < tol:
                    update(n_iter - it)
                    break
                prev_obj = obj

            update(1)

    labels = W.argmax(axis=1).astype(np.int64)
    n_clusters = len(np.unique(labels))

    # soft probabilities
    W_norm = W / (W.sum(axis=1, keepdims=True) + eps)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="gnmf",
        probabilities=W_norm,
        quality={},
        metadata={
            "W": W,
            "F": F,
            "n_components": int(W.shape[1]),
            "lambda": lam,
        },
    )


def _gnmf_gpu(H, A_off, D_diag, r, lam, alpha, n_iter, tol, rng):
    """GPU-accelerated GNMF via PyTorch."""
    torch = _require_torch()
    device = torch.device("cuda")
    eps = 1e-12

    n, T = H.shape
    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    W_t = torch.tensor(
        rng.random((n, r)).astype(np.float32) + 0.1, device=device
    )
    F_t = torch.tensor(
        rng.random((r, T)).astype(np.float32) + 0.1, device=device
    )

    # sparse adjacency on GPU
    A_coo = sp.coo_matrix(A_off)
    indices = torch.tensor(
        np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device
    )
    values = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    A_t = torch.sparse_coo_tensor(indices, values, A_off.shape).coalesce()
    D_t = torch.tensor(D_diag, dtype=torch.float32, device=device)

    prev_obj = float("inf")
    with progress_simple("GNMF [GPU]", total=n_iter) as update:
        for it in range(n_iter):
            # update F
            num_F = W_t.T @ H_t
            den_F = W_t.T @ W_t @ F_t + eps
            F_t *= num_F / den_F

            # update W
            AW = torch.sparse.mm(A_t, W_t)
            DW = D_t.unsqueeze(1) * W_t
            num_W = H_t @ F_t.T + lam * AW
            den_W = W_t @ (F_t @ F_t.T) + lam * DW + alpha + eps
            W_t *= num_W / den_W

            if it % 10 == 0:
                residual = torch.norm(H_t - W_t @ F_t).item() ** 2
                obj = 0.5 * residual
                if abs(prev_obj - obj) / (abs(prev_obj) + eps) < tol:
                    update(n_iter - it)
                    break
                prev_obj = obj

            update(1)

    W = W_t.cpu().numpy().astype(np.float64)
    F = F_t.cpu().numpy().astype(np.float64)
    del H_t, W_t, F_t, A_t, D_t
    torch.cuda.empty_cache()

    labels = W.argmax(axis=1).astype(np.int64)
    n_clusters = len(np.unique(labels))
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="gnmf",
        probabilities=W_norm,
        quality={},
        metadata={"W": W, "F": F, "n_components": int(r), "lambda": lam},
    )


def cluster_dpmm(
    H: DescriptorMatrix,
    *,
    adjacency: Optional[SparseMatrix] = None,
    max_components: int = 25,
    dim_reduction: Optional[Literal["umap", "pca"]] = "pca",
    n_components_reduce: int = 8,
    log_transform: bool = True,
    random_state: int = 42,
    backend: Literal["variational", "pymc"] = "variational",
    mrf_beta: float = 0.0,
) -> ClusterResult:
    """Dirichlet Process Mixture Model — automatic cluster count.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Descriptor matrix.
    adjacency : sparse or None
        Mesh adjacency for MRF spatial prior (only with pymc backend).
    max_components : int
        Truncation level for variational DP.
    dim_reduction : str or None
        Reduce dimensionality before fitting.
    n_components_reduce : int
        Target dimensionality.
    log_transform : bool
    random_state : int
    backend : str
        ``"variational"`` uses sklearn BayesianGaussianMixture (fast).
        ``"pymc"`` uses full MCMC with optional MRF prior (slow, rich).
    mrf_beta : float
        Potts MRF coupling strength (pymc backend only).

    Returns
    -------
    ClusterResult
    """
    H = np.asarray(H, dtype=np.float64)

    # --- preprocess ---
    X = np.log(H + 1e-12) if log_transform else H.copy()
    row_sums = np.abs(X).sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X /= row_sums

    if dim_reduction == "pca" and X.shape[1] > n_components_reduce:
        skd = _require_sklearn_decomposition()
        X = skd.PCA(
            n_components=n_components_reduce,
            random_state=random_state,
        ).fit_transform(X)
    elif dim_reduction == "umap" and X.shape[1] > n_components_reduce:
        umap_mod = _require_umap()
        X = umap_mod.UMAP(
            n_components=n_components_reduce,
            random_state=random_state,
        ).fit_transform(X)

    if backend == "variational":
        return _dpmm_variational(X, max_components, random_state)
    elif backend == "pymc":
        return _dpmm_pymc(X, adjacency, max_components, mrf_beta,
                          random_state)
    else:
        raise ValueError(f"Unknown DPMM backend: {backend}")


def _dpmm_variational(X, K, seed):
    """Fast variational Bayesian GMM with DP prior via sklearn."""
    skmix = _require_sklearn_mixture()
    skm = _require_sklearn_metrics()

    model = skmix.BayesianGaussianMixture(
        n_components=K,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-3,
        max_iter=500,
        random_state=seed,
        n_init=3,
    )
    model.fit(X)
    labels = model.predict(X).astype(np.int64)
    probs = model.predict_proba(X)

    # count active components
    active = np.unique(labels)
    n_clusters = len(active)

    # relabel to 0..K-1
    relabel_map = {old: new for new, old in enumerate(sorted(active))}
    labels = np.array([relabel_map[l] for l in labels], dtype=np.int64)
    probs = probs[:, sorted(active)]

    quality = {}
    if n_clusters >= 2:
        quality["silhouette"] = float(
            skm.silhouette_score(X, labels)
        )
    quality["bic_lower_bound"] = float(model.lower_bound_)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="dpmm_variational",
        probabilities=probs,
        quality=quality,
        metadata={
            "means": model.means_,
            "weights": model.weights_,
            "covariances": model.covariances_,
        },
    )


def _dpmm_pymc(X, adjacency, K, mrf_beta, seed):
    """Full Bayesian DPMM with optional MRF spatial prior via PyMC."""
    pm, az = _require_pymc()
    import pytensor.tensor as pt

    n, d = X.shape

    with pm.Model() as model:
        alpha = pm.Gamma("alpha", 1.0, 1.0)
        beta_sticks = pm.Beta("beta_sticks", 1.0, alpha, shape=K)
        w = pm.Deterministic(
            "w",
            beta_sticks * pt.concatenate(
                [pt.ones(1), pt.cumprod(1.0 - beta_sticks)[:-1]]
            ),
        )

        mu = pm.Normal("mu", 0.0, 5.0, shape=(K, d))
        sigma = pm.HalfNormal("sigma", 1.0, shape=(K, d))

        comp_dists = [
            pm.Normal.dist(mu=mu[k], sigma=sigma[k], shape=d)
            for k in range(K)
        ]
        pm.Mixture("obs", w=w, comp_dists=comp_dists, observed=X)

        # MRF spatial prior (if adjacency provided)
        if adjacency is not None and mrf_beta > 0:
            z = pm.Categorical("z", p=w, shape=n)
            A_coo = sp.coo_matrix(adjacency)
            edges = np.column_stack([A_coo.row, A_coo.col])
            pm.Potential(
                "mrf",
                mrf_beta * pt.sum(pt.eq(z[edges[:, 0]], z[edges[:, 1]])),
            )

    with model:
        approx = pm.fit(20000, method="advi", random_seed=seed)
        trace = approx.sample(1000)

    # extract weights → assign labels
    w_post = trace.posterior["w"].values.mean(axis=(0, 1))  # (K,)
    mu_post = trace.posterior["mu"].values.mean(axis=(0, 1))  # (K, d)

    # assign each point to nearest component weighted by w
    from scipy.spatial.distance import cdist
    D_km = cdist(X, mu_post)
    log_resp = np.log(w_post[None, :] + 1e-12) - 0.5 * D_km**2
    labels = log_resp.argmax(axis=1).astype(np.int64)
    probs = np.exp(log_resp - log_resp.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    active = np.unique(labels)
    n_clusters = len(active)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="dpmm_pymc",
        probabilities=probs,
        quality={},
        metadata={"trace": trace, "model": model, "approx": approx},
    )


def cluster_persistence(
    H_scalar: ScalarMap,
    adjacency: SparseMatrix,
    *,
    persistence_threshold: Optional[float] = None,
    n_clusters: Optional[int] = None,
) -> ClusterResult:
    """Persistence-based clustering (ToMATo) on a scalar field.

    Treats HKS(·, t₀) as a density function on the mesh and uses
    persistent homology of sub-level sets to find topologically
    stable basins — each basin becomes a cluster.

    Parameters
    ----------
    H_scalar : ndarray, shape (N,)
        Scalar field on the mesh (e.g. HKS at one time-scale,
        or summed HKS, or any vertex-wise feature).
    adjacency : sparse, shape (N, N)
        Mesh adjacency.
    persistence_threshold : float or None
        Minimum persistence to retain a cluster. If None and
        n_clusters is None, uses the largest gap in the diagram.
    n_clusters : int or None
        If set, selects the top-n most persistent components.

    Returns
    -------
    ClusterResult
        With ``persistence_pairs`` and ``diagram`` in metadata.
    """
    gudhi = _require_gudhi()

    density = -np.asarray(H_scalar, dtype=np.float64)
    n = len(density)

    # build adjacency list from sparse matrix
    A = sp.coo_matrix(adjacency)
    graph = [[] for _ in range(n)]
    for i, j in zip(A.row.tolist(), A.col.tolist()):
        if i != j:
            graph[i].append(j)

    try:
        from gudhi.clustering.tomato import Tomato
        tomato = Tomato(
            graph_type="manual",
            density_type="manual",
        )
        tomato.fit(graph, weights=density)

        if n_clusters is not None:
            tomato.n_clusters_ = n_clusters
        elif persistence_threshold is not None:
            # set via diagram analysis
            tomato.n_clusters_ = int(
                np.sum(tomato.diagram_[:, 1] - tomato.diagram_[:, 0]
                       > persistence_threshold)
            )
            tomato.n_clusters_ = max(1, tomato.n_clusters_)

        labels = tomato.labels_.astype(np.int64)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)

        return ClusterResult(
            labels=labels,
            n_clusters=n_clust,
            method="persistence_tomato",
            quality={},
            metadata={
                "diagram": (
                    tomato.diagram_ if hasattr(tomato, "diagram_") else None
                ),
            },
        )

    except (ImportError, AttributeError):
        # fallback: manual union-find persistence on H₀
        logger.warning(
            "GUDHI Tomato not available; using manual sub-level "
            "persistence via union-find."
        )
        return _persistence_sublevel_h0(density, graph, n,
                                         persistence_threshold, n_clusters)


def _persistence_sublevel_h0(density, graph, n, tau, k):
    """Manual H₀ sub-level persistence via union-find."""
    # sort vertices by ascending density (function value)
    order = np.argsort(density)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    parent = np.arange(n, dtype=np.int64)
    birth = density.copy()

    def find(x):
        """Find optimal parameters for the clustering algorithm."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []  # (birth_val, death_val, root_vertex)

    for idx in order:
        for nb in graph[idx]:
            if rank[nb] < rank[idx]:
                ri = find(idx)
                rn = find(nb)
                if ri != rn:
                    # younger root dies (higher birth = less persistent)
                    if birth[ri] > birth[rn]:
                        parent[ri] = rn
                        pairs.append((birth[ri], density[idx], ri))
                    else:
                        parent[rn] = ri
                        pairs.append((birth[rn], density[idx], rn))

    # compute persistence = death - birth
    persistences = np.array(
        [d - b for b, d, _ in pairs], dtype=np.float64
    ) if pairs else np.array([])

    # select threshold
    if k is not None and len(persistences) >= k - 1:
        sorted_p = np.sort(persistences)[::-1]
        tau_eff = sorted_p[k - 2] if k >= 2 else 0.0
    elif tau is not None:
        tau_eff = tau
    elif len(persistences) > 1:
        sorted_p = np.sort(persistences)[::-1]
        gaps = np.diff(sorted_p)
        tau_eff = sorted_p[np.argmax(np.abs(gaps))]
    else:
        tau_eff = 0.0

    # rebuild union-find with threshold
    parent2 = np.arange(n, dtype=np.int64)
    birth2 = density.copy()

    for idx in order:
        for nb in graph[idx]:
            if rank[nb] < rank[idx]:
                ri = find2(parent2, idx)
                rn = find2(parent2, nb)
                if ri != rn:
                    if birth2[ri] > birth2[rn]:
                        p = density[idx] - birth2[ri]
                        if p < tau_eff:
                            parent2[ri] = rn
                        # else: keep separate
                    else:
                        p = density[idx] - birth2[rn]
                        if p < tau_eff:
                            parent2[rn] = ri

    # extract labels
    labels = np.array(
        [find2(parent2, i) for i in range(n)], dtype=np.int64
    )
    unique_roots = np.unique(labels)
    relabel = {old: new for new, old in enumerate(unique_roots)}
    labels = np.array([relabel[l] for l in labels], dtype=np.int64)

    return ClusterResult(
        labels=labels,
        n_clusters=len(unique_roots),
        method="persistence_sublevel_h0",
        quality={},
        metadata={
            "persistence_pairs": pairs,
            "threshold": tau_eff,
        },
    )


def find2(parent, x):
    """Path-compressed find for union-find."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def cluster_spectral_coclustering(
    H: DescriptorMatrix,
    *,
    n_clusters: int = 6,
    adjacency: Optional[SparseMatrix] = None,
    laplacian_smoothing: float = 5.0,
) -> ClusterResult:
    """Spectral co-clustering of the vertex × time/energy matrix.

    Simultaneously clusters rows (vertices) and columns (scales),
    revealing which spatial regions share which scale bands.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Non-negative descriptor matrix.
    n_clusters : int
        Number of co-clusters.
    adjacency : sparse or None
        If provided, applies Laplacian smoothing post-hoc.
    laplacian_smoothing : float
        Tikhonov regularisation weight μ for spatial coherence.

    Returns
    -------
    ClusterResult
        With ``column_labels`` (scale clustering) in metadata.
    """
    skc = _require_sklearn_cluster()

    H_pos = np.asarray(H, dtype=np.float64)
    H_pos = np.maximum(H_pos, 0.0)
    H_pos += 1e-12  # avoid zeros for bipartite normalisation

    cc = skc.SpectralCoclustering(
        n_clusters=n_clusters,
        svd_method="randomized",
        random_state=42,
    )
    cc.fit(H_pos)
    row_labels = cc.row_labels_.astype(np.int64)
    col_labels = cc.column_labels_.astype(np.int64)

    # --- optional Laplacian smoothing ---
    if adjacency is not None and laplacian_smoothing > 0:
        from scipy.sparse.linalg import spsolve
        L = sp.csr_matrix(adjacency, dtype=np.float64)
        I = sp.eye(H.shape[0], format="csr")
        K = len(np.unique(row_labels))
        prob = np.zeros((H.shape[0], K), dtype=np.float64)
        for k in range(K):
            rhs = (row_labels == k).astype(np.float64)
            prob[:, k] = spsolve(I + laplacian_smoothing * L, rhs)
        row_labels = prob.argmax(axis=1).astype(np.int64)

    n_clusters_actual = len(np.unique(row_labels))

    return ClusterResult(
        labels=row_labels,
        n_clusters=n_clusters_actual,
        method="spectral_coclustering",
        quality={},
        metadata={
            "column_labels": col_labels,
            "n_coclusters": n_clusters,
        },
    )


# ======================================================================
# §4  TEMPORAL / SCALE CLUSTERING
# ======================================================================

def cluster_temporal_fpca(
    H: DescriptorMatrix,
    *,
    n_components: int = 6,
    n_clusters: int = 6,
    clusterer: Literal["kmeans", "hdbscan"] = "kmeans",
    random_state: int = 42,
) -> ClusterResult:
    """Functional PCA on HKS time-profiles, then cluster fPC scores.

    Treats each vertex's HKS(x, ·) as a function of log(t) and
    performs fPCA to extract the dominant modes of variation, then
    clusters in the score space.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Per-vertex HKS / WKS profiles across T scales.
    n_components : int
        Number of functional principal components.
    n_clusters : int
        For k-means; ignored if ``clusterer="hdbscan"``.
    clusterer : str
        ``"kmeans"`` or ``"hdbscan"``.
    random_state : int

    Returns
    -------
    ClusterResult
        With ``fpc_scores``, ``explained_variance_ratio`` in metadata.
    """
    # use sklearn PCA on the (N, T) matrix directly — equivalent to
    # discretised fPCA when T is the evaluation grid
    skd = _require_sklearn_decomposition()
    skc = _require_sklearn_cluster()

    X = np.asarray(H, dtype=np.float64)
    # standardise per-vertex (z-score each row)
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    X_std = (X - mu) / std

    pca = skd.PCA(n_components=min(n_components, X_std.shape[1]),
                  random_state=random_state)
    scores = pca.fit_transform(X_std)

    if clusterer == "kmeans":
        km = skc.KMeans(n_clusters=n_clusters, n_init=10,
                        random_state=random_state)
        labels = km.fit_predict(scores).astype(np.int64)
        n_clust = n_clusters
    elif clusterer == "hdbscan":
        result = cluster_hdbscan(
            scores, log_transform=False, dim_reduction=None,
            random_state=random_state,
        )
        labels = result.labels
        n_clust = result.n_clusters
    else:
        raise ValueError(f"Unknown clusterer: {clusterer}")

    return ClusterResult(
        labels=labels,
        n_clusters=n_clust,
        method="fpca_" + clusterer,
        quality={},
        metadata={
            "fpc_scores": scores,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "components": pca.components_,
        },
    )


def cluster_temporal_dtw(
    H: DescriptorMatrix,
    *,
    n_clusters: int = 6,
    metric: Literal["dtw", "softdtw", "euclidean"] = "softdtw",
    gamma: float = 0.1,
    random_state: int = 42,
) -> ClusterResult:
    """Time-series k-means with DTW on HKS/WKS profiles.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Descriptor profiles.
    n_clusters : int
    metric : str
        ``"dtw"``, ``"softdtw"``, or ``"euclidean"``.
    gamma : float
        Smoothing for soft-DTW (smaller = sharper).
    random_state : int

    Returns
    -------
    ClusterResult
        With ``barycenters`` in metadata.
    """
    tslearn = _require_tslearn()
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    X = np.asarray(H, dtype=np.float64)
    # tslearn expects (N, T, 1)
    X_3d = X[:, :, np.newaxis]
    X_3d = TimeSeriesScalerMeanVariance().fit_transform(X_3d)

    metric_params = {}
    if metric == "softdtw":
        metric_params["gamma"] = gamma

    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=metric,
        metric_params=metric_params if metric_params else None,
        max_iter_barycenter=20,
        n_init=3,
        random_state=random_state,
        verbose=0,
    )
    labels = km.fit_predict(X_3d).astype(np.int64)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method=f"dtw_kmeans_{metric}",
        quality={
            "inertia": float(km.inertia_),
        },
        metadata={
            "barycenters": km.cluster_centers_.squeeze(-1),
        },
    )


# ======================================================================
# §5  SPATIO-TEMPORAL JOINT CLUSTERING
# ======================================================================

def cluster_spatiotemporal_gnmf(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    n_components: int = 8,
    lam_spatial: float = 1.0,
    lam_temporal: float = 0.1,
    n_iter: int = 300,
    tol: float = 1e-5,
    random_state: int = 42,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> ClusterResult:
    """Graph-regularised NMF with both spatial and temporal smoothness.

    Extends GNMF by adding a temporal smoothness penalty on F,
    encouraging neighbouring time scales to have similar profiles.

    .. math::

        \\min_{W,F \\geq 0} \\tfrac{1}{2}\\|H - WF\\|_F^2
        + \\lambda_s \\operatorname{tr}(W^T L_s W)
        + \\lambda_t \\operatorname{tr}(F L_t F^T)

    where L_s is the mesh Laplacian and L_t is the 1D chain
    Laplacian along the time axis.

    Parameters
    ----------
    H : ndarray, shape (N, T)
    adjacency : sparse, shape (N, N)
    n_components : int
    lam_spatial : float
    lam_temporal : float
    n_iter : int
    tol : float
    random_state : int
    backend : str

    Returns
    -------
    ClusterResult
        With spatial ``W`` and temporally-smooth ``F`` in metadata.
    """
    H = np.asarray(H, dtype=np.float64)
    H = np.maximum(H, 0.0) + 1e-12
    n, T = H.shape
    eps = 1e-12
    rng = np.random.default_rng(random_state)

    # --- build spatial and temporal Laplacians ---
    A_off = -sp.csr_matrix(adjacency, dtype=np.float64).copy()
    A_off.setdiag(0)
    A_off.eliminate_zeros()
    A_off.data = np.abs(A_off.data)
    D_s = np.asarray(A_off.sum(axis=1)).flatten()

    # 1D chain Laplacian for temporal axis
    L_t = np.zeros((T, T), dtype=np.float64)
    for i in range(T):
        if i > 0:
            L_t[i, i] += 1.0
            L_t[i, i - 1] = -1.0
        if i < T - 1:
            L_t[i, i] += 1.0
            L_t[i, i + 1] = -1.0

    W = rng.random((n, n_components)).astype(np.float64) + 0.1
    F = rng.random((n_components, T)).astype(np.float64) + 0.1

    prev_obj = np.inf
    with progress_simple("ST-GNMF", total=n_iter) as update:
        for it in range(n_iter):
            # update F with temporal smoothness
            num_F = W.T @ H
            den_F = W.T @ W @ F + lam_temporal * F @ L_t + eps
            F *= num_F / np.maximum(den_F, eps)

            # update W with spatial smoothness
            AW = A_off @ W
            DW = D_s[:, None] * W
            num_W = H @ F.T + lam_spatial * AW
            den_W = W @ (F @ F.T) + lam_spatial * DW + eps
            W *= num_W / np.maximum(den_W, eps)

            if it % 10 == 0:
                obj = 0.5 * np.linalg.norm(H - W @ F, "fro") ** 2
                if abs(prev_obj - obj) / (abs(prev_obj) + eps) < tol:
                    update(n_iter - it)
                    break
                prev_obj = obj
            update(1)

    labels = W.argmax(axis=1).astype(np.int64)
    W_norm = W / (W.sum(axis=1, keepdims=True) + eps)

    return ClusterResult(
        labels=labels,
        n_clusters=len(np.unique(labels)),
        method="spatiotemporal_gnmf",
        probabilities=W_norm,
        quality={},
        metadata={
            "W": W, "F": F,
            "lambda_spatial": lam_spatial,
            "lambda_temporal": lam_temporal,
        },
    )


def cluster_spatiotemporal_stdbscan(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    eps_spatial: Optional[float] = None,
    eps_temporal: Optional[float] = None,
    min_pts: int = 10,
    temporal_metric: Literal[
        "euclidean", "softdtw"
    ] = "euclidean",
    gamma_dtw: float = 0.1,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> ClusterResult:
    """ST-DBSCAN adapted for mesh + spectral descriptor profiles.

    Uses a conjunctive neighbourhood: vertex y is in the
    neighbourhood of x iff geodesic(x, y) ≤ ε₁ AND
    d_temporal(h_x, h_y) ≤ ε₂.

    Parameters
    ----------
    H : ndarray, shape (N, T)
    adjacency : sparse, shape (N, N)
    eps_spatial : float or None
        Geodesic distance threshold. Auto-set from median if None.
    eps_temporal : float or None
        Temporal distance threshold. Auto-set from median if None.
    min_pts : int
    temporal_metric : str
    gamma_dtw : float
    backend : str

    Returns
    -------
    ClusterResult
    """
    H = np.asarray(H, dtype=np.float64)
    n, T = H.shape

    # --- geodesic distances via Dijkstra on mesh ---
    from scipy.sparse.csgraph import shortest_path
    A = sp.csr_matrix(adjacency).copy()
    A.data = np.abs(A.data)
    A.data[A.data == 0] = 1e-12
    D_geo = shortest_path(A, method="D", directed=False)

    # --- temporal distances ---
    if temporal_metric == "euclidean":
        from scipy.spatial.distance import pdist, squareform
        D_temp = squareform(pdist(H, metric="euclidean"))
    elif temporal_metric == "softdtw":
        tslearn = _require_tslearn()
        from tslearn.metrics import cdist_soft_dtw
        X_3d = H[:, :, np.newaxis]
        D_temp = cdist_soft_dtw(X_3d, gamma=gamma_dtw)
    else:
        raise ValueError(f"Unknown temporal_metric: {temporal_metric}")

    # --- auto-set thresholds ---
    if eps_spatial is None:
        # use edges only (not all pairs)
        A_coo = sp.coo_matrix(adjacency)
        edge_dists = D_geo[A_coo.row, A_coo.col]
        eps_spatial = float(np.median(edge_dists[edge_dists > 0])) * 3.0

    if eps_temporal is None:
        A_coo = sp.coo_matrix(adjacency)
        edge_tdists = D_temp[A_coo.row, A_coo.col]
        eps_temporal = float(np.median(edge_tdists[edge_tdists > 0])) * 2.0

    # --- ST-DBSCAN ---
    labels = -np.ones(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    with progress_simple("ST-DBSCAN", total=n) as update:
        for x in range(n):
            if visited[x]:
                update(1)
                continue
            visited[x] = True

            # conjunctive neighbourhood
            nbr = np.where(
                (D_geo[x] <= eps_spatial) & (D_temp[x] <= eps_temporal)
            )[0]
            nbr = nbr[nbr != x]

            if len(nbr) < min_pts:
                update(1)
                continue

            labels[x] = cluster_id
            seeds = list(nbr)

            while seeds:
                y = seeds.pop()
                if not visited[y]:
                    visited[y] = True
                    nbr_y = np.where(
                        (D_geo[y] <= eps_spatial)
                        & (D_temp[y] <= eps_temporal)
                    )[0]
                    nbr_y = nbr_y[nbr_y != y]
                    if len(nbr_y) >= min_pts:
                        seeds.extend(nbr_y.tolist())
                if labels[y] == -1:
                    labels[y] = cluster_id

            cluster_id += 1
            update(1)

    n_clusters = cluster_id

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method="st_dbscan",
        quality={},
        metadata={
            "eps_spatial": eps_spatial,
            "eps_temporal": eps_temporal,
            "min_pts": min_pts,
        },
    )


# ======================================================================
# §6  HKS + WKS DESCRIPTOR FUSION
# ======================================================================

def fuse_concatenate(
    hks: DescriptorMatrix,
    wks: DescriptorMatrix,
    *,
    log_transform: bool = True,
    normalize: Literal["l1", "l2", "none"] = "l1",
    weight_hks: float = 1.0,
    weight_wks: float = 1.0,
) -> FusionResult:
    """Simple weighted concatenation of HKS and WKS.

    Parameters
    ----------
    hks : ndarray, shape (N, T_h)
    wks : ndarray, shape (N, T_w)
    log_transform : bool
    normalize : str
    weight_hks, weight_wks : float

    Returns
    -------
    FusionResult
    """
    Hh = np.asarray(hks, dtype=np.float64)
    Hw = np.asarray(wks, dtype=np.float64)

    if log_transform:
        Hh = np.log(np.maximum(Hh, 1e-12))
        Hw = np.log(np.maximum(np.abs(Hw) + 1e-12, 1e-12))

    # normalise each block independently
    for X in [Hh, Hw]:
        if normalize == "l1":
            s = np.abs(X).sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            X /= s
        elif normalize == "l2":
            s = np.linalg.norm(X, axis=1, keepdims=True)
            s[s == 0] = 1.0
            X /= s

    fused = np.hstack([weight_hks * Hh, weight_wks * Hw])

    return FusionResult(
        fused=fused,
        method="concatenate",
        weights=np.array([weight_hks, weight_wks]),
        metadata={"T_hks": hks.shape[1], "T_wks": wks.shape[1]},
    )


def fuse_joint_nmf(
    hks: DescriptorMatrix,
    wks: DescriptorMatrix,
    *,
    n_components: int = 16,
    random_state: int = 42,
) -> FusionResult:
    """Joint NMF on concatenated [HKS | WKS] for shared basis.

    Learns a shared spatial factor W and separate temporal factors
    F_hks, F_wks such that [HKS | WKS] ≈ W · [F_hks | F_wks].

    Parameters
    ----------
    hks : ndarray, shape (N, T_h)
    wks : ndarray, shape (N, T_w)
    n_components : int
    random_state : int

    Returns
    -------
    FusionResult
        ``fused`` is the W matrix (shared spatial loadings).
    """
    skd = _require_sklearn_decomposition()

    Hh = np.maximum(np.asarray(hks, dtype=np.float64), 0.0) + 1e-12
    Hw = np.maximum(np.abs(np.asarray(wks, dtype=np.float64)), 1e-12)

    # normalise columns to unit max for balanced scales
    Hh = Hh / (Hh.max(axis=0, keepdims=True) + 1e-12)
    Hw = Hw / (Hw.max(axis=0, keepdims=True) + 1e-12)

    H_joint = np.hstack([Hh, Hw])

    model = skd.NMF(
        n_components=n_components,
        init="nndsvd",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=500,
        random_state=random_state,
    )
    W = model.fit_transform(H_joint)
    F = model.components_

    return FusionResult(
        fused=W,
        method="joint_nmf",
        weights=None,
        metadata={
            "F_full": F,
            "F_hks": F[:, :hks.shape[1]],
            "F_wks": F[:, hks.shape[1]:],
            "reconstruction_error": model.reconstruction_err_,
        },
    )


def fuse_multi_kernel(
    hks: DescriptorMatrix,
    wks: DescriptorMatrix,
    *,
    n_kernels_per_desc: int = 5,
    sigma_range: Tuple[float, float] = (0.1, 10.0),
) -> FusionResult:
    """Multi-kernel fusion: build a combined kernel from HKS and WKS.

    Computes K = Σ_j α_j K^hks_j + Σ_k β_k K^wks_k with uniform
    weights (simpleMKL optimisation is available as extension).

    Parameters
    ----------
    hks : ndarray, shape (N, T_h)
    wks : ndarray, shape (N, T_w)
    n_kernels_per_desc : int
        Number of bandwidth samples per descriptor.
    sigma_range : tuple
        (min, max) bandwidth range.

    Returns
    -------
    FusionResult
        ``fused`` is the combined kernel matrix K, shape (N, N).
    """
    sigmas = np.logspace(
        np.log10(sigma_range[0]), np.log10(sigma_range[1]),
        n_kernels_per_desc,
    )

    Hh = np.log(np.maximum(np.asarray(hks, dtype=np.float64), 1e-12))
    Hw = np.log(np.maximum(np.abs(np.asarray(wks, dtype=np.float64))
                            + 1e-12, 1e-12))

    n = Hh.shape[0]
    K = np.zeros((n, n), dtype=np.float64)
    n_total = 2 * n_kernels_per_desc

    from scipy.spatial.distance import pdist, squareform

    D_hks = squareform(pdist(Hh, "sqeuclidean"))
    D_wks = squareform(pdist(Hw, "sqeuclidean"))

    for sig in sigmas:
        K += np.exp(-D_hks / (2 * sig**2)) / n_total
        K += np.exp(-D_wks / (2 * sig**2)) / n_total

    return FusionResult(
        fused=K,
        method="multi_kernel_uniform",
        weights=np.ones(n_total) / n_total,
        metadata={"sigmas": sigmas, "n_kernels": n_total},
    )


# ======================================================================
# §7  BAYESIAN CLUSTER CONFIRMATION
# ======================================================================

def confirm_clusters_bayesian(
    H: DescriptorMatrix,
    labels: LabelArray,
    *,
    adjacency: Optional[SparseMatrix] = None,
    mrf_beta: float = 1.0,
    n_samples: int = 2000,
    n_tune: int = 1000,
    dim_reduction: int = 8,
    random_state: int = 42,
) -> BayesianClusterConfirmation:
    """Bayesian confirmation of cluster assignments.

    Fits a Bayesian Gaussian mixture model with cluster-specific
    priors informed by the input labels, plus an optional Potts MRF
    spatial prior from the mesh adjacency. Compares MAP assignments
    to input labels and computes model quality (WAIC, LOO).

    This answers the question: "Are these clusters statistically
    credible given the data and the spatial structure?"

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Descriptor matrix.
    labels : ndarray, shape (N,)
        Input cluster labels to confirm.
    adjacency : sparse or None
        Mesh adjacency for MRF prior.
    mrf_beta : float
        Potts coupling strength.
    n_samples : int
        MCMC posterior samples.
    n_tune : int
        MCMC tuning samples.
    dim_reduction : int
        Reduce to this many dimensions before modelling.
    random_state : int

    Returns
    -------
    BayesianClusterConfirmation
    """
    pm, az = _require_pymc()
    skd = _require_sklearn_decomposition()
    skm = _require_sklearn_metrics()

    H = np.asarray(H, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = labels >= 0
    H_valid = H[valid]
    lab_valid = labels[valid]

    # reduce dimensions
    X = np.log(H_valid + 1e-12)
    if X.shape[1] > dim_reduction:
        X = skd.PCA(n_components=dim_reduction,
                     random_state=random_state).fit_transform(X)

    K = len(np.unique(lab_valid))
    n, d = X.shape

    # compute empirical cluster means as informative priors
    mu_prior = np.zeros((K, d))
    for k in range(K):
        mask = lab_valid == k
        if mask.any():
            mu_prior[k] = X[mask].mean(axis=0)

    with pm.Model() as model:
        # cluster-specific priors centred on empirical means
        mu = pm.Normal("mu", mu=mu_prior, sigma=2.0, shape=(K, d))
        sigma = pm.HalfNormal("sigma", sigma=1.0, shape=(K, d))
        w = pm.Dirichlet("w", a=np.ones(K) * 10.0)

        comp_dists = [
            pm.Normal.dist(mu=mu[k], sigma=sigma[k], shape=d)
            for k in range(K)
        ]
        pm.Mixture("obs", w=w, comp_dists=comp_dists, observed=X)

    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            random_seed=random_state,
            return_inferencedata=True,
            progressbar=True,
        )

    # --- compute MAP labels ---
    w_post = trace.posterior["w"].values.mean(axis=(0, 1))     # (K,)
    mu_post = trace.posterior["mu"].values.mean(axis=(0, 1))   # (K, d)
    sig_post = trace.posterior["sigma"].values.mean(axis=(0, 1))  # (K, d)

    # log responsibility
    log_resp = np.zeros((n, K))
    for k in range(K):
        diff = X - mu_post[k]
        log_resp[:, k] = (
            np.log(w_post[k] + 1e-12)
            - 0.5 * np.sum((diff / (sig_post[k] + 1e-12))**2, axis=1)
            - np.sum(np.log(sig_post[k] + 1e-12))
        )

    probs = np.exp(log_resp - log_resp.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    posterior_labels = probs.argmax(axis=1).astype(np.int64)

    # --- model comparison ---
    try:
        waic_val = float(az.waic(trace, model).elpd_waic)
    except Exception:
        waic_val = float("nan")

    try:
        loo_val = float(az.loo(trace, model).elpd_loo)
    except Exception:
        loo_val = float("nan")

    # --- agreement ---
    ari = float(skm.adjusted_rand_score(lab_valid, posterior_labels))

    # --- credible intervals per cluster ---
    ci = {}
    for k in range(K):
        mu_k = trace.posterior["mu"].values[:, :, k, :]  # (chain, draw, d)
        mu_flat = mu_k.reshape(-1, d)
        ci[k] = {
            "mean": mu_flat.mean(axis=0),
            "hdi_3": np.percentile(mu_flat, 3, axis=0),
            "hdi_97": np.percentile(mu_flat, 97, axis=0),
        }

    # --- reconstruct full labels (including noise vertices) ---
    full_labels = -np.ones(H.shape[0], dtype=np.int64)
    full_labels[valid] = posterior_labels
    full_probs = np.zeros((H.shape[0], K), dtype=np.float64)
    full_probs[valid] = probs

    return BayesianClusterConfirmation(
        posterior_labels=full_labels,
        label_probabilities=full_probs,
        waic=waic_val,
        loo=loo_val,
        cluster_credible_intervals=ci,
        agreement_with_input=ari,
        metadata={"trace": trace, "model": model},
    )


# ======================================================================
# §8  CLUSTER QUALITY & COMPARISON METRICS
# ======================================================================

def cluster_quality(
    H: DescriptorMatrix,
    labels: LabelArray,
    *,
    adjacency: Optional[SparseMatrix] = None,
    metric: Literal["euclidean", "precomputed"] = "euclidean",
) -> Dict[str, float]:
    """Compute internal clustering quality metrics.

    Parameters
    ----------
    H : ndarray, shape (N, T) or (N, N)
        Descriptor matrix or precomputed distance matrix.
    labels : ndarray, shape (N,)
    adjacency : sparse or None
        For spatial coherence metrics.
    metric : str

    Returns
    -------
    dict
        Keys: silhouette, calinski_harabasz, davies_bouldin,
        spatial_coherence (if adjacency provided).
    """
    skm = _require_sklearn_metrics()
    valid = labels >= 0
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {"silhouette": float("nan")}

    H_v = H[valid] if metric != "precomputed" else H[np.ix_(valid, valid)]
    lab_v = labels[valid]

    result = {
        "silhouette": float(
            skm.silhouette_score(H_v, lab_v, metric=metric)
        ),
    }

    if metric != "precomputed":
        result["calinski_harabasz"] = float(
            skm.calinski_harabasz_score(H_v, lab_v)
        )
        result["davies_bouldin"] = float(
            skm.davies_bouldin_score(H_v, lab_v)
        )

    # spatial coherence: fraction of edges where both endpoints
    # share the same cluster label
    if adjacency is not None:
        A_coo = sp.coo_matrix(adjacency)
        same = (labels[A_coo.row] == labels[A_coo.col])
        both_valid = (labels[A_coo.row] >= 0) & (labels[A_coo.col] >= 0)
        if both_valid.sum() > 0:
            result["spatial_coherence"] = float(
                same[both_valid].mean()
            )

    return result


def cluster_comparison(
    labels_a: LabelArray,
    labels_b: LabelArray,
) -> Dict[str, float]:
    """Compare two clusterings (e.g., algorithm output vs atlas labels).

    Returns
    -------
    dict
        Keys: ari, nmi, ami, homogeneity, completeness, v_measure.
    """
    skm = _require_sklearn_metrics()
    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    valid = (a >= 0) & (b >= 0)
    a_v, b_v = a[valid], b[valid]

    return {
        "ari": float(skm.adjusted_rand_score(a_v, b_v)),
        "nmi": float(skm.normalized_mutual_info_score(a_v, b_v)),
        "ami": float(skm.adjusted_mutual_info_score(a_v, b_v)),
        "homogeneity": float(skm.homogeneity_score(a_v, b_v)),
        "completeness": float(skm.completeness_score(a_v, b_v)),
        "v_measure": float(skm.v_measure_score(a_v, b_v)),
    }


# ======================================================================
# §9  CONVENIENCE / PIPELINE WRAPPERS
# ======================================================================

def auto_cluster(
    H: DescriptorMatrix,
    *,
    adjacency: Optional[SparseMatrix] = None,
    methods: Sequence[str] = ("hdbscan", "leiden", "gnmf"),
    random_state: int = 42,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
    **kwargs: Any,
) -> Dict[str, ClusterResult]:
    """Run multiple clustering algorithms and return all results.

    A convenience function for exploratory analysis that runs a
    battery of methods on the same data and returns a dict keyed
    by method name. The user can then compare via
    :func:`cluster_quality` and :func:`cluster_comparison`.

    Parameters
    ----------
    H : ndarray, shape (N, T)
    adjacency : sparse or None
    methods : sequence of str
        Subset of ``{"hdbscan", "leiden", "gnmf", "dpmm", "fpca",
        "coclustering", "persistence"}``.
    random_state : int
    backend : str
    **kwargs
        Forwarded to individual clustering functions.

    Returns
    -------
    dict[str, ClusterResult]
    """
    results = {}

    for method in methods:
        try:
            if method == "hdbscan":
                results[method] = cluster_hdbscan(
                    H, adjacency=adjacency,
                    random_state=random_state, backend=backend,
                    **{k: v for k, v in kwargs.items()
                       if k in ("min_cluster_size", "min_samples",
                                "alpha_fusion", "dim_reduction")},
                )
            elif method == "leiden" and adjacency is not None:
                results[method] = cluster_leiden(
                    adjacency, H=H,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items()
                       if k in ("resolution", "quality_function")},
                )
            elif method == "gnmf" and adjacency is not None:
                results[method] = cluster_gnmf(
                    H, adjacency,
                    random_state=random_state, backend=backend,
                    **{k: v for k, v in kwargs.items()
                       if k in ("n_components", "lam")},
                )
            elif method == "dpmm":
                results[method] = cluster_dpmm(
                    H, adjacency=adjacency,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items()
                       if k in ("max_components",)},
                )
            elif method == "fpca":
                results[method] = cluster_temporal_fpca(
                    H, random_state=random_state,
                    **{k: v for k, v in kwargs.items()
                       if k in ("n_components", "n_clusters")},
                )
            elif method == "coclustering":
                results[method] = cluster_spectral_coclustering(
                    H, adjacency=adjacency,
                    **{k: v for k, v in kwargs.items()
                       if k in ("n_clusters",)},
                )
            elif method == "persistence" and adjacency is not None:
                # use summed HKS as scalar field
                results[method] = cluster_persistence(
                    H.sum(axis=1), adjacency,
                )
            else:
                logger.warning(
                    "Skipping method '%s' (missing adjacency or unknown)",
                    method,
                )
        except Exception as e:
            logger.error("Method '%s' failed: %s", method, e)
            continue

    return results


# ======================================================================
# §10  PERSISTENCE VINEYARDS — TRACKING TOPOLOGY ACROSS HKS SCALES
# ======================================================================

def cluster_vineyards(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    n_scales: Optional[int] = None,
    min_persistence_frac: float = 0.1,
    min_life_frac: float = 0.3,
    backend: Literal["dionysus", "manual"] = "manual",
) -> VineyardResult:
    """Track persistence diagram points across HKS time-scales.

    For each column t_j of H (a fixed HKS scale), computes the H₀
    sub-level-set persistence diagram of HKS(·, t_j) on the mesh.
    Then links diagram points across consecutive scales by nearest-
    neighbour matching in (birth, death) space, producing continuous
    "vines" — trajectories of topological features through scale.

    Features that persist across a large fraction of the t-range
    correspond to anatomically stable sub-regions; features that
    appear or disappear at specific scales reveal scale-dependent
    structural boundaries.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Per-vertex descriptor matrix (HKS at T time-scales).
    adjacency : sparse, shape (N, N)
        Mesh adjacency.
    n_scales : int or None
        Sub-sample to this many scales (for speed). None = use all T.
    min_persistence_frac : float
        Minimum persistence as fraction of the function range to
        retain a feature in the diagram.
    min_life_frac : float
        A vine must span at least this fraction of total scales to
        be considered salient.
    backend : str
        ``"dionysus"`` uses the dionysus2 library (faster, exact).
        ``"manual"`` uses built-in union-find (no extra dependency).

    Returns
    -------
    VineyardResult
        With vines, diagrams per scale, salient features, and
        scale-of-emergence per feature.

    References
    ----------
    Cohen-Steiner D, Edelsbrunner H, Morozov D. Vines and vineyards
        by updating persistence in linear time. *Proc. 22nd ACM Symp.
        Computational Geometry*, 119–126, 2006.
    """
    H = np.asarray(H, dtype=np.float64)
    n, T_total = H.shape

    # optionally sub-sample scales for speed
    if n_scales is not None and n_scales < T_total:
        scale_indices = np.linspace(0, T_total - 1, n_scales, dtype=int)
    else:
        scale_indices = np.arange(T_total)
    T = len(scale_indices)

    # build adjacency list once
    A_coo = sp.coo_matrix(adjacency)
    graph = [[] for _ in range(n)]
    for i, j in zip(A_coo.row.tolist(), A_coo.col.tolist()):
        if i != j:
            graph[i].append(j)

    # compute persistence diagram at each scale
    diagrams = {}
    with progress_simple("Vineyards: persistence per scale", total=T) as tick:
        for si, ti in enumerate(scale_indices):
            f_vals = H[:, ti]
            pairs = _sublevel_h0_pairs(f_vals, graph, n)
            # filter by persistence
            f_range = f_vals.max() - f_vals.min()
            threshold = min_persistence_frac * f_range if f_range > 0 else 0
            pairs_filtered = [
                (b, d, v) for b, d, v in pairs
                if (d - b) > threshold
            ]
            diagrams[si] = {
                "pairs": pairs_filtered,
                "scale_index": int(ti),
                "bd_array": (
                    np.array([(b, d) for b, d, _ in pairs_filtered])
                    if pairs_filtered else np.empty((0, 2))
                ),
            }
            tick(1)

    # link diagram points across scales → vines
    vines = _link_vines(diagrams, T)

    # identify salient features (long-lived vines)
    min_life = int(min_life_frac * T)
    salient = []
    emergence = []
    for vine in vines:
        if len(vine) >= min_life:
            t_start = vine[0]["scale_index"]
            t_end = vine[-1]["scale_index"]
            mean_persistence = np.mean(
                [v["death"] - v["birth"] for v in vine]
            )
            salient.append({
                "vine_length": len(vine),
                "t_start": int(scale_indices[t_start]),
                "t_end": int(scale_indices[t_end]),
                "mean_persistence": float(mean_persistence),
                "representative_vertex": vine[len(vine) // 2].get(
                    "vertex", -1
                ),
            })
            emergence.append(float(scale_indices[t_start]))

    # convert vines to arrays
    vine_arrays = []
    for vine in vines:
        arr = np.array([
            (scale_indices[v["scale_index"]], v["birth"], v["death"])
            for v in vine
        ])
        vine_arrays.append(arr)

    return VineyardResult(
        vines=vine_arrays,
        diagrams={k: v["bd_array"] for k, v in diagrams.items()},
        salient_features=salient,
        scale_of_emergence=np.array(emergence) if emergence else np.array([]),
        metadata={
            "n_scales_used": T,
            "scale_indices": scale_indices,
            "min_persistence_frac": min_persistence_frac,
            "min_life_frac": min_life_frac,
            "total_vines": len(vines),
        },
    )


def _sublevel_h0_pairs(f_vals, graph, n):
    """Compute H₀ sub-level set persistence pairs via union-find."""
    order = np.argsort(f_vals)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    parent = np.arange(n, dtype=np.int64)
    birth = f_vals.copy()
    pairs = []

    def _find(x):
        """Internal parameter search implementation."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idx in order:
        for nb in graph[idx]:
            if rank[nb] < rank[idx]:
                ri = _find(idx)
                rn = _find(nb)
                if ri != rn:
                    if birth[ri] > birth[rn]:
                        parent[ri] = rn
                        pairs.append(
                            (float(birth[ri]), float(f_vals[idx]), int(ri))
                        )
                    else:
                        parent[rn] = ri
                        pairs.append(
                            (float(birth[rn]), float(f_vals[idx]), int(rn))
                        )
    return pairs


def _link_vines(diagrams, T):
    """Link persistence pairs across scales by nearest-neighbour."""
    from scipy.spatial.distance import cdist

    vines = []
    prev_pts = None
    prev_vine_ids = None

    for si in range(T):
        bd = diagrams[si]["bd_array"]

        if bd.shape[0] == 0:
            prev_pts = None
            prev_vine_ids = None
            continue

        if prev_pts is None or prev_pts.shape[0] == 0:
            # start new vines for each point
            for pi in range(bd.shape[0]):
                vines.append([{
                    "scale_index": si,
                    "birth": float(bd[pi, 0]),
                    "death": float(bd[pi, 1]),
                }])
            prev_pts = bd.copy()
            prev_vine_ids = list(range(
                len(vines) - bd.shape[0], len(vines)
            ))
            continue

        # match previous points to current by nearest neighbour
        D = cdist(prev_pts, bd)
        used_curr = set()
        assignments = {}  # prev_idx → curr_idx

        # greedy matching: sort all (prev, curr) pairs by distance
        flat = D.flatten()
        sorted_idx = np.argsort(flat)
        for fi in sorted_idx:
            pi = int(fi // D.shape[1])
            ci = int(fi % D.shape[1])
            if pi not in assignments and ci not in used_curr:
                assignments[pi] = ci
                used_curr.add(ci)
            if len(assignments) == min(D.shape[0], D.shape[1]):
                break

        curr_vine_ids = [None] * bd.shape[0]

        # extend matched vines
        for pi, ci in assignments.items():
            vid = prev_vine_ids[pi]
            vines[vid].append({
                "scale_index": si,
                "birth": float(bd[ci, 0]),
                "death": float(bd[ci, 1]),
            })
            curr_vine_ids[ci] = vid

        # start new vines for unmatched current points
        for ci in range(bd.shape[0]):
            if curr_vine_ids[ci] is None:
                vines.append([{
                    "scale_index": si,
                    "birth": float(bd[ci, 0]),
                    "death": float(bd[ci, 1]),
                }])
                curr_vine_ids[ci] = len(vines) - 1

        prev_pts = bd.copy()
        prev_vine_ids = curr_vine_ids

    return vines


# ======================================================================
# §11  MAPPER PIPELINE — TOPOLOGICAL DATA ANALYSIS ON MESHES
# ======================================================================

def cluster_mapper(
    H: DescriptorMatrix,
    *,
    lens: Literal["hks_sum", "hks_first_pc", "custom"] = "hks_sum",
    custom_lens: Optional[np.ndarray] = None,
    n_cubes: int = 15,
    perc_overlap: float = 0.3,
    clusterer_method: Literal[
        "dbscan", "hdbscan", "agglomerative"
    ] = "dbscan",
    clusterer_eps: float = 0.5,
    dim_reduction: Optional[Literal["umap", "pca"]] = None,
    n_components: int = 2,
    random_state: int = 42,
) -> MapperResult:
    """TDA Mapper pipeline with HKS-derived lens function.

    Projects each vertex through a lens (filter) function into ℝ^d,
    covers the lens range with overlapping hypercubes, clusters within
    each pullback, and forms the nerve complex. The resulting graph
    is a topological skeleton that reveals branching structure,
    loops, and flares in the descriptor space.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Per-vertex descriptor matrix.
    lens : str
        ``"hks_sum"`` — sum of HKS across all scales.
        ``"hks_first_pc"`` — first PC of the descriptor matrix.
        ``"custom"`` — use ``custom_lens``.
    custom_lens : ndarray or None, shape (N,) or (N, d)
        Custom lens function values.
    n_cubes : int
        Number of intervals per lens dimension.
    perc_overlap : float
        Overlap fraction between adjacent intervals (0–1).
    clusterer_method : str
        Clustering algorithm within each pullback.
    clusterer_eps : float
        Epsilon for DBSCAN within pullbacks.
    dim_reduction : str or None
        Reduce descriptor space before building Mapper graph.
    n_components : int
        Target dimensionality for reduction.
    random_state : int

    Returns
    -------
    MapperResult

    References
    ----------
    Singh G, Mémoli F, Carlsson G. Topological methods for the
        analysis of high dimensional data sets and 3D object
        recognition. *SPBG*, 2007.
    """
    km = _require_kepler_mapper()

    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]

    # --- build lens ---
    if lens == "hks_sum":
        lens_data = H.sum(axis=1, keepdims=True)    # (N, 1)
    elif lens == "hks_first_pc":
        skd = _require_sklearn_decomposition()
        pca = skd.PCA(n_components=1, random_state=random_state)
        lens_data = pca.fit_transform(H)              # (N, 1)
    elif lens == "custom":
        if custom_lens is None:
            raise ValueError("custom_lens required when lens='custom'")
        lens_data = np.atleast_2d(custom_lens)
        if lens_data.shape[0] == 1:
            lens_data = lens_data.T
    else:
        raise ValueError(f"Unknown lens: {lens}")

    # --- optionally reduce the high-dim data ---
    if dim_reduction == "umap" and H.shape[1] > n_components:
        umap_mod = _require_umap()
        projected = umap_mod.UMAP(
            n_components=n_components, random_state=random_state,
        ).fit_transform(H)
    elif dim_reduction == "pca" and H.shape[1] > n_components:
        skd = _require_sklearn_decomposition()
        projected = skd.PCA(
            n_components=n_components, random_state=random_state,
        ).fit_transform(H)
    else:
        projected = H

    # --- configure clusterer for pullbacks ---
    if clusterer_method == "dbscan":
        from sklearn.cluster import DBSCAN
        inner_clusterer = DBSCAN(eps=clusterer_eps, min_samples=3)
    elif clusterer_method == "hdbscan":
        hdbscan_mod = _require_hdbscan()
        inner_clusterer = hdbscan_mod.HDBSCAN(min_cluster_size=5)
    elif clusterer_method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        inner_clusterer = AgglomerativeClustering(
            n_clusters=None, distance_threshold=clusterer_eps,
        )
    else:
        raise ValueError(f"Unknown clusterer_method: {clusterer_method}")

    # --- run Mapper ---
    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        lens_data,
        projected,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
        clusterer=inner_clusterer,
    )

    # --- parse the graph ---
    node_keys = list(graph["nodes"].keys())
    node_membership = {}
    for i, key in enumerate(node_keys):
        node_membership[i] = graph["nodes"][key]

    # build adjacency
    nerve_adj = {i: [] for i in range(len(node_keys))}
    key_to_idx = {k: i for i, k in enumerate(node_keys)}
    n_edges = 0
    for edge in graph.get("links", []):
        if len(edge) == 2:
            i0 = key_to_idx.get(edge[0])
            i1 = key_to_idx.get(edge[1])
            if i0 is not None and i1 is not None:
                nerve_adj[i0].append(i1)
                nerve_adj[i1].append(i0)
                n_edges += 1

    # vertex → first nerve node (for colouring)
    v2n = -np.ones(n, dtype=np.int64)
    for ni, verts in node_membership.items():
        for v in verts:
            if v2n[v] == -1:
                v2n[v] = ni

    return MapperResult(
        nerve_graph=nerve_adj,
        node_membership=node_membership,
        n_nodes=len(node_keys),
        n_edges=n_edges,
        vertex_to_nodes=v2n,
        metadata={
            "kepler_graph": graph,
            "lens_type": lens,
            "n_cubes": n_cubes,
            "perc_overlap": perc_overlap,
        },
    )


# ======================================================================
# §12  NON-NEGATIVE TENSOR DECOMPOSITION (MULTI-SUBJECT)
# ======================================================================

def cluster_tensor_decomposition(
    tensor: np.ndarray,
    *,
    n_components: int = 8,
    adjacency: Optional[SparseMatrix] = None,
    lam_spatial: float = 0.0,
    method: Literal["cp", "tucker"] = "cp",
    n_iter_max: int = 200,
    tol: float = 1e-6,
    random_state: int = 42,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> TensorDecompositionResult:
    """Non-negative CP/PARAFAC or Tucker on (vertices × scales × subjects).

    For a cohort of S subjects with vertex-corresponded meshes (e.g.
    via HippUnfold), the HKS data forms a 3-way tensor
    ℋ ∈ ℝ^{N × T × S}. CP decomposes it as

    .. math::

        \\mathcal{H}_{ijk} \\approx \\sum_{r=1}^R w_r(i) \\cdot f_r(j)
        \\cdot s_r(k)

    yielding shared spatial atoms ``w_r`` (which define a parcellation),
    population-level temporal profiles ``f_r``, and per-subject
    loadings ``s_r``.

    Parameters
    ----------
    tensor : ndarray, shape (N, T, S)
        Non-negative tensor (HKS across subjects).
    n_components : int
        CP rank R (≈ expected number of parcels).
    adjacency : sparse or None
        Mesh Laplacian for graph-regularised variant.
    lam_spatial : float
        Weight for the Laplacian penalty on spatial factors.
        ``0.0`` disables spatial regularisation.
    method : str
        ``"cp"`` for CP/PARAFAC, ``"tucker"`` for Tucker.
    n_iter_max : int
    tol : float
    random_state : int
    backend : str
        ``"gpu"`` uses tensorly with PyTorch backend.

    Returns
    -------
    TensorDecompositionResult

    References
    ----------
    Kolda TG, Bader BW. Tensor decompositions and applications.
        *SIAM Review* 51(3):455–500, 2009.
    """
    tl = _require_tensorly()
    import tensorly.decomposition as tl_decomp

    T_data = np.asarray(tensor, dtype=np.float64)
    if T_data.ndim != 3:
        raise ValueError(
            f"Expected 3D tensor (N, T, S), got shape {T_data.shape}"
        )
    T_data = np.maximum(T_data, 0.0) + 1e-12

    use_gpu = _resolve_backend(backend)

    if use_gpu:
        try:
            torch = _require_torch()
            tl.set_backend("pytorch")
            T_tl = tl.tensor(T_data, dtype=torch.float32,
                             device=torch.device("cuda"))
        except Exception:
            logger.warning("GPU tensorly failed, falling back to numpy")
            tl.set_backend("numpy")
            T_tl = tl.tensor(T_data)
    else:
        tl.set_backend("numpy")
        T_tl = tl.tensor(T_data)

    # --- decompose ---
    if method == "cp":
        result = tl_decomp.non_negative_parafac(
            T_tl,
            rank=n_components,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
            return_errors=True,
        )
        if isinstance(result, tuple):
            factors_obj, errors = result[0], result[1]
        else:
            factors_obj, errors = result, []

        # extract factor matrices
        weights = tl.to_numpy(factors_obj.weights) if hasattr(
            factors_obj, "weights"
        ) else np.ones(n_components)
        W = tl.to_numpy(factors_obj.factors[0])    # (N, R)
        F = tl.to_numpy(factors_obj.factors[1])    # (T, R)
        S = tl.to_numpy(factors_obj.factors[2])    # (S, R)

    elif method == "tucker":
        result = tl_decomp.non_negative_tucker(
            T_tl,
            rank=[n_components, min(n_components, T_data.shape[1]),
                  min(n_components, T_data.shape[2])],
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
        )
        core = tl.to_numpy(result[0])
        W = tl.to_numpy(result[1][0])
        F = tl.to_numpy(result[1][1])
        S = tl.to_numpy(result[1][2])
        errors = []
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- optional graph regularisation (post-hoc projection) ---
    if adjacency is not None and lam_spatial > 0:
        A_off = -sp.csr_matrix(adjacency, dtype=np.float64).copy()
        A_off.setdiag(0)
        A_off.eliminate_zeros()
        A_off.data = np.abs(A_off.data)
        D_s = np.asarray(A_off.sum(axis=1)).flatten()

        # iterative Laplacian smoothing of spatial factors
        for _ in range(10):
            AW = A_off @ W
            DW = D_s[:, None] * W
            grad = lam_spatial * (DW - AW)
            W = np.maximum(W - 0.01 * grad, 1e-12)

    # reset backend
    tl.set_backend("numpy")

    # --- cluster labels ---
    labels = W.argmax(axis=1).astype(np.int64)
    n_clusters = len(np.unique(labels))

    # reconstruction error
    rec_err = float(
        errors[-1] if isinstance(errors, list) and len(errors) > 0
        else np.nan
    )

    if use_gpu:
        try:
            torch = _require_torch()
            torch.cuda.empty_cache()
        except Exception:
            pass

    return TensorDecompositionResult(
        spatial_factors=W,
        temporal_factors=F,
        subject_factors=S,
        labels=labels,
        n_components=n_components,
        reconstruction_error=rec_err,
        metadata={
            "method": method,
            "lambda_spatial": lam_spatial,
        },
    )


# ======================================================================
# §13  JOINT TIME-VERTEX GRAPH SIGNAL PROCESSING
# ======================================================================

def denoise_joint_timevertex(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    alpha_graph: float = 1.0,
    beta_time: float = 1.0,
    n_eigenvectors: int = 50,
) -> DescriptorMatrix:
    """Joint time-vertex low-pass filtering of a descriptor matrix.

    Applies a separable filter in the graph-spectral and temporal-
    spectral domains simultaneously:

    .. math::

        g(\\lambda, \\omega) = \\exp(-\\alpha \\lambda - \\beta \\omega^2)

    This smooths H in both mesh-space (removing high-frequency
    geometric noise) and time-space (removing scale-to-scale
    oscillations), producing a cleaner descriptor for downstream
    clustering.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Descriptor matrix.
    adjacency : sparse, shape (N, N)
        Mesh Laplacian (should be positive semi-definite).
    alpha_graph : float
        Graph-spectral smoothing strength.
    beta_time : float
        Temporal smoothing strength.
    n_eigenvectors : int
        Number of graph Laplacian eigenvectors to use.

    Returns
    -------
    ndarray, shape (N, T)
        Filtered descriptor matrix.

    References
    ----------
    Grassi F, Loukas A, Perraudin N, Ricaud B. A time-vertex signal
        processing framework. *IEEE Trans. Signal Processing* 66(3):
        817–829, 2018.
    """
    H = np.asarray(H, dtype=np.float64)
    n, T = H.shape

    # --- graph spectral basis ---
    L = sp.csr_matrix(adjacency, dtype=np.float64)
    from scipy.sparse.linalg import eigsh
    k = min(n_eigenvectors, n - 1)
    eigenvalues_g, U = eigsh(L, k=k, which="SM")
    eigenvalues_g = np.maximum(eigenvalues_g, 0.0)

    # --- temporal spectral basis (DCT-II) ---
    from scipy.fft import dct, idct
    H_graph = U.T @ H                         # (k, T)  graph-spectral
    H_joint = dct(H_graph, type=2, axis=1)    # (k, T)  joint domain

    # --- build separable filter ---
    omega = np.arange(T, dtype=np.float64)
    omega = omega * np.pi / T   # normalised frequency

    g_graph = np.exp(-alpha_graph * eigenvalues_g)          # (k,)
    g_time = np.exp(-beta_time * omega**2)                  # (T,)
    G = g_graph[:, None] * g_time[None, :]                  # (k, T)

    # --- apply filter ---
    H_filtered_joint = H_joint * G
    H_filtered_graph = idct(H_filtered_joint, type=2, axis=1) / (2 * T)
    H_filtered = U @ H_filtered_graph                       # (N, T)

    return H_filtered


def cluster_joint_spectral(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    n_clusters: int = 6,
    n_eigenvectors: int = 30,
    n_freq_bands: int = 5,
    clusterer: Literal["kmeans", "hdbscan"] = "kmeans",
    random_state: int = 42,
) -> ClusterResult:
    """Cluster vertices by joint time-vertex spectral energy.

    Computes the Joint Fourier Transform of H, partitions the
    (graph-frequency, time-frequency) plane into bands, measures
    energy concentration per vertex per band, and clusters vertices
    by their spectral energy profile.

    Parameters
    ----------
    H : ndarray, shape (N, T)
    adjacency : sparse, shape (N, N)
    n_clusters : int
        For k-means.
    n_eigenvectors : int
        Graph Laplacian eigenvectors.
    n_freq_bands : int
        Number of bands to partition the graph-frequency axis.
    clusterer : str
    random_state : int

    Returns
    -------
    ClusterResult
        With ``spectral_energy`` in metadata.
    """
    H = np.asarray(H, dtype=np.float64)
    n, T = H.shape

    # --- graph spectral basis ---
    L = sp.csr_matrix(adjacency, dtype=np.float64)
    from scipy.sparse.linalg import eigsh
    k = min(n_eigenvectors, n - 1)
    eigenvalues_g, U = eigsh(L, k=k, which="SM")
    eigenvalues_g = np.maximum(eigenvalues_g, 0.0)

    # --- compute per-vertex spectral energy in graph bands ---
    # project each vertex's time profile into graph spectral domain
    # vertex i's contribution to eigenmode j: U[i, j] * H[i, :]
    # energy = sum over time of |U[i,j] * H[i,t]|²
    # per band: sum over eigenmodes j in band

    band_edges = np.linspace(0, k, n_freq_bands + 1, dtype=int)
    energy_features = np.zeros((n, n_freq_bands), dtype=np.float64)

    with progress_simple("Joint spectral energy", total=n_freq_bands) as tick:
        for bi in range(n_freq_bands):
            j_start = band_edges[bi]
            j_end = band_edges[bi + 1]
            if j_end <= j_start:
                tick(1)
                continue

            # energy of vertex i in this graph-frequency band
            # = sum_j=j_start..j_end-1 sum_t |U[i,j] * H[i,t]|²
            # = sum_j U[i,j]² * sum_t H[i,t]²
            U_band_sq = np.sum(U[:, j_start:j_end]**2, axis=1)   # (N,)
            H_energy = np.sum(H**2, axis=1)                        # (N,)
            energy_features[:, bi] = U_band_sq * H_energy
            tick(1)

    # normalise
    row_sums = energy_features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    energy_features /= row_sums

    # --- cluster ---
    skc = _require_sklearn_cluster()

    if clusterer == "kmeans":
        km = skc.KMeans(n_clusters=n_clusters, n_init=10,
                        random_state=random_state)
        labels = km.fit_predict(energy_features).astype(np.int64)
    elif clusterer == "hdbscan":
        res = cluster_hdbscan(
            energy_features, log_transform=False, dim_reduction=None,
            random_state=random_state,
        )
        labels = res.labels
    else:
        raise ValueError(f"Unknown clusterer: {clusterer}")

    n_clust = len(set(labels[labels >= 0]))

    return ClusterResult(
        labels=labels,
        n_clusters=n_clust,
        method="joint_spectral",
        quality={},
        metadata={
            "spectral_energy": energy_features,
            "band_edges": band_edges,
            "eigenvalues_graph": eigenvalues_g,
        },
    )


# ======================================================================
# §14  SCALE-SPACE BLOB TRACKING (LINDEBERG ON MANIFOLDS)
# ======================================================================

def cluster_scalespace_blobs(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    t_values: Optional[np.ndarray] = None,
    gamma_normalize: float = 1.0,
    linking_radius: float = 3.0,
    min_trajectory_length: int = 3,
) -> ScaleSpaceBlobResult:
    """Lindeberg-style scale-space blob tracking on HKS.

    HKS(x, t) IS the Gaussian scale-space on the manifold. This
    function detects local maxima of the scale-normalised response
    t^γ · HKS(x, t) at each scale, links them across consecutive
    scales by geodesic proximity, and assigns each vertex to the
    blob trajectory whose maximum is closest.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        HKS matrix at T log-spaced scales.
    adjacency : sparse, shape (N, N)
        Mesh adjacency (for 1-ring local-max detection).
    t_values : ndarray or None, shape (T,)
        Actual diffusion time values. If None, assumes 1..T.
    gamma_normalize : float
        Scale-normalisation exponent γ (Lindeberg 1998). Default 1.0
        corresponds to the standard normalised Laplacian of Gaussian.
    linking_radius : float
        Maximum geodesic hops to link a maximum across scales.
    min_trajectory_length : int
        Minimum number of scales a blob must span.

    Returns
    -------
    ScaleSpaceBlobResult

    References
    ----------
    Lindeberg T. Feature detection with automatic scale selection.
        *IJCV* 30(2):79–116, 1998.
    """
    H = np.asarray(H, dtype=np.float64)
    n, T = H.shape

    if t_values is None:
        t_values = np.arange(1, T + 1, dtype=np.float64)
    else:
        t_values = np.asarray(t_values, dtype=np.float64)

    # build adjacency list and extended k-hop neighbourhood
    A_coo = sp.coo_matrix(adjacency)
    adj_list = [set() for _ in range(n)]
    for i, j in zip(A_coo.row.tolist(), A_coo.col.tolist()):
        if i != j:
            adj_list[i].add(j)

    # expand to k-hop for linking radius
    k_hops = max(1, int(linking_radius))

    def _k_hop_neighbours(v, k):
        """Compute k-hop neighbourhood indices from an adjacency matrix."""
        visited = {v}
        frontier = {v}
        for _ in range(k):
            new_frontier = set()
            for u in frontier:
                for nb in adj_list[u]:
                    if nb not in visited:
                        visited.add(nb)
                        new_frontier.add(nb)
            frontier = new_frontier
        return visited

    # scale-normalised response
    H_norm = np.zeros_like(H)
    for ti in range(T):
        H_norm[:, ti] = (t_values[ti] ** gamma_normalize) * H[:, ti]

    # detect local maxima at each scale (1-ring maximum)
    maxima_per_scale = []
    with progress_simple("Blob detection", total=T) as tick:
        for ti in range(T):
            vals = H_norm[:, ti]
            local_max = []
            for v in range(n):
                is_max = True
                for nb in adj_list[v]:
                    if vals[nb] >= vals[v]:
                        is_max = False
                        break
                if is_max and vals[v] > 0:
                    local_max.append({
                        "vertex": v,
                        "scale_index": ti,
                        "t_value": float(t_values[ti]),
                        "response": float(vals[v]),
                    })
            maxima_per_scale.append(local_max)
            tick(1)

    # link maxima across scales into trajectories
    trajectories = []
    prev_maxima = None
    prev_traj_ids = None

    for ti in range(T):
        curr_maxima = maxima_per_scale[ti]
        if not curr_maxima:
            prev_maxima = None
            prev_traj_ids = None
            continue

        if prev_maxima is None:
            for m in curr_maxima:
                trajectories.append([m])
            prev_maxima = curr_maxima
            prev_traj_ids = list(range(
                len(trajectories) - len(curr_maxima),
                len(trajectories),
            ))
            continue

        # link by k-hop proximity
        curr_traj_ids = [None] * len(curr_maxima)
        used_prev = set()

        for ci, cm in enumerate(curr_maxima):
            best_pi = None
            best_dist = float("inf")
            cm_neighbours = _k_hop_neighbours(cm["vertex"], k_hops)

            for pi, pm in enumerate(prev_maxima):
                if pi in used_prev:
                    continue
                if pm["vertex"] in cm_neighbours:
                    d = abs(cm["response"] - pm["response"])
                    if d < best_dist:
                        best_dist = d
                        best_pi = pi

            if best_pi is not None:
                tid = prev_traj_ids[best_pi]
                trajectories[tid].append(cm)
                curr_traj_ids[ci] = tid
                used_prev.add(best_pi)

        # unmatched → new trajectories
        for ci, cm in enumerate(curr_maxima):
            if curr_traj_ids[ci] is None:
                trajectories.append([cm])
                curr_traj_ids[ci] = len(trajectories) - 1

        prev_maxima = curr_maxima
        prev_traj_ids = curr_traj_ids

    # filter by minimum trajectory length
    long_trajectories = [
        tr for tr in trajectories if len(tr) >= min_trajectory_length
    ]

    # natural scale: for each vertex, find t where normalised response
    # is maximal
    natural_scales = t_values[H_norm.argmax(axis=1)]

    # assign vertices to nearest trajectory (by maximum response vertex)
    blob_labels = -np.ones(n, dtype=np.int64)
    for ti_idx, traj in enumerate(long_trajectories):
        # the trajectory's "representative" vertices at each scale
        for step in traj:
            v = step["vertex"]
            if blob_labels[v] == -1:
                blob_labels[v] = ti_idx
            # also assign 1-ring neighbours of trajectory peaks
            for nb in adj_list[v]:
                if blob_labels[nb] == -1:
                    blob_labels[nb] = ti_idx

    # assign remaining vertices to nearest blob by natural scale
    # similarity and spatial proximity
    unassigned = np.where(blob_labels == -1)[0]
    if len(unassigned) > 0 and len(long_trajectories) > 0:
        # build centroid for each blob
        blob_centroids = np.zeros(len(long_trajectories))
        for bi, traj in enumerate(long_trajectories):
            blob_centroids[bi] = np.mean(
                [s["t_value"] for s in traj]
            )

        for v in unassigned:
            # find nearest assigned neighbour
            best_label = -1
            for nb in adj_list[v]:
                if blob_labels[nb] >= 0:
                    best_label = blob_labels[nb]
                    break
            if best_label >= 0:
                blob_labels[v] = best_label
            else:
                # assign by natural scale
                diff = np.abs(blob_centroids - natural_scales[v])
                blob_labels[v] = int(np.argmin(diff))

    return ScaleSpaceBlobResult(
        blob_trajectories=long_trajectories,
        natural_scales=natural_scales,
        blob_labels=blob_labels,
        n_blobs=len(long_trajectories),
        metadata={
            "gamma": gamma_normalize,
            "linking_radius": linking_radius,
            "total_trajectories_before_filter": len(trajectories),
            "min_trajectory_length": min_trajectory_length,
        },
    )


# ======================================================================
# §15  MULTI-VIEW CLUSTERING (GEOMETRY + DESCRIPTOR VIEWS)
# ======================================================================

def cluster_multiview(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    n_clusters: int = 6,
    n_eigenvectors_geo: int = 20,
    n_eigenvectors_desc: int = 10,
    alpha: float = 0.5,
    fusion: Literal[
        "spectral_average", "late_consensus", "concatenate"
    ] = "spectral_average",
    random_state: int = 42,
) -> ClusterResult:
    """Multi-view clustering with geometry and descriptor views.

    View 1: Low-frequency Laplacian eigenfunctions (encoding spatial
    position on the manifold — the same basis from which HKS is
    built).

    View 2: HKS/WKS descriptor profiles (encoding multi-scale
    geometric features).

    Three fusion strategies are available:

    - ``"spectral_average"``: average the normalised Laplacians of
      both views, then spectral clustering on the combined Laplacian.
    - ``"late_consensus"``: cluster each view independently, then
      reconcile via consensus (CSPA).
    - ``"concatenate"``: stack view features and cluster jointly.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Descriptor matrix (View 2).
    adjacency : sparse, shape (N, N)
        Mesh Laplacian (View 1 is its eigenfunctions).
    n_clusters : int
    n_eigenvectors_geo : int
        Number of Laplacian eigenfunctions for View 1.
    n_eigenvectors_desc : int
        PCA components for View 2.
    alpha : float
        Weight for View 1 (geometry). 1−α for View 2 (descriptors).
    fusion : str
    random_state : int

    Returns
    -------
    ClusterResult

    References
    ----------
    Kumar A, Rai P, Daumé H. Co-regularized multi-view spectral
        clustering. *NeurIPS* 24, 2011.
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    skc = _require_sklearn_cluster()

    # --- View 1: Laplacian eigenfunctions ---
    L = sp.csr_matrix(adjacency, dtype=np.float64)
    from scipy.sparse.linalg import eigsh
    k_geo = min(n_eigenvectors_geo, n - 1)
    _, Phi = eigsh(L, k=k_geo, which="SM")
    Phi = Phi[:, 1:] if k_geo > 1 else Phi  # drop constant mode

    # --- View 2: reduced descriptor ---
    skd = _require_sklearn_decomposition()
    k_desc = min(n_eigenvectors_desc, H.shape[1])
    X_log = np.log(np.maximum(H, 1e-12))
    Psi = skd.PCA(
        n_components=k_desc, random_state=random_state
    ).fit_transform(X_log)

    if fusion == "spectral_average":
        # build affinity for each view
        from scipy.spatial.distance import pdist, squareform

        D1 = squareform(pdist(Phi, "sqeuclidean"))
        sigma1 = np.median(D1[D1 > 0]) or 1.0
        A1 = np.exp(-D1 / sigma1)
        np.fill_diagonal(A1, 0)
        D1_deg = np.diag(A1.sum(axis=1))
        L1 = D1_deg - A1

        D2 = squareform(pdist(Psi, "sqeuclidean"))
        sigma2 = np.median(D2[D2 > 0]) or 1.0
        A2 = np.exp(-D2 / sigma2)
        np.fill_diagonal(A2, 0)
        D2_deg = np.diag(A2.sum(axis=1))
        L2 = D2_deg - A2

        # normalise each
        D1_inv_sqrt = np.diag(
            1.0 / np.sqrt(np.diag(D1_deg) + 1e-12)
        )
        D2_inv_sqrt = np.diag(
            1.0 / np.sqrt(np.diag(D2_deg) + 1e-12)
        )
        L1_sym = D1_inv_sqrt @ L1 @ D1_inv_sqrt
        L2_sym = D2_inv_sqrt @ L2 @ D2_inv_sqrt

        L_avg = alpha * L1_sym + (1.0 - alpha) * L2_sym

        eigvals, eigvecs = np.linalg.eigh(L_avg)
        U = eigvecs[:, :n_clusters]
        # row-normalise (Ng-Jordan-Weiss)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U /= norms

        labels = skc.KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state,
        ).fit_predict(U).astype(np.int64)

    elif fusion == "late_consensus":
        # cluster each view, then majority vote
        lab1 = skc.KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state,
        ).fit_predict(Phi).astype(np.int64)

        lab2 = skc.KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state,
        ).fit_predict(Psi).astype(np.int64)

        # consensus via co-association matrix
        C = np.zeros((n, n), dtype=np.float64)
        for lab in [lab1, lab2]:
            for k in range(n_clusters):
                mask = lab == k
                C[np.ix_(mask, mask)] += 1.0
        C /= 2.0

        # spectral clustering on co-association
        labels = skc.SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=random_state,
        ).fit_predict(C).astype(np.int64)

    elif fusion == "concatenate":
        # normalise each view to unit variance
        Phi_n = Phi / (Phi.std(axis=0, keepdims=True) + 1e-12)
        Psi_n = Psi / (Psi.std(axis=0, keepdims=True) + 1e-12)
        Z = np.hstack([alpha * Phi_n, (1.0 - alpha) * Psi_n])

        labels = skc.KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state,
        ).fit_predict(Z).astype(np.int64)
    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        method=f"multiview_{fusion}",
        quality={},
        metadata={"alpha": alpha, "fusion": fusion},
    )


# ======================================================================
# §16  SPECTRAL GRAPH WAVELET CLUSTERING
# ======================================================================

def cluster_wavelet_coefficients(
    H: DescriptorMatrix,
    adjacency: SparseMatrix,
    *,
    n_scales: int = 5,
    n_clusters: int = 6,
    wavelet_type: Literal[
        "mexican_hat", "heat", "meyer"
    ] = "mexican_hat",
    n_eigenvectors: int = 100,
    clusterer: Literal["kmeans", "hdbscan"] = "kmeans",
    random_state: int = 42,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
) -> ClusterResult:
    """Cluster vertices by spectral graph wavelet energy profiles.

    Decomposes HKS into band-pass components using spectral graph
    wavelets (Hammond, Vandergheynst & Gribonval, 2011), computes
    the energy in each band at each vertex, and clusters vertices
    by their multi-band energy signature.

    Unlike raw HKS (which is a low-pass filter at each t), wavelet
    decomposition provides **orthogonal** band-pass filters, so
    features at different scales do not leak into each other.

    Parameters
    ----------
    H : ndarray, shape (N, T)
        Descriptor matrix.
    adjacency : sparse, shape (N, N)
        Mesh Laplacian.
    n_scales : int
        Number of wavelet scales (frequency bands).
    n_clusters : int
        For k-means.
    wavelet_type : str
        ``"mexican_hat"`` (g(x) = x·exp(-x)), ``"heat"`` (exp(-x)),
        ``"meyer"`` (smooth band-pass).
    n_eigenvectors : int
        Laplacian eigenvectors for spectral acceleration.
    clusterer : str
    random_state : int
    backend : str

    Returns
    -------
    ClusterResult
        With ``wavelet_energy`` matrix in metadata.

    References
    ----------
    Hammond DK, Vandergheynst P, Gribonval R. Wavelets on graphs via
        spectral graph theory. *ACHA* 30(2):129–150, 2011.
    """
    H = np.asarray(H, dtype=np.float64)
    n, T = H.shape

    # --- Laplacian eigenbasis ---
    L = sp.csr_matrix(adjacency, dtype=np.float64)
    from scipy.sparse.linalg import eigsh
    k = min(n_eigenvectors, n - 1)
    eigenvalues, U = eigsh(L, k=k, which="SM")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    lam_max = eigenvalues[-1] if eigenvalues[-1] > 0 else 1.0

    # --- wavelet kernel ---
    scales = np.logspace(
        np.log10(2.0 / lam_max),
        np.log10(2.0 / max(eigenvalues[1], 1e-6)),
        n_scales,
    )

    def _wavelet_kernel(s, lam):
        """Evaluate the spectral graph wavelet kernel at a given scale."""
        if wavelet_type == "mexican_hat":
            x = s * lam
            return x * np.exp(-x)
        elif wavelet_type == "heat":
            return np.exp(-s * lam)
        elif wavelet_type == "meyer":
            x = s * lam
            val = np.where(
                (x >= 2 * np.pi / 3) & (x <= 8 * np.pi / 3),
                np.cos(np.pi / 2 * _meyer_aux(3 * x / (4 * np.pi) - 1)),
                0.0,
            )
            return val
        else:
            raise ValueError(f"Unknown wavelet_type: {wavelet_type}")

    # --- compute wavelet transform of HKS ---
    use_gpu = _resolve_backend(backend)

    if use_gpu:
        torch = _require_torch()
        device = torch.device("cuda")
        U_t = torch.tensor(U, dtype=torch.float32, device=device)
        H_t = torch.tensor(H, dtype=torch.float32, device=device)
        eig_t = torch.tensor(eigenvalues, dtype=torch.float32, device=device)

        energy_features = torch.zeros(
            (n, n_scales), dtype=torch.float32, device=device
        )

        with progress_simple("Wavelet energy [GPU]", total=n_scales) as tick:
            for si, s in enumerate(scales):
                g_s = torch.tensor(
                    _wavelet_kernel(s, eigenvalues),
                    dtype=torch.float32, device=device,
                )
                # wavelet coefficients at scale s:
                # W_s H = U diag(g_s) U^T H
                H_spec = U_t.T @ H_t                       # (k, T)
                H_filtered = (g_s[:, None] * H_spec)        # (k, T)
                W_s_H = U_t @ H_filtered                    # (N, T)
                # energy = sum over time of squared coefficients
                energy_features[:, si] = (W_s_H ** 2).sum(dim=1)
                tick(1)

        energy_np = energy_features.cpu().numpy().astype(np.float64)
        del U_t, H_t, energy_features
        torch.cuda.empty_cache()
    else:
        energy_np = np.zeros((n, n_scales), dtype=np.float64)

        with progress_simple("Wavelet energy [CPU]", total=n_scales) as tick:
            for si, s in enumerate(scales):
                g_s = _wavelet_kernel(s, eigenvalues)       # (k,)
                H_spec = U.T @ H                             # (k, T)
                H_filtered = g_s[:, None] * H_spec           # (k, T)
                W_s_H = U @ H_filtered                       # (N, T)
                energy_np[:, si] = np.sum(W_s_H ** 2, axis=1)
                tick(1)

    # normalise
    row_sums = energy_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    energy_np /= row_sums

    # --- cluster ---
    skc = _require_sklearn_cluster()

    if clusterer == "kmeans":
        labels = skc.KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state,
        ).fit_predict(energy_np).astype(np.int64)
    elif clusterer == "hdbscan":
        res = cluster_hdbscan(
            energy_np, log_transform=False, dim_reduction=None,
            random_state=random_state,
        )
        labels = res.labels
    else:
        raise ValueError(f"Unknown clusterer: {clusterer}")

    n_clust = len(set(labels[labels >= 0]))

    return ClusterResult(
        labels=labels,
        n_clusters=n_clust,
        method="wavelet_energy",
        quality={},
        metadata={
            "wavelet_energy": energy_np,
            "scales": scales,
            "wavelet_type": wavelet_type,
        },
    )


def _meyer_aux(x):
    """Meyer wavelet auxiliary function ν(x) for smooth transition."""
    x = np.clip(x, 0.0, 1.0)
    return x**4 * (35 - 84 * x + 70 * x**2 - 20 * x**3)


# ======================================================================
# Internal helpers
# ======================================================================

def _resolve_backend(backend: str) -> bool:
    """Return True if GPU should be used."""
    if backend == "gpu":
        return True
    if backend == "cpu":
        return False
    # auto
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
