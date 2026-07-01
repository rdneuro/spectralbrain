"""Distance-dependent Chinese Restaurant Process (ddCRP) clustering.

SpectralBrain-harmonised entry points around the vendored ddCRP cores. The
sampler restricts candidate links to mesh neighbours (enforcing spatial
contiguity), scores partitions with the Normal-Inverse-Wishart collapsed
marginal likelihood (Murphy 2007/2012), and modulates contiguity by real edge
distance when mesh ``vertices`` are supplied.

These functions complement the existing :mod:`spectralbrain.statistics.clustering`
algorithms and return the same :class:`~spectralbrain.statistics.clustering.ClusterResult`
container, so they slot into the clustering API uniformly.

- :func:`cluster_ddcrp` -- spatial ddCRP on a descriptor matrix.
- :func:`cluster_ddcrp_functional` -- ddCRP on fPCA-compressed HKS/WKS curves.
- :func:`cluster_consensus` -- consensus partition from a clustering ensemble.
- :func:`autotune_ddcrp` -- data-driven hyperparameter search (anti-degeneracy
  guarded) with optional refit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from spectralbrain.statistics.clustering import ClusterResult

__all__ = [
    "cluster_ddcrp",
    "cluster_ddcrp_functional",
    "cluster_consensus",
    "autotune_ddcrp",
    "DDCRPTuningResult",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _resolve_adjacency_list(n_vertices: int, faces, adjacency):
    """Build a per-vertex neighbour list from faces or a sparse adjacency."""
    from spectralbrain.statistics._clustercore import (
        adjacency_list_from_faces, adjacency_list_from_sparse,
    )
    if faces is not None:
        return adjacency_list_from_faces(np.asarray(faces), n_vertices)
    if adjacency is not None:
        return adjacency_list_from_sparse(adjacency)
    raise ValueError(
        "ddCRP requires mesh contiguity: pass `faces` (preferred) or a sparse "
        "`adjacency` matrix so candidate links are restricted to mesh neighbours.")


def _default_decay_scale(decay_scale, vertices, adjacency_list):
    """Median edge length as the natural decay scale when vertices are given."""
    if decay_scale is not None:
        return float(decay_scale)
    if vertices is not None:
        from spectralbrain.statistics._clustercore import edge_distances
        d = edge_distances(np.asarray(vertices, float), adjacency_list)
        nonempty = [x for x in d if np.size(x)]
        if nonempty:
            return float(np.median(np.concatenate(nonempty)))
    return 1.0


def _to_cluster_result(r, method: str, extra_meta: dict) -> ClusterResult:
    labels = np.asarray(r.labels, dtype=np.int64)
    n_clusters = int(np.unique(labels[labels >= 0]).size)
    meta = {
        "co_association": r.co_association,
        "n_clusters_trace": r.n_clusters_trace,
        "chains": r.chains,
    }
    meta.update(extra_meta)
    return ClusterResult(
        labels=labels, n_clusters=n_clusters, method=method,
        probabilities=None,
        quality={"rhat_n_clusters": float(r.rhat_n_clusters)},
        metadata=meta,
    )


# ----------------------------------------------------------------------
# spatial ddCRP
# ----------------------------------------------------------------------
def cluster_ddcrp(
    H: np.ndarray,
    *,
    faces: Optional[np.ndarray] = None,
    adjacency: Any = None,
    vertices: Optional[np.ndarray] = None,
    decay_kind: str = "exponential",
    decay_scale: Optional[float] = None,
    alpha: float = 1.0,
    prior: Any = None,
    n_components: Optional[int] = None,
    n_draws: int = 200,
    burn_in: int = 100,
    thin: int = 2,
    chains: int = 4,
    random_state: int = 42,
    progress: bool = True,
) -> ClusterResult:
    """Contiguous ddCRP clustering with a NIW collapsed marginal likelihood.

    Parameters
    ----------
    H : ndarray, shape (V, d)
        Per-vertex descriptor / feature matrix (e.g. fused HKS/WKS scores).
    faces : ndarray, shape (F, 3), optional
        Mesh triangles; used to build the neighbour list and (with ``vertices``)
        the edge distances. Provide this or ``adjacency``.
    adjacency : sparse matrix, optional
        Pre-built vertex adjacency (alternative to ``faces``).
    vertices : ndarray, shape (V, 3), optional
        Mesh coordinates; if given, the decay kernel uses real Euclidean edge
        distances and ``decay_scale`` defaults to the median edge length.
    decay_kind : {"window", "exponential", "logistic"}
        Distance-decay form of the ddCRP link prior.
    decay_scale : float, optional
        Decay length scale (mesh units). Defaults to the median edge length when
        ``vertices`` is given, else 1.0.
    alpha : float
        ddCRP self-link concentration (higher -> more, smaller clusters).
    prior : NIWPrior, optional
        Custom NIW prior; by default fit from the data.
    n_components : int, optional
        If set, PCA-whiten ``H`` to this many components before sampling. HKS/WKS
        descriptors are highly redundant across scales, so a small value (e.g. 6-10)
        loses little information while making each marginal-likelihood evaluation
        dramatically cheaper (the cost is cubic in the feature dimension). Strongly
        recommended for dense surfaces; ``None`` keeps the full feature space.
    n_draws, burn_in, thin, chains : int
        Collapsed-Gibbs sampling budget.
    random_state : int
    progress : bool

    Returns
    -------
    ClusterResult
        ``metadata`` carries the posterior ``co_association`` matrix,
        ``n_clusters_trace`` and the decay settings.
    """
    from spectralbrain.statistics._clustercore import cluster_ddcrp as _ddcrp

    H = np.asarray(H, dtype=np.float64)
    if n_components is not None and H.shape[1] > n_components:
        from sklearn.decomposition import PCA
        H = PCA(n_components=int(n_components), whiten=True,
                random_state=random_state).fit_transform(H)
    adj = _resolve_adjacency_list(H.shape[0], faces, adjacency)
    scale = _default_decay_scale(decay_scale, vertices, adj)
    r = _ddcrp(H, adj, decay_kind=decay_kind, decay_scale=scale, alpha=alpha,
               prior=prior, n_draws=n_draws, burn_in=burn_in, thin=thin,
               chains=chains, random_state=random_state, vertices=vertices,
               progress=progress)
    return _to_cluster_result(r, "ddcrp",
                              {"decay_kind": decay_kind, "decay_scale": scale,
                               "alpha": alpha, "n_components": n_components})


# ----------------------------------------------------------------------
# functional ddCRP (fPCA of curve blocks)
# ----------------------------------------------------------------------
def cluster_ddcrp_functional(
    curves_blocks: Sequence[np.ndarray],
    *,
    faces: Optional[np.ndarray] = None,
    adjacency: Any = None,
    vertices: Optional[np.ndarray] = None,
    n_fpca: int = 5,
    decay_kind: str = "exponential",
    decay_scale: Optional[float] = None,
    alpha: float = 1.0,
    prior: Any = None,
    n_draws: int = 200,
    burn_in: int = 100,
    thin: int = 2,
    chains: int = 4,
    random_state: int = 42,
    progress: bool = True,
) -> ClusterResult:
    """ddCRP on fPCA-compressed curve blocks (e.g. HKS and WKS together).

    Each block is fPCA-compressed independently and the scores are concatenated
    with MFA-style first-singular-value balancing, then clustered by the spatial
    NIW ddCRP. Blocks must share the vertex axis.

    Parameters
    ----------
    curves_blocks : sequence of (V, T_b) arrays
        Curve stacks to fuse (e.g. ``[hks_VxT, wks_VxE]``).
    faces, adjacency, vertices, decay_kind, decay_scale, alpha, prior,
    n_draws, burn_in, thin, chains, random_state, progress
        As in :func:`cluster_ddcrp`.
    n_fpca : int
        fPCA components kept per block.

    Returns
    -------
    ClusterResult
    """
    from spectralbrain.statistics._clustercore import (
        cluster_ddcrp_functional as _ddcrp_func,
    )

    blocks = [np.asarray(b, dtype=np.float64) for b in curves_blocks]
    n_vertices = blocks[0].shape[0]
    adj = _resolve_adjacency_list(n_vertices, faces, adjacency)
    scale = _default_decay_scale(decay_scale, vertices, adj)
    r, info = _ddcrp_func(blocks, adj, n_fpca=n_fpca, decay_kind=decay_kind,
                          decay_scale=scale, alpha=alpha, prior=prior,
                          n_draws=n_draws, burn_in=burn_in, thin=thin,
                          chains=chains, random_state=random_state,
                          vertices=vertices, progress=progress)
    meta = {"decay_kind": decay_kind, "decay_scale": scale, "alpha": alpha,
            "n_fpca": n_fpca, "fpca_info": info}
    return _to_cluster_result(r, "ddcrp_functional", meta)


# ----------------------------------------------------------------------
# consensus clustering
# ----------------------------------------------------------------------
def cluster_consensus(
    partitions: Sequence[np.ndarray],
    *,
    n_clusters: Optional[int] = None,
    threshold: Optional[float] = None,
) -> ClusterResult:
    """Consensus partition from an ensemble of labelings (co-association + AC).

    Builds the co-association matrix (fraction of partitions in which two
    vertices co-cluster) and extracts a consensus partition either by cutting an
    average-linkage agglomeration to ``n_clusters`` or by thresholding the
    co-association graph and taking connected components.

    Parameters
    ----------
    partitions : sequence of (V,) label arrays
        Ensemble of partitions of the same vertices.
    n_clusters : int, optional
        Cut the dendrogram to this many clusters.
    threshold : float, optional
        Co-association threshold for the connected-components route. If neither
        ``n_clusters`` nor ``threshold`` is given, ``threshold=0.5`` is used.

    Returns
    -------
    ClusterResult
        ``metadata`` holds the ``co_association`` matrix and per-vertex
        ``stability`` (mean within-cluster co-association).
    """
    from spectralbrain.statistics._clustercore import (
        co_association_matrix, consensus_partition, stability_per_vertex,
    )

    parts = [np.asarray(p, dtype=np.int64) for p in partitions]
    co = co_association_matrix(parts)
    if n_clusters is None and threshold is None:
        threshold = 0.5
    labels = np.asarray(consensus_partition(co, n_clusters=n_clusters,
                                            threshold=threshold), dtype=np.int64)
    stab = stability_per_vertex(co, labels)
    n = int(np.unique(labels[labels >= 0]).size)
    return ClusterResult(
        labels=labels, n_clusters=n, method="consensus",
        probabilities=None,
        quality={"mean_stability": float(np.nanmean(stab))},
        metadata={"co_association": co, "stability": stab,
                  "n_partitions": len(parts)},
    )


# ----------------------------------------------------------------------
# autotuning
# ----------------------------------------------------------------------
@dataclass
class DDCRPTuningResult:
    """Result of :func:`autotune_ddcrp`.

    Attributes
    ----------
    best_params : dict
    best_score : float
    history : list of dict
    backend : str
    cluster_result : ClusterResult or None
        The refit partition at full sampling budget (if ``refit=True``).
    """

    best_params: dict
    best_score: float
    history: list
    backend: str
    cluster_result: Optional[ClusterResult] = None
    metadata: dict = field(default_factory=dict)


def autotune_ddcrp(
    H: np.ndarray,
    *,
    faces: Optional[np.ndarray] = None,
    adjacency: Any = None,
    vertices: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
    n_trials: int = 40,
    objective: str = "silhouette",
    spatial_distance: Optional[np.ndarray] = None,
    eval_draws: int = 30,
    eval_burn_in: int = 20,
    eval_chains: int = 1,
    random_state: int = 42,
    progress: bool = True,
    refit: bool = True,
    refit_kwargs: Optional[dict] = None,
) -> DDCRPTuningResult:
    """Optimise ddCRP hyperparameters from the data (avoids the K=1 collapse).

    Searches ``psi_scale``/``kappa0``/``alpha``/``decay`` with an anti-degeneracy
    guard so partitions that collapse to one cluster or shatter into singletons
    are rejected. When ``vertices`` are given, the ``decay_scale`` search bounds
    are scaled to the mesh's median edge length.

    Parameters
    ----------
    H : ndarray, shape (V, d)
    faces, adjacency, vertices
        Mesh contiguity (see :func:`cluster_ddcrp`).
    backend : {"optuna", "hyperopt", "botorch", "random", None}
        Optimiser; falls back to reproducible random search if unavailable.
    n_trials, objective, spatial_distance, eval_draws, eval_burn_in,
    eval_chains, random_state, progress, refit, refit_kwargs
        Forwarded to the tuner.

    Returns
    -------
    DDCRPTuningResult
    """
    from spectralbrain.statistics._clustercore import autotune_ddcrp as _autotune

    H = np.asarray(H, dtype=np.float64)
    adj = _resolve_adjacency_list(H.shape[0], faces, adjacency)
    tr = _autotune(H, adj, backend=backend, n_trials=n_trials, objective=objective,
                   spatial_distance=spatial_distance, vertices=vertices,
                   eval_draws=eval_draws, eval_burn_in=eval_burn_in,
                   eval_chains=eval_chains, random_state=random_state,
                   progress=progress, refit=refit, refit_kwargs=refit_kwargs)

    cluster_result = None
    if getattr(tr, "refit_result", None) is not None:
        cluster_result = _to_cluster_result(
            tr.refit_result, "ddcrp_autotuned",
            {"best_params": tr.best_params, "objective": objective})
    return DDCRPTuningResult(
        best_params=tr.best_params, best_score=float(tr.best_score),
        history=tr.history, backend=tr.backend, cluster_result=cluster_result,
        metadata={"objective": objective, "n_trials": n_trials},
    )
