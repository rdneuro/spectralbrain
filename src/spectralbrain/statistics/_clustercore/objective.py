"""Objective/scoring utilities for ddCRP hyperparameter optimisation.

The optimiser maximises a partition-quality score computed from the labels a
candidate hyperparameter set produces. All objectives include an
**anti-degeneracy guard**: partitions that collapse to a single cluster (the
classic ddCRP failure) or shatter into singletons receive a large negative
score, so the search is steered away from them.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("spectralbrain.statistics._clustercore")

# Score returned for degenerate / invalid partitions (minimised away).
DEGENERATE_SCORE = -1.0e6

VALID_OBJECTIVES = ("silhouette", "intra_inter", "calinski_harabasz",
                    "stability", "evidence")


def _n_clusters(labels: np.ndarray) -> int:
    valid = labels[labels >= 0]
    return int(np.unique(valid).size)


def score_partition(features: np.ndarray, labels: np.ndarray,
                    objective: str = "silhouette",
                    spatial_distance: Optional[np.ndarray] = None,
                    co_association: Optional[np.ndarray] = None,
                    min_clusters: int = 2,
                    max_cluster_fraction: float = 0.95) -> float:
    """Score a partition for hyperparameter optimisation (higher is better).

    Parameters
    ----------
    features : np.ndarray, shape (V, d)
        Features the partition was computed on.
    labels : np.ndarray, shape (V,)
        Candidate partition (``-1`` allowed as noise).
    objective : {"silhouette", "intra_inter", "calinski_harabasz",
                 "stability", "evidence"}
        Quality criterion. ``"stability"`` needs ``co_association``;
        ``"evidence"`` expects ``features`` to already include the per-cluster
        NIW evidence in ``co_association`` is not used.
    spatial_distance : np.ndarray, optional
        Geodesic/graph distance, used by the spatial silhouette.
    co_association : np.ndarray, optional
        Posterior co-association matrix (for ``objective="stability"``).
    min_clusters : int
        Minimum acceptable number of clusters.
    max_cluster_fraction : float
        Reject partitions whose largest cluster exceeds this fraction of all
        vertices (near-collapse), and reject all-singleton partitions.

    Returns
    -------
    float
        Quality score, or :data:`DEGENERATE_SCORE` for degenerate partitions.
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    k = _n_clusters(labels)

    # --- anti-degeneracy guard ---
    if k < min_clusters:
        return DEGENERATE_SCORE
    valid = labels[labels >= 0]
    _, counts = np.unique(valid, return_counts=True)
    if counts.max() > max_cluster_fraction * n:
        return DEGENERATE_SCORE
    if np.all(counts == 1):
        return DEGENERATE_SCORE

    X = np.asarray(features, float)

    if objective == "stability":
        if co_association is None:
            raise ValueError("objective='stability' requires co_association.")
        from .consensus import stability_per_vertex
        return float(np.mean(stability_per_vertex(co_association, labels)))

    if objective == "silhouette":
        from .cluster_stats import spatial_silhouette
        s = spatial_silhouette(X, labels, spatial_distance=spatial_distance)
        return float(s) if np.isfinite(s) else DEGENERATE_SCORE

    if objective == "intra_inter":
        from .cluster_stats import intra_inter_homogeneity
        r = intra_inter_homogeneity(X, labels)["ratio"]
        return float(r) if np.isfinite(r) else DEGENERATE_SCORE

    if objective == "calinski_harabasz":
        from sklearn.metrics import calinski_harabasz_score
        mask = labels >= 0
        if _n_clusters(labels[mask]) < 2:
            return DEGENERATE_SCORE
        return float(calinski_harabasz_score(X[mask], labels[mask]))

    if objective == "evidence":
        # Empirical-Bayes style: total NIW log marginal likelihood of the
        # partition under a weak prior (model evidence proxy). Higher = better
        # fit; the NIW Occam penalty keeps K finite.
        from .ddcrp import NIWPrior, niw_log_marginal
        prior = NIWPrior.from_data(X)
        total = 0.0
        for c in np.unique(valid):
            idx = np.where(labels == c)[0]
            sub = X[idx]
            total += niw_log_marginal(idx.size, sub.sum(0), sub.T @ sub, prior)
        return float(total)

    raise ValueError(f"Unknown objective {objective!r}; choose {VALID_OBJECTIVES}.")
