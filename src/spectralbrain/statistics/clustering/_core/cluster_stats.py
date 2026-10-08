"""Spatially-honest cluster quality and structure statistics.

Provides intra/inter-cluster homogeneity and separation measures, a
spatially-aware silhouette, energy distance between clusters, and
representational-similarity (RSA) / Mantel comparisons. Null distributions for
these should be built with the SA-preserving surrogates in
:mod:`brainmosaic.statistics.nulls` rather than naive permutation, which inherits
the spatial-autocorrelation bias.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("brainmosaic.statistics.cluster_stats")


def intra_inter_homogeneity(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Within- vs between-cluster feature homogeneity.

    Returns
    -------
    dict
        ``intra`` (mean within-cluster pairwise similarity),
        ``inter`` (mean between-cluster similarity),
        ``ratio`` (intra / inter; higher is better-separated).
    """
    from sklearn.metrics import pairwise_distances

    X = np.asarray(features, float)
    labels = np.asarray(labels)
    D = pairwise_distances(X)
    sim = 1.0 / (1.0 + D)
    n = X.shape[0]
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(n, dtype=bool)
    intra = float(sim[same & off].mean()) if np.any(same & off) else float("nan")
    inter = float(sim[~same].mean()) if np.any(~same) else float("nan")
    ratio = intra / inter if inter and np.isfinite(inter) and inter != 0 else float("nan")
    return {"intra": intra, "inter": inter, "ratio": ratio}


def spatial_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    spatial_distance: np.ndarray | None = None,
    alpha: float = 0.5,
) -> float:
    """Silhouette score on a feature/spatial blended distance.

    Parameters
    ----------
    features : np.ndarray, shape (V, d)
    labels : np.ndarray, shape (V,)
    spatial_distance : np.ndarray, shape (V, V), optional
        Geodesic/graph distance; if given, blended with the feature distance.
    alpha : float
        Weight on feature distance (1 = features only).

    Returns
    -------
    float
        Mean silhouette in [-1, 1].
    """
    from sklearn.metrics import pairwise_distances, silhouette_score

    X = np.asarray(features, float)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    Dfeat = pairwise_distances(X)
    if spatial_distance is not None:
        Dsp = np.asarray(spatial_distance, float)

        def _norm(D):
            mx = D[np.isfinite(D)].max()
            return D / mx if mx > 0 else D

        D = alpha * _norm(Dfeat) + (1 - alpha) * _norm(Dsp)
    else:
        D = Dfeat
    return float(silhouette_score(D, labels, metric="precomputed"))
