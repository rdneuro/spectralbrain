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
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("brainmosaic.statistics.cluster_stats")


def intra_inter_homogeneity(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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


def spatial_silhouette(features: np.ndarray, labels: np.ndarray,
                       spatial_distance: Optional[np.ndarray] = None,
                       alpha: float = 0.5) -> float:
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


def energy_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Energy distance between two multivariate samples (Szekely & Rizzo 2013).

    Parameters
    ----------
    sample_a : np.ndarray, shape (n_a, d)
    sample_b : np.ndarray, shape (n_b, d)

    Returns
    -------
    float
        Non-negative energy distance (0 iff distributions coincide).
    """
    from sklearn.metrics import pairwise_distances
    A = np.atleast_2d(np.asarray(sample_a, float))
    B = np.atleast_2d(np.asarray(sample_b, float))
    d_ab = pairwise_distances(A, B).mean()
    d_aa = pairwise_distances(A).mean()
    d_bb = pairwise_distances(B).mean()
    val = 2.0 * d_ab - d_aa - d_bb
    return float(max(val, 0.0))


def rsa_compare(rdm_a: np.ndarray, rdm_b: np.ndarray, method: str = "spearman") -> float:
    """Representational similarity: correlation of two RDMs' lower triangles.

    Parameters
    ----------
    rdm_a, rdm_b : np.ndarray, shape (V, V)
        Representational dissimilarity matrices.
    method : {"spearman", "pearson", "kendall"}

    Returns
    -------
    float
        Correlation coefficient between the off-diagonal entries.
    """
    from scipy.stats import spearmanr, pearsonr, kendalltau
    a = np.asarray(rdm_a, float)
    b = np.asarray(rdm_b, float)
    iu = np.triu_indices(a.shape[0], k=1)
    va, vb = a[iu], b[iu]
    if method == "spearman":
        return float(spearmanr(va, vb).correlation)
    if method == "pearson":
        return float(pearsonr(va, vb)[0])
    if method == "kendall":
        return float(kendalltau(va, vb).correlation)
    raise ValueError(f"Unknown method {method!r}.")


def mantel_test(rdm_a: np.ndarray, rdm_b: np.ndarray, n_perm: int = 10000,
                method: str = "spearman", random_state: int = 0):
    """Mantel test: permutation significance of an RDM-RDM correlation.

    Returns
    -------
    r : float
        Observed correlation.
    p : float
        Two-sided permutation p-value.
    """
    a = np.asarray(rdm_a, float)
    b = np.asarray(rdm_b, float)
    n = a.shape[0]
    r_obs = rsa_compare(a, b, method=method)
    rng = np.random.default_rng(random_state)
    iu = np.triu_indices(n, k=1)
    va = a[iu]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        bp = b[np.ix_(perm, perm)][iu]
        from scipy.stats import spearmanr, pearsonr
        if method == "spearman":
            r = spearmanr(va, bp).correlation
        else:
            r = pearsonr(va, bp)[0]
        if abs(r) >= abs(r_obs):
            count += 1
    return float(r_obs), float((count + 1) / (n_perm + 1))
