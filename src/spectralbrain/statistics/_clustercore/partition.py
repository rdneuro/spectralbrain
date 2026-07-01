"""Partition-comparison indices, spatially aware and classical.

Classical ARI/NMI/V-measure treat vertices as exchangeable and ignore that a
*distant* misassignment should be penalised differently from a *near* one. The
spatially-aware Rand index (spRI) and its adjusted form (spARI) weight
disagreement pairs by spatial distance (Yan, Feng & Luo 2025, Biometrics).

IMPORTANT
---------
The spRI/spARI implementation here reproduces the *structure* of the index
(distance-weighted disagreement pairs, with an adjustment to a chance baseline)
using a documented kernel weighting. The original publication is R-only and its
exact weighting/expectation derivation has NOT yet been validated cell-for-cell
against that reference. The result is exposed with ``validated_against_R=False``
in the returned dict; run ``validation/validate_spari.py`` (analytical
properties + an R cell-for-cell bridge) and validate against the R package
before publication use.

References
----------
- Yan, Feng & Luo (2025), Spatially aware adjusted Rand index, Biometrics
  81(3):ujaf127.
- Meila (2007), Comparing clusterings - an information based distance,
  J. Multivariate Analysis 98(5):873-895.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("spectralbrain.statistics._clustercore")


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Classical Adjusted Rand Index (exact, via scikit-learn)."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_a, labels_b))


def normalized_mutual_info(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Classical Normalized Mutual Information (exact, via scikit-learn)."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(labels_a, labels_b))


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Variation of Information (Meila 2007), a true metric on partitions (bits).

    Returns
    -------
    float
        ``VI = H(A) + H(B) - 2 I(A; B)`` in bits (0 = identical partitions).
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("Partitions must have equal length.")

    def _entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / n
        return -np.sum(p * np.log2(p))

    # Mutual information in bits.
    contingency: Dict = {}
    for ai, bi in zip(a, b):
        contingency[(ai, bi)] = contingency.get((ai, bi), 0) + 1
    _, ca = np.unique(a, return_counts=True)
    _, cb = np.unique(b, return_counts=True)
    pa = {lab: c / n for lab, c in zip(np.unique(a), ca)}
    pb = {lab: c / n for lab, c in zip(np.unique(b), cb)}
    mi = 0.0
    for (ai, bi), nij in contingency.items():
        pij = nij / n
        mi += pij * np.log2(pij / (pa[ai] * pb[bi]))
    return float(_entropy(a) + _entropy(b) - 2.0 * mi)


def _pair_disagreement_weights(coords_or_dist: np.ndarray, length_scale: float,
                               is_distance: bool) -> np.ndarray:
    """Symmetric pairwise weight ``w_ij = exp(-d_ij / length_scale)`` in [0, 1]."""
    if is_distance:
        D = np.asarray(coords_or_dist, float)
    else:
        from scipy.spatial.distance import squareform, pdist
        D = squareform(pdist(np.asarray(coords_or_dist, float)))
    ls = float(length_scale)
    if ls <= 0:
        raise ValueError("length_scale must be positive.")
    return np.exp(-D / ls)


def spatial_rand_index(labels_a: np.ndarray, labels_b: np.ndarray,
                       coords: Optional[np.ndarray] = None,
                       distance: Optional[np.ndarray] = None,
                       length_scale: Optional[float] = None,
                       adjusted: bool = True) -> Dict[str, object]:
    """Spatially-aware (adjusted) Rand index.

    Concordant pairs (same/same or different/different in both partitions) score
    1; disagreement pairs score a distance-dependent weight ``w_ij`` in [0, 1]
    (closer disagreements are treated as less severe). The raw spRI is the mean
    pair score; the adjusted spARI rescales against a permutation chance baseline.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray, shape (V,)
    coords : np.ndarray, shape (V, k), optional
        Vertex coordinates; used to derive distances if ``distance`` is None.
    distance : np.ndarray, shape (V, V), optional
        Precomputed (e.g. geodesic) distance matrix; preferred for surfaces.
    length_scale : float, optional
        Kernel length scale; defaults to the median pairwise distance.
    adjusted : bool
        If True also compute spARI against a label-permutation baseline.

    Returns
    -------
    dict
        Keys: ``spRI``, ``spARI`` (NaN if ``adjusted=False``),
        ``length_scale``, ``validated_against_R`` (always False here).
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("Partitions must have equal length.")
    if distance is None and coords is None:
        raise ValueError("Provide coords or a precomputed distance matrix.")

    if distance is not None:
        D = np.asarray(distance, float)
        if length_scale is None:
            off = D[~np.eye(n, dtype=bool)]
            length_scale = float(np.median(off[np.isfinite(off)]))
        W = _pair_disagreement_weights(D, length_scale, is_distance=True)
    else:
        from scipy.spatial.distance import squareform, pdist
        Dvec = pdist(np.asarray(coords, float))
        if length_scale is None:
            length_scale = float(np.median(Dvec))
        W = _pair_disagreement_weights(coords, length_scale, is_distance=False)

    warnings.warn(
        "spRI/spARI here uses a documented kernel weighting not yet validated "
        "cell-for-cell against the R reference (Yan, Feng & Luo 2025). "
        "Validate before publication; see module docstring.",
        stacklevel=2,
    )

    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    concord = (same_a == same_b)                 # agreement pairs (bool VxV)

    iu = np.triu_indices(n, k=1)
    concord_u = concord[iu]
    w_u = W[iu]
    # Pair score: 1 if concordant else the distance weight.
    pair_score = np.where(concord_u, 1.0, w_u)
    spri = float(pair_score.mean())

    spari = float("nan")
    if adjusted:
        rng = np.random.default_rng(0)
        n_perm = 200
        baseline = np.empty(n_perm)
        for p in range(n_perm):
            bp = b[rng.permutation(n)]
            same_bp = bp[:, None] == bp[None, :]
            concord_p = (same_a == same_bp)[iu]
            baseline[p] = np.where(concord_p, 1.0, w_u).mean()
        exp_spri = float(baseline.mean())
        denom = 1.0 - exp_spri
        spari = (spri - exp_spri) / denom if abs(denom) > 1e-12 else float("nan")

    return {
        "spRI": spri,
        "spARI": spari,
        "length_scale": float(length_scale),
        "validated_against_R": False,
    }
