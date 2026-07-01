"""Compare brainmosaic's data-driven parcellations against reference atlases.

The central methodological lesson of the parcellation-evaluation literature
(Arslan 2018; Eickhoff 2015; Craddock 2012) is that ARI/AMI/NMI/VI and
within-parcel homogeneity are ALL confounded by the number and size of parcels,
and that "any clustering finds clusters". Therefore every quality number here is
designed to be reported **against a size-matched random (contiguous) parcellation
null**, never on its own.

Functions
---------
- :func:`compare_partitions` -- ARI, AMI, NMI, VI between two labellings.
- :func:`parcel_overlap` -- Dice/Jaccard contingency + best-match mapping.
- :func:`spectral_homogeneity` -- within-parcel descriptor homogeneity.
- :func:`random_parcellation` -- contiguous size-matched null (graph Voronoi).
- :func:`homogeneity_vs_null` -- observed homogeneity vs the null distribution.
- :func:`cluster_atlas_concordance` -- full report with a size-matched null z/p.
- :func:`aggregate_across_subjects` -- median + bootstrap CI for a clinical cohort.

References
----------
- Arslan et al. (2018), NeuroImage 170:5-30 (systematic parcellation comparison).
- Eickhoff, Thirion, Varoquaux & Bzdok (2015), HBM 36(12):4771-4792.
- Craddock et al. (2012), HBM 33(8):1914-1928 (homogeneity vs random).
- Schaefer et al. (2018), Cereb Cortex 28(9):3095-3114; Gordon et al. (2016).
- Hubert & Arabie (1985); Vinh, Epps & Bailey (2010); Meila (2007).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("spectralbrain.statistics._clustercore")


def compare_partitions(labels_a: np.ndarray, labels_b: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """ARI, AMI, NMI and variation of information between two labellings.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray, shape (V,)
    mask : np.ndarray of bool, optional
        If given, restrict to ``mask`` (e.g. drop medial-wall/background vertices).

    Returns
    -------
    dict
        Keys ``ari`` (Hubert & Arabie 1985), ``ami`` and ``nmi`` (Vinh 2010),
        ``vi_bits`` (Meila 2007), ``n_clusters_a``, ``n_clusters_b``.

    Notes
    -----
    All of these depend on cluster count/size; compare only granularity-matched
    partitions, or interpret against a size-matched null (see
    :func:`cluster_atlas_concordance`).
    """
    from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                                 normalized_mutual_info_score)
    from .partition import variation_of_information

    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if mask is not None:
        mask = np.asarray(mask, bool)
        a, b = a[mask], b[mask]
    return {
        "ari": float(adjusted_rand_score(a, b)),
        "ami": float(adjusted_mutual_info_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
        "vi_bits": float(variation_of_information(a, b)),
        "n_clusters_a": int(np.unique(a).size),
        "n_clusters_b": int(np.unique(b).size),
    }


def parcel_overlap(cluster_labels: np.ndarray, atlas_labels: np.ndarray,
                   background: Optional[int] = None) -> Dict[str, object]:
    """Dice/Jaccard overlap between data-driven clusters and atlas ROIs.

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (V,)
    atlas_labels : np.ndarray, shape (V,)
    background : int, optional
        Atlas label to exclude (e.g. 0 medial wall).

    Returns
    -------
    dict
        ``dice`` (C x A matrix), ``jaccard`` (C x A), ``cluster_ids``,
        ``atlas_ids``, ``best_atlas_for_cluster`` (mapping + Dice),
        ``mean_best_dice`` (mean over clusters of their best atlas Dice).
    """
    cl = np.asarray(cluster_labels)
    at = np.asarray(atlas_labels)
    keep = np.ones(cl.shape[0], bool)
    if background is not None:
        keep &= at != background
    cl, at = cl[keep], at[keep]

    cluster_ids = np.unique(cl)
    atlas_ids = np.unique(at)
    dice = np.zeros((cluster_ids.size, atlas_ids.size))
    jacc = np.zeros_like(dice)
    cl_masks = {c: cl == c for c in cluster_ids}
    at_masks = {a: at == a for a in atlas_ids}
    for ci, c in enumerate(cluster_ids):
        mc = cl_masks[c]
        sc = mc.sum()
        for ai, a in enumerate(atlas_ids):
            ma = at_masks[a]
            inter = np.count_nonzero(mc & ma)
            sa = ma.sum()
            dice[ci, ai] = 2.0 * inter / (sc + sa) if (sc + sa) else 0.0
            union = sc + sa - inter
            jacc[ci, ai] = inter / union if union else 0.0

    best_idx = dice.argmax(axis=1) if atlas_ids.size else np.array([], int)
    best = {int(cluster_ids[i]): {"atlas_id": int(atlas_ids[best_idx[i]]),
                                  "dice": float(dice[i, best_idx[i]])}
            for i in range(cluster_ids.size)} if atlas_ids.size else {}
    mean_best = float(np.mean([v["dice"] for v in best.values()])) if best else float("nan")
    return {"dice": dice, "jaccard": jacc, "cluster_ids": cluster_ids,
            "atlas_ids": atlas_ids, "best_atlas_for_cluster": best,
            "mean_best_dice": mean_best}


def spectral_homogeneity(features: np.ndarray, labels: np.ndarray,
                         method: str = "correlation",
                         size_weighted: bool = True) -> float:
    """Within-parcel descriptor homogeneity (Schaefer/Gordon/Craddock style).

    Parameters
    ----------
    features : np.ndarray, shape (V, d)
        Per-vertex descriptors (e.g. HKS/WKS/SI-HKS).
    labels : np.ndarray, shape (V,)
    method : {"correlation", "first_eigenvalue"}
        ``"correlation"``: mean pairwise correlation of member feature vectors
        within a parcel. ``"first_eigenvalue"``: fraction of variance explained
        by the first principal component within a parcel.
    size_weighted : bool
        Weight parcels by size when averaging.

    Returns
    -------
    float
        Mean within-parcel homogeneity in [0, 1] (approx).

    Notes
    -----
    Homogeneity rises mechanically as parcels shrink; ALWAYS compare against a
    size-matched null (:func:`homogeneity_vs_null`).
    """
    X = np.asarray(features, float)
    labels = np.asarray(labels)
    vals, weights = [], []
    for c in np.unique(labels):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        if idx.size < 2:
            continue
        Xp = X[idx]
        if method == "correlation":
            C = np.corrcoef(Xp)
            iu = np.triu_indices(idx.size, k=1)
            h = np.nanmean(C[iu])
        elif method == "first_eigenvalue":
            Xc = Xp - Xp.mean(axis=0, keepdims=True)
            s = np.linalg.svd(Xc, compute_uv=False)
            h = float(s[0] ** 2 / np.clip((s ** 2).sum(), 1e-12, None))
        else:
            raise ValueError("method must be 'correlation' or 'first_eigenvalue'.")
        if np.isfinite(h):
            vals.append(h)
            weights.append(idx.size)
    if not vals:
        return float("nan")
    if size_weighted:
        return float(np.average(vals, weights=weights))
    return float(np.mean(vals))


def random_parcellation(adjacency_list: Sequence[np.ndarray], n_parcels: int,
                        seed: int = 0) -> np.ndarray:
    """Contiguous random parcellation via multi-source BFS (graph Voronoi).

    Picks ``n_parcels`` random seed vertices and grows regions simultaneously,
    yielding spatially-contiguous parcels of roughly matched size -- the
    size/number-matched null recommended by Arslan 2018 for parcellation-quality
    comparison.

    Parameters
    ----------
    adjacency_list : sequence of int arrays
        Mesh neighbours per vertex.
    n_parcels : int
    seed : int

    Returns
    -------
    np.ndarray, shape (V,)
        Contiguous random parcellation (labels 0..n_parcels-1).
    """
    n = len(adjacency_list)
    if not 1 <= n_parcels <= n:
        raise ValueError(f"n_parcels must be in [1, {n}].")
    rng = np.random.default_rng(seed)
    seeds = rng.choice(n, size=n_parcels, replace=False)
    labels = np.full(n, -1, dtype=np.int64)
    dq = deque()
    for c, s in enumerate(seeds):
        labels[s] = c
        dq.append(s)
    while dq:
        u = dq.popleft()
        for w in adjacency_list[u]:
            if labels[w] == -1:
                labels[w] = labels[u]
                dq.append(w)
    # Assign any disconnected leftovers to a random existing parcel.
    missing = np.where(labels < 0)[0]
    if missing.size:
        labels[missing] = rng.integers(0, n_parcels, size=missing.size)
    return labels


def homogeneity_vs_null(features: np.ndarray, labels: np.ndarray,
                        adjacency_list: Sequence[np.ndarray], n_null: int = 100,
                        method: str = "correlation", seed: int = 0,
                        progress: bool = False) -> Dict[str, float]:
    """Observed within-parcel homogeneity vs a size-matched random-parcellation null.

    Returns
    -------
    dict
        ``observed``, ``null_mean``, ``null_std``, ``z`` (standardised effect),
        ``p`` (one-sided empirical p that random >= observed), ``n_parcels``.
    """
    from ._progress import progress_bar

    labels = np.asarray(labels)
    n_parcels = int(np.unique(labels[labels >= 0]).size)
    observed = spectral_homogeneity(features, labels, method=method)
    null = np.empty(n_null)
    with progress_bar("Homogeneity null", total=n_null, disable=not progress) as adv:
        for i in range(n_null):
            rp = random_parcellation(adjacency_list, n_parcels, seed=seed + i)
            null[i] = spectral_homogeneity(features, rp, method=method)
            adv()
    null_mean, null_std = float(np.nanmean(null)), float(np.nanstd(null))
    z = (observed - null_mean) / null_std if null_std > 0 else float("nan")
    p = float((np.sum(null >= observed) + 1) / (n_null + 1))
    return {"observed": float(observed), "null_mean": null_mean,
            "null_std": null_std, "z": float(z), "p": p, "n_parcels": n_parcels}


def cluster_atlas_concordance(cluster_labels: np.ndarray, atlas_labels: np.ndarray,
                              adjacency_list: Optional[Sequence[np.ndarray]] = None,
                              distance: Optional[np.ndarray] = None,
                              coords: Optional[np.ndarray] = None,
                              background: Optional[int] = None,
                              n_null: int = 100, seed: int = 0,
                              progress: bool = False) -> Dict[str, object]:
    """Full cluster-vs-atlas concordance report with a size-matched null.

    Computes ARI/AMI/NMI/VI and mean best-Dice between the data-driven clusters
    and the atlas, optionally spARI (if ``distance`` or ``coords`` given), and --
    crucially -- compares the observed ARI to the distribution of ARI between a
    size-matched random contiguous parcellation and the atlas (Arslan 2018). A
    significant positive ARI z-score means the clusters match the atlas more than
    a random parcellation of the same granularity would.

    Parameters
    ----------
    cluster_labels, atlas_labels : np.ndarray, shape (V,)
    adjacency_list : sequence of int arrays, optional
        Required for the size-matched random-parcellation null.
    distance : np.ndarray, optional
        (V, V) geodesic/graph distance for spARI.
    coords : np.ndarray, optional
        (V, k) coordinates for spARI (used if ``distance`` is None).
    background : int, optional
        Atlas background label to drop.
    n_null : int
        Random parcellations for the ARI null.
    seed : int

    Returns
    -------
    dict
        ``metrics`` (compare_partitions), ``overlap`` (parcel_overlap summary),
        ``spARI`` (if available), and ``ari_null`` (observed/null_mean/z/p).
    """
    from ._progress import progress_bar
    from sklearn.metrics import adjusted_rand_score

    cl = np.asarray(cluster_labels)
    at = np.asarray(atlas_labels)
    mask = np.ones(cl.shape[0], bool)
    if background is not None:
        mask &= at != background

    metrics = compare_partitions(cl, at, mask=mask)
    overlap = parcel_overlap(cl, at, background=background)
    report: Dict[str, object] = {
        "metrics": metrics,
        "overlap": {"mean_best_dice": overlap["mean_best_dice"],
                    "n_clusters": int(overlap["cluster_ids"].size),
                    "n_atlas_rois": int(overlap["atlas_ids"].size)},
    }

    if distance is not None or coords is not None:
        from .partition import spatial_rand_index
        d = distance[np.ix_(mask, mask)] if distance is not None else None
        c = coords[mask] if coords is not None else None
        report["spARI"] = spatial_rand_index(cl[mask], at[mask], distance=d,
                                             coords=c, adjusted=True)

    if adjacency_list is not None:
        obs = float(adjusted_rand_score(cl[mask], at[mask]))
        n_parcels = int(np.unique(cl[mask]).size)
        null = np.empty(n_null)
        with progress_bar("ARI null", total=n_null, disable=not progress) as adv:
            for i in range(n_null):
                rp = random_parcellation(adjacency_list, n_parcels, seed=seed + i)
                null[i] = adjusted_rand_score(rp[mask], at[mask])
                adv()
        nm, ns = float(np.mean(null)), float(np.std(null))
        report["ari_null"] = {
            "observed": obs, "null_mean": nm, "null_std": ns,
            "z": (obs - nm) / ns if ns > 0 else float("nan"),
            "p": float((np.sum(null >= obs) + 1) / (n_null + 1)),
        }
    return report


def aggregate_across_subjects(values: Sequence[float], n_boot: int = 5000,
                              ci: float = 0.95, seed: int = 0) -> Dict[str, float]:
    """Median + bootstrap CI of a per-subject metric (clinical-cohort summary).

    For a small clinical cohort, prefer per-subject cluster-vs-atlas comparison
    followed by this aggregation over a single group statistic (Eickhoff 2015).

    Returns
    -------
    dict
        ``median``, ``ci_low``, ``ci_high``, ``mean``, ``n``.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "mean": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([np.median(rng.choice(v, size=v.size, replace=True))
                     for _ in range(n_boot)])
    lo = float(np.percentile(boot, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot, 100 * (1 - (1 - ci) / 2)))
    return {"median": float(np.median(v)), "ci_low": lo, "ci_high": hi,
            "mean": float(np.mean(v)), "n": int(v.size)}
