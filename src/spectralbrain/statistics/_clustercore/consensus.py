"""Consensus / ensemble clustering from a co-association matrix.

Aggregates multiple partitions (ddCRP draws, Louvain, Ward, HDBSCAN, ...) into a
co-association matrix and derives a single consensus partition. Increases
robustness and yields a per-vertex stability estimate.

References
----------
Strehl & Ghosh (2002), Cluster ensembles, JMLR 3:583-617.
Fred & Jain (2005), Combining multiple clusterings using evidence accumulation,
IEEE TPAMI 27(6):835-850.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Sequence
import numpy as np

logger = logging.getLogger("spectralbrain.statistics._clustercore")


def co_association_matrix(partitions: Sequence[np.ndarray]) -> np.ndarray:
    """Fraction of partitions in which each vertex pair co-clusters.

    Parameters
    ----------
    partitions : sequence of (V,) label arrays

    Returns
    -------
    np.ndarray, shape (V, V)
        Symmetric co-association in [0, 1].
    """
    partitions = [np.asarray(p) for p in partitions]
    n = partitions[0].shape[0]
    if any(p.shape[0] != n for p in partitions):
        raise ValueError("All partitions must have the same length.")
    co = np.zeros((n, n), dtype=np.float64)
    for p in partitions:
        co += (p[:, None] == p[None, :]).astype(np.float64)
    co /= len(partitions)
    return co


def consensus_partition(co_association: np.ndarray, n_clusters: Optional[int] = None,
                        connectivity=None, threshold: Optional[float] = None,
                        ) -> np.ndarray:
    """Derive a consensus partition from a co-association matrix.

    Parameters
    ----------
    co_association : np.ndarray, shape (V, V)
    n_clusters : int, optional
        If given, average-linkage agglomeration on ``1 - co`` to this many
        clusters (optionally connectivity-constrained for contiguity).
    connectivity : scipy.sparse matrix, optional
        Mesh adjacency for contiguous agglomeration.
    threshold : float, optional
        Alternative to ``n_clusters``: connect vertices with co >= threshold and
        take connected components.

    Returns
    -------
    np.ndarray, shape (V,)
    """
    co = np.asarray(co_association, float)
    if threshold is not None:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        A = (co >= threshold).astype(np.int8)
        np.fill_diagonal(A, 0)
        _, labels = connected_components(sp.csr_matrix(A), directed=False)
        return labels
    if n_clusters is None:
        raise ValueError("Provide either n_clusters or threshold.")
    from sklearn.cluster import AgglomerativeClustering
    dist = 1.0 - co
    ac = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                 linkage="average", connectivity=connectivity)
    return ac.fit_predict(dist)


def stability_per_vertex(co_association: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean within-cluster co-association for each vertex (a stability score)."""
    co = np.asarray(co_association, float)
    labels = np.asarray(labels)
    stab = np.zeros(labels.shape[0])
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size > 1:
            block = co[np.ix_(idx, idx)]
            stab[idx] = (block.sum(axis=1) - 1.0) / (idx.size - 1)
        else:
            stab[idx] = 1.0
    return stab
