"""Cluster-vs-atlas comparison statistics.

Compare SpectralBrain's data-driven partitions against reference atlases with the
size-matched-null framework that the parcellation-evaluation literature
(Arslan 2018; Eickhoff 2015; Craddock 2012) requires: ARI/AMI/NMI/VI and
within-parcel homogeneity are all confounded by parcel number and size, so every
quality number is reported against a size-matched random contiguous
parcellation, never on its own.

This module complements :mod:`spectralbrain.utils.atlas` (which provides atlas
*label* lookups) with the *statistical* comparison of partitions.

- :func:`compare_partitions` -- ARI, AMI, NMI, VI.
- :func:`parcel_overlap` -- Dice/Jaccard contingency + best-match mapping.
- :func:`spectral_homogeneity` -- within-parcel descriptor homogeneity.
- :func:`random_parcellation` -- contiguous size-matched null (graph Voronoi).
- :func:`homogeneity_vs_null` -- observed homogeneity vs the null distribution.
- :func:`cluster_atlas_concordance` -- full report with a size-matched null z/p.
- :func:`aggregate_across_subjects` -- median + bootstrap CI for a cohort.
- :func:`spatial_rand_index` -- spatially-aware ARI (spARI).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Re-export the numpy-clean cores directly (harmonised names).
from spectralbrain.statistics._clustercore.atlas import (
    aggregate_across_subjects,
    compare_partitions,
    parcel_overlap,
    spectral_homogeneity,
)
from spectralbrain.statistics._clustercore.atlas import (
    cluster_atlas_concordance as _cluster_atlas_concordance,
)
from spectralbrain.statistics._clustercore.atlas import (
    homogeneity_vs_null as _homogeneity_vs_null,
)
from spectralbrain.statistics._clustercore.atlas import (
    random_parcellation as _random_parcellation,
)
from spectralbrain.statistics._clustercore.partition import (
    adjusted_rand_index,
    normalized_mutual_info,
    spatial_rand_index,
    variation_of_information,
)

__all__ = [
    "adjusted_rand_index",
    "aggregate_across_subjects",
    "cluster_atlas_concordance",
    "compare_partitions",
    "homogeneity_vs_null",
    "normalized_mutual_info",
    "parcel_overlap",
    "random_parcellation",
    "spatial_rand_index",
    "spectral_homogeneity",
    "variation_of_information",
]


def _adjacency(faces, adjacency_list, n_vertices):
    if adjacency_list is not None:
        return adjacency_list
    if faces is not None:
        from spectralbrain.statistics._clustercore import adjacency_list_from_faces
        return adjacency_list_from_faces(np.asarray(faces), n_vertices)
    raise ValueError("Provide `faces` or `adjacency_list` for the size-matched null.")


def random_parcellation(n_parcels: int, *, faces: np.ndarray | None = None,
                        adjacency_list: Sequence[np.ndarray] | None = None,
                        n_vertices: int | None = None, seed: int = 0):
    """Contiguous size-matched random parcellation (graph Voronoi).

    Accepts either ``faces`` (preferred) or a precomputed ``adjacency_list``.
    """
    if n_vertices is None and faces is not None:
        n_vertices = int(np.asarray(faces).max()) + 1
    adj = _adjacency(faces, adjacency_list, n_vertices)
    return _random_parcellation(adj, n_parcels, seed=seed)


def homogeneity_vs_null(features: np.ndarray, labels: np.ndarray, *,
                        faces: np.ndarray | None = None,
                        adjacency_list: Sequence[np.ndarray] | None = None,
                        n_null: int = 100, method: str = "correlation",
                        seed: int = 0, progress: bool = False) -> dict:
    """Within-parcel homogeneity vs a size-matched random-parcellation null.

    Accepts ``faces`` or ``adjacency_list``. Returns ``observed``, ``null_mean``,
    ``null_std``, ``z`` and one-sided empirical ``p``.
    """
    adj = _adjacency(faces, adjacency_list, np.asarray(features).shape[0])
    return _homogeneity_vs_null(features, labels, adj, n_null=n_null,
                                method=method, seed=seed, progress=progress)


def cluster_atlas_concordance(cluster_labels: np.ndarray, atlas_labels: np.ndarray,
                              *, faces: np.ndarray | None = None,
                              adjacency_list: Sequence[np.ndarray] | None = None,
                              distance: np.ndarray | None = None,
                              coords: np.ndarray | None = None,
                              background: int | None = None,
                              n_null: int = 100, seed: int = 0,
                              progress: bool = False) -> dict:
    """Full cluster-vs-atlas concordance with a size-matched ARI null.

    Computes ARI/AMI/NMI/VI and mean best-Dice between the data-driven clusters
    and the atlas, optionally spARI (if ``distance``/``coords`` given), and the
    distribution of ARI between a size-matched random contiguous parcellation and
    the atlas. A significant positive ARI z-score means the clusters match the
    atlas more than a random parcellation of the same granularity would
    (Arslan 2018). Accepts ``faces`` or ``adjacency_list`` for the null.
    """
    adj = None
    if faces is not None or adjacency_list is not None:
        adj = _adjacency(faces, adjacency_list, np.asarray(cluster_labels).shape[0])
    return _cluster_atlas_concordance(
        cluster_labels, atlas_labels, adjacency_list=adj, distance=distance,
        coords=coords, background=background, n_null=n_null, seed=seed,
        progress=progress)
