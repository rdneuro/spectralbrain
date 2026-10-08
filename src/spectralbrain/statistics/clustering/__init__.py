"""Clustering of spectral descriptors and partition statistics -- one subpackage.

- :mod:`.methods` -- spatial, temporal and spatio-temporal clustering
  (HDBSCAN, Leiden, GNMF, DPMM, persistence, Mapper, tensor, wavelet,
  multiview), descriptor fusion, quality metrics and Bayesian confirmation.
- :mod:`.ddcrp` -- distance-dependent Chinese Restaurant Process clustering
  (NIW collapsed likelihood), functional ddCRP, consensus and autotuning.
- :mod:`.atlas_stats` -- cluster-vs-atlas comparison: ARI / AMI / NMI / VI,
  spatial ARI, parcel overlap, homogeneity against size-matched null
  parcellations, cohort aggregation.
- ``_core`` -- the vendored numpy cores behind ``ddcrp`` and ``atlas_stats``
  (private).
"""

from spectralbrain.statistics.clustering.atlas_stats import (  # noqa: F401
    adjusted_rand_index,
    aggregate_across_subjects,
    cluster_atlas_concordance,
    compare_partitions,
    homogeneity_vs_null,
    normalized_mutual_info,
    parcel_overlap,
    random_parcellation,
    spatial_rand_index,
    spectral_homogeneity,
    variation_of_information,
)
from spectralbrain.statistics.clustering.ddcrp import (  # noqa: F401
    DDCRPTuningResult,
    autotune_ddcrp,
    cluster_consensus,
    cluster_ddcrp,
    cluster_ddcrp_functional,
)
from spectralbrain.statistics.clustering.methods import (  # noqa: F401
    BayesianClusterConfirmation,
    ClusterResult,
    FusionResult,
    MapperResult,
    ScaleSpaceBlobResult,
    TemporalClusterResult,
    TensorDecompositionResult,
    VineyardResult,
    auto_cluster,
    build_descriptor_distance,
    build_hks_affinity_graph,
    build_hybrid_distance,
    cluster_comparison,
    cluster_dpmm,
    cluster_gnmf,
    cluster_hdbscan,
    cluster_joint_spectral,
    cluster_leiden,
    cluster_mapper,
    cluster_multiview,
    cluster_persistence,
    cluster_quality,
    cluster_scalespace_blobs,
    cluster_spatiotemporal_gnmf,
    cluster_spatiotemporal_stdbscan,
    cluster_spectral_coclustering,
    cluster_temporal_dtw,
    cluster_temporal_fpca,
    cluster_tensor_decomposition,
    cluster_vineyards,
    cluster_wavelet_coefficients,
    confirm_clusters_bayesian,
    denoise_joint_timevertex,
    fuse_concatenate,
    fuse_joint_nmf,
    fuse_multi_kernel,
)
