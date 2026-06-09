"""SpectralBrain statistics — EDA, frequentist, Bayesian, normative, surrogates, clustering."""

from spectralbrain.statistics.analysis import (  # noqa: F401
    ClassificationResult,
    VertexWiseResult,
    bag_of_spectral_words,
    classify,
    cohens_d_map,
    emd_distance,
    energy_distance,
    fisher_vector,
    fit_gmm_codebook,
    hedges_g_map,
    js_divergence,
    kernel_mean_embedding,
    kl_divergence,
    surprise_map,
    surprise_map_percentile,
    tfce,
    vertexwise_correlation,
    vertexwise_mannwhitney,
    vertexwise_permutation,
    vertexwise_ttest,
)
from spectralbrain.statistics.bayesian import (  # noqa: F401
    BayesianConnectome,
    BayesianGroupComparison,
    BayesianModel,
    BayesianSpatialModel,
    GaussianProcessNormative,
    HierarchicalLinearModel,
    HorseshoeRegression,
)
from spectralbrain.statistics.clustering import (  # noqa: F401
    BayesianClusterConfirmation,
    # Result containers
    ClusterResult,
    FusionResult,
    MapperResult,
    ScaleSpaceBlobResult,
    TemporalClusterResult,
    TensorDecompositionResult,
    VineyardResult,
    # Convenience
    auto_cluster,
    # Distance / affinity construction
    build_descriptor_distance,
    build_hks_affinity_graph,
    build_hybrid_distance,
    cluster_comparison,
    cluster_dpmm,
    cluster_gnmf,
    # Spatial clustering
    cluster_hdbscan,
    cluster_joint_spectral,
    cluster_leiden,
    # Mapper TDA
    cluster_mapper,
    # Multi-view clustering
    cluster_multiview,
    cluster_persistence,
    # Quality metrics
    cluster_quality,
    # Scale-space blob tracking
    cluster_scalespace_blobs,
    # Spatio-temporal joint clustering
    cluster_spatiotemporal_gnmf,
    cluster_spatiotemporal_stdbscan,
    cluster_spectral_coclustering,
    cluster_temporal_dtw,
    # Temporal / scale clustering
    cluster_temporal_fpca,
    # Tensor decomposition
    cluster_tensor_decomposition,
    # Persistence vineyards
    cluster_vineyards,
    # Wavelet clustering
    cluster_wavelet_coefficients,
    # Bayesian cluster confirmation
    confirm_clusters_bayesian,
    # Joint time-vertex GSP
    denoise_joint_timevertex,
    # HKS + WKS descriptor fusion
    fuse_concatenate,
    fuse_joint_nmf,
    fuse_multi_kernel,
)
from spectralbrain.statistics.eda import (  # noqa: F401
    DescriptorRecommendation,
    OptimalKResult,
    SpectralQCReport,
    batch_effect_scan,
    compute_icc,
    descriptor_correlation,
    descriptor_profile,
    eigenvalue_stability,
    optimal_k,
    recommend_descriptor,
    spectral_qc,
)
from spectralbrain.statistics.normative import (  # noqa: F401
    HarmonizationResult,
    MethodComparisonResult,
    NonInferiorityResult,
    NormativeModel,
    auc_comparison_delong,
    centile_curves,
    compare_methods,
    equivalence_test_tost,
    extreme_value_map,
    harmonize,
    harmonize_combat,
    harmonize_combat_gam,
    non_inferiority_test,
    z_score_map,
)
from spectralbrain.statistics.surrogates import (  # noqa: F401
    SyntheticDescriptors,
    SyntheticMesh,
    SyntheticPointCloud,
    bootstrap_ci,
    bootstrap_paired_difference,
    null_edge_rewiring,
    null_eigenvalue_permutation,
    null_parametric,
    null_phase_randomisation,
    null_spin_permutation,
    null_subject_permutation,
)
