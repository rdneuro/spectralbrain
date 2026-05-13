"""SpectralBrain statistics — EDA, frequentist, Bayesian, normative, surrogates."""

from spectralbrain.statistics.eda import (  # noqa: F401
    SpectralQCReport, spectral_qc,
    OptimalKResult, optimal_k,
    descriptor_profile, descriptor_correlation,
    compute_icc, batch_effect_scan, eigenvalue_stability,
    DescriptorRecommendation, recommend_descriptor,
)
from spectralbrain.statistics.analysis import (  # noqa: F401
    VertexWiseResult,
    vertexwise_ttest, vertexwise_mannwhitney, vertexwise_permutation,
    tfce, cohens_d_map, hedges_g_map,
    vertexwise_correlation, surprise_map, surprise_map_percentile,
    ClassificationResult, classify,
    fisher_vector, fit_gmm_codebook, bag_of_spectral_words,
    kernel_mean_embedding,
    emd_distance, kl_divergence, js_divergence, energy_distance,
)
from spectralbrain.statistics.bayesian import (  # noqa: F401
    BayesianModel, HorseshoeRegression,
    BayesianGroupComparison, HierarchicalLinearModel,
    GaussianProcessNormative, BayesianSpatialModel,
    BayesianConnectome,
)
from spectralbrain.statistics.normative import (  # noqa: F401
    NormativeModel, centile_curves, z_score_map, extreme_value_map,
    NonInferiorityResult, non_inferiority_test,
    equivalence_test_tost, auc_comparison_delong,
    MethodComparisonResult, compare_methods,
)
from spectralbrain.statistics.surrogates import (  # noqa: F401
    bootstrap_ci, bootstrap_paired_difference,
    null_eigenvalue_permutation, null_phase_randomisation,
    null_spin_permutation, null_subject_permutation,
    null_edge_rewiring, null_parametric,
    SyntheticDescriptors, SyntheticMesh, SyntheticPointCloud,
)
