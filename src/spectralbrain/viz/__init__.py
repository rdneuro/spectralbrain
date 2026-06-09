"""SpectralBrain visualization — 7 modules across 2D stats, 3D brain, geometry, and clustering.

============================  ==========================  ============================
Viz module                    Domain                      Paired statistics module
============================  ==========================  ============================
``graphics``                  Palettes, distplot, stats   ``statistics.analysis``
``brainplots``                Cortical / subcortical 3D   ``statistics.normative``
``hipp``                      Hippocampal surfaces        ``statistics.normative``
``bayes``                     Posterior / trace / ROPE     ``statistics.bayesian``
``geometry.points``           Point cloud 3D renders      ``statistics.eda``
``geometry.meshes``           Mesh 3D renders             ``statistics.eda``
``clusters``                  Cluster 3D + 2D plots       ``statistics.clustering``
============================  ==========================  ============================
"""

# ── graphics.py ──
# ── bayes.py ──
from spectralbrain.viz.bayes import (  # noqa: F401
    plot_best_posterior,
    plot_connectome_posterior,
    plot_forest,
    plot_gp_trajectory,
    plot_horseshoe_coefficients,
    plot_posterior,
    plot_prior_posterior,
    plot_ridgeline,
    plot_rope_decision,
    plot_site_effects,
)

# ── brainplots.py ──
from spectralbrain.viz.brainplots import (  # noqa: F401
    DESCRIPTOR_STYLES,
    VIEWS_CORTEX,
    VIEWS_FULL,
    VIEWS_MEDIAL,
    BrainPlotSpec,
    plot_bilateral_comparison,
    plot_brain,
    plot_brain_subcortical,
    plot_brain_tracts,
    plot_clustering_map,
    plot_group_comparison,
    plot_morphometric_gallery,
    plot_multi_descriptor_panel,
    plot_normative_map,
    plot_spectral_progression,
    plot_top10_morphometrics,
)

# ── clusters ──
from spectralbrain.viz.clusters import (  # noqa: F401
    CLUSTER_COLORS,
    VIEWS_3POSE,
    plot_agreement_heatmap,
    plot_bayesian_confirmation,
    plot_cluster_boundaries,
    plot_cluster_exploded,
    # 3D mesh renders (vedo)
    plot_cluster_map,
    # 2D statistical plots (matplotlib)
    plot_cluster_profiles,
    plot_cluster_scatter,
    plot_cluster_sizes,
    # Summary panel
    plot_cluster_summary,
    plot_coclustering_heatmap,
    plot_descriptor_evolution_comparison,
    plot_fusion_panel,
    plot_gnmf_components,
    plot_gnmf_temporal_factors,
    plot_hks_cluster_progression,
    plot_hovmoller,
    plot_kymograph,
    plot_method_comparison_3d,
    plot_persistence_diagram,
    plot_quality_comparison,
    plot_silhouette_diagram,
    plot_soft_membership,
    plot_spatiotemporal_animation,
    # Spatio-temporal field visualization
    plot_spatiotemporal_field,
    plot_warped_surface,
)

# ── geometry.meshes ──
from spectralbrain.viz.geometry.meshes import (  # noqa: F401
    CAMERA_PRESETS,
    CURVATURE_METHODS,
    plot_curvature,
    plot_mesh,
    plot_mesh_comparison,
    plot_mesh_pyvista,
    plot_multi_view,
    plot_scalar_difference,
    plot_wireframe,
)

# ── geometry.points ──
from spectralbrain.viz.geometry.points import (  # noqa: F401
    plot_clusters,
    plot_mls_reconstruction,
    plot_point_cloud,
    plot_point_cloud_o3d,
    plot_point_cloud_panel,
    plot_voronoi,
    plot_warp,
)
from spectralbrain.viz.graphics import (  # noqa: F401
    CMAP_DIVERGING,
    CMAP_QUALITATIVE,
    CMAP_SEQUENTIAL,
    CMAP_SPECTRAL,
    COLOR_CONTROL,
    COLOR_NS,
    COLOR_PATIENT,
    COLOR_SIGNIFICANT,
    PALETTE,
    distplot,
    figure,
    plot_connectome_matrix,
    plot_effect_size_distribution,
    plot_embedding,
    plot_laterality,
    plot_pvalue_histogram,
    plot_rdm,
    plot_roc_curve,
    plot_volcano,
    savefig,
    set_style,
)

# ── hipp.py ──
from spectralbrain.viz.hipp import (  # noqa: F401
    DENSITIES,
    plot_hippocampus,
    plot_hippocampus_bilateral,
    plot_hippocampus_comparison,
    plot_hippocampus_gallery,
    plot_hippocampus_hovmoller,
    plot_hippocampus_normative,
    plot_hippocampus_spatiotemporal,
)

# ── hipp3d.py (template-free vedo six-view) ──
from spectralbrain.viz.hipp3d import (  # noqa: F401
    SIXVIEWS,
    plot_hippocampus_sixview,
    plot_surface_sixview,
)
