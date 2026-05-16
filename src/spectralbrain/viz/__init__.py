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
from spectralbrain.viz.graphics import (  # noqa: F401
    PALETTE, COLOR_CONTROL, COLOR_PATIENT, COLOR_SIGNIFICANT, COLOR_NS,
    CMAP_DIVERGING, CMAP_SEQUENTIAL, CMAP_SPECTRAL, CMAP_QUALITATIVE,
    set_style, figure, savefig, distplot,
    plot_volcano, plot_roc_curve, plot_rdm,
    plot_connectome_matrix, plot_embedding,
    plot_effect_size_distribution, plot_laterality,
    plot_pvalue_histogram,
)

# ── brainplots.py ──
from spectralbrain.viz.brainplots import (  # noqa: F401
    VIEWS_CORTEX, VIEWS_FULL, VIEWS_MEDIAL, DESCRIPTOR_STYLES,
    BrainPlotSpec,
    plot_brain, plot_brain_subcortical, plot_brain_tracts,
    plot_group_comparison, plot_normative_map, plot_clustering_map,
    plot_morphometric_gallery, plot_top10_morphometrics,
    plot_multi_descriptor_panel, plot_bilateral_comparison,
    plot_spectral_progression,
)

# ── hipp.py ──
from spectralbrain.viz.hipp import (  # noqa: F401
    DENSITIES,
    plot_hippocampus, plot_hippocampus_bilateral,
    plot_hippocampus_comparison, plot_hippocampus_gallery,
    plot_hippocampus_normative,
    plot_hippocampus_spatiotemporal, plot_hippocampus_hovmoller,
)

# ── bayes.py ──
from spectralbrain.viz.bayes import (  # noqa: F401
    plot_posterior, plot_forest, plot_prior_posterior,
    plot_rope_decision, plot_ridgeline,
    plot_horseshoe_coefficients, plot_best_posterior,
    plot_site_effects, plot_gp_trajectory,
    plot_connectome_posterior,
)

# ── geometry.points ──
from spectralbrain.viz.geometry.points import (  # noqa: F401
    plot_point_cloud, plot_mls_reconstruction,
    plot_clusters, plot_point_cloud_panel,
    plot_warp, plot_voronoi, plot_point_cloud_o3d,
)

# ── geometry.meshes ──
from spectralbrain.viz.geometry.meshes import (  # noqa: F401
    CURVATURE_METHODS, CAMERA_PRESETS,
    plot_mesh, plot_wireframe, plot_curvature,
    plot_multi_view, plot_mesh_comparison,
    plot_scalar_difference, plot_mesh_pyvista,
)

# ── clusters ──
from spectralbrain.viz.clusters import (  # noqa: F401
    CLUSTER_COLORS, VIEWS_3POSE,
    # 3D mesh renders (vedo)
    plot_cluster_map, plot_cluster_boundaries,
    plot_method_comparison_3d, plot_gnmf_components,
    plot_soft_membership, plot_cluster_exploded,
    plot_hks_cluster_progression, plot_fusion_panel,
    # 2D statistical plots (matplotlib)
    plot_cluster_profiles, plot_silhouette_diagram,
    plot_quality_comparison, plot_agreement_heatmap,
    plot_persistence_diagram, plot_gnmf_temporal_factors,
    plot_bayesian_confirmation, plot_cluster_sizes,
    plot_cluster_scatter, plot_coclustering_heatmap,
    # Summary panel
    plot_cluster_summary,
    # Spatio-temporal field visualization
    plot_spatiotemporal_field, plot_spatiotemporal_animation,
    plot_hovmoller, plot_kymograph,
    plot_warped_surface, plot_descriptor_evolution_comparison,
)
