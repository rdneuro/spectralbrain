"""SpectralBrain visualisation -- one rendering module per analysis module.

============================  ===================================  =============================
Viz module                    Domain                               Paired module
============================  ===================================  =============================
``graphics``                  Style, palettes, 2D statistics       ``statistics.analysis``
``bayes``                     Posterior / forest / ROPE / GP       ``statistics.bayesian``
``clusters``                  Cluster maps (3D) + 2D diagnostics,  ``statistics.clustering``
                              parcellation-vs-cluster grids
``spectral``                  Eigenmode panels, spectral           ``spectral.lbo``
                              deformation maps
``hipp``                      Hippocampal surfaces + flatmaps,     ``statistics.normative``
                              template-free six-view
``render3d``                  3D engine: cortex / subcortex,       (shared by all of the above)
                              tracts, generic meshes, six-view
============================  ===================================  =============================
"""

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
from spectralbrain.viz.clusters import (  # noqa: F401
    plot_parcellation_cluster_grid,
    plot_parcellation_vs_clusters,
)
from spectralbrain.viz.graphics import (  # noqa: F401
    CMAP_DIVERGING,
    CMAP_QUALITATIVE,
    CMAP_SEQUENTIAL,
    CMAP_SPECTRAL,
    COLOR_CONTROL,
    COLOR_PATIENT,
    DPI,
    PALETTE,
    PALETTE_LIST,
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
from spectralbrain.viz.hipp import (  # noqa: F401
    DENSITIES,
    HIPP_DESCRIPTOR_STYLES,
    HIPP_LABELS,
    HIPP_VIEWS_3D,
    HIPP_VIEWS_FULL,
    plot_hippocampus,
    plot_hippocampus_bilateral,
    plot_hippocampus_comparison,
    plot_hippocampus_gallery,
    plot_hippocampus_hovmoller,
    plot_hippocampus_normative,
    plot_hippocampus_sixview,
    plot_hippocampus_spatiotemporal,
)
from spectralbrain.viz.render3d import (  # noqa: F401
    CAMERA_PRESETS,
    CURVATURE_METHODS,
    DESCRIPTOR_STYLES,
    SIXVIEWS,
    TRACT_VIEWS,
    VIEWS_CORTEX,
    VIEWS_FULL,
    VIEWS_MEDIAL,
    BrainPlotSpec,
    add_glass_brain,
    compose_tract_panel,
    get_cmap,
    load_streamlines,
    mask_to_mesh,
    plot_bilateral_comparison,
    plot_brain,
    plot_brain_subcortical,
    plot_brain_tracts,
    plot_clustering_map,
    plot_curvature,
    plot_group_comparison,
    plot_mesh,
    plot_mesh_comparison,
    plot_mesh_pyvista,
    plot_morphometric_gallery,
    plot_multi_descriptor_panel,
    plot_multi_view,
    plot_normative_map,
    plot_scalar_difference,
    plot_spectral_progression,
    plot_surface_sixview,
    plot_top10_morphometrics,
    plot_wireframe,
    render_bundle_surface,
    render_streamlines,
    resolve_scalar_clim,
    robust_clim,
    spectral_overlay,
    streamlines_multiview,
)
from spectralbrain.viz.spectral import (  # noqa: F401
    assemble_eigenmode_panel,
    plot_eigenmode_panel,
    render_eigenmodes,
    render_scale_on_mesh,
    select_mode_indices,
    standardize_sign,
)
