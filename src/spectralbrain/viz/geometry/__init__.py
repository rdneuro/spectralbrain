"""Geometry visualization subpackage — point clouds and meshes.

This subpackage provides 3D rendering functions for the two core
geometric representations in SpectralBrain:

- **Point clouds** (``points`` module): atlas-free 3D scatter,
  MLS reconstruction, clustering, warping, Voronoi diagrams.
- **Meshes** (``meshes`` module): surface rendering, wireframe,
  curvature maps, multi-view panels, difference maps.

Both modules use vedo (VTK-based) as the primary renderer with
optional fallbacks to Open3D (points) and PyVista (meshes).

Typical usage::

    from spectralbrain.viz.geometry import (
        plot_point_cloud, plot_clusters, plot_mesh, plot_curvature,
    )
"""

from spectralbrain.viz.geometry.points import (
    plot_point_cloud,
    plot_mls_reconstruction,
    plot_clusters,
    plot_point_cloud_panel,
    plot_warp,
    plot_voronoi,
    plot_point_cloud_o3d,
)

from spectralbrain.viz.geometry.meshes import (
    CURVATURE_METHODS,
    CAMERA_PRESETS,
    plot_mesh,
    plot_wireframe,
    plot_curvature,
    plot_multi_view,
    plot_mesh_comparison,
    plot_scalar_difference,
    plot_mesh_pyvista,
)

__all__ = [
    # Points
    "plot_point_cloud",
    "plot_mls_reconstruction",
    "plot_clusters",
    "plot_point_cloud_panel",
    "plot_warp",
    "plot_voronoi",
    "plot_point_cloud_o3d",
    # Meshes
    "CURVATURE_METHODS",
    "CAMERA_PRESETS",
    "plot_mesh",
    "plot_wireframe",
    "plot_curvature",
    "plot_multi_view",
    "plot_mesh_comparison",
    "plot_scalar_difference",
    "plot_mesh_pyvista",
]
