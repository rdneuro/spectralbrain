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

from spectralbrain.viz.geometry.meshes import (
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
from spectralbrain.viz.geometry.points import (
    plot_clusters,
    plot_mls_reconstruction,
    plot_point_cloud,
    plot_point_cloud_o3d,
    plot_point_cloud_panel,
    plot_voronoi,
    plot_warp,
)

__all__ = [
    "CAMERA_PRESETS",
    # Meshes
    "CURVATURE_METHODS",
    "plot_clusters",
    "plot_curvature",
    "plot_mesh",
    "plot_mesh_comparison",
    "plot_mesh_pyvista",
    "plot_mls_reconstruction",
    "plot_multi_view",
    # Points
    "plot_point_cloud",
    "plot_point_cloud_o3d",
    "plot_point_cloud_panel",
    "plot_scalar_difference",
    "plot_voronoi",
    "plot_warp",
    "plot_wireframe",
]
