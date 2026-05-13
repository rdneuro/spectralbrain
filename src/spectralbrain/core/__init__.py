"""SpectralBrain core geometric objects — base ops, meshes, point clouds."""

from spectralbrain.core.base import (  # noqa: F401
    SpectralDecomposition,
    compute_centroid, compute_bounding_box, compute_pca_axes,
    center_points, normalize_scale, align_to_pca,
    procrustes_align, farthest_point_sampling,
    knn_search, radius_search, compute_adjacency_from_knn,
    hausdorff_distance, chamfer_distance,
    marching_cubes, convex_hull_volume, convex_hull_area,
    estimate_point_density, detect_density_outliers,
)
from spectralbrain.core.meshes import BrainMesh  # noqa: F401
from spectralbrain.core.pointclouds import BrainPointCloud  # noqa: F401
