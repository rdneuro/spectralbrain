"""SpectralBrain core -- geometric objects, base ops and compute backends.

- :mod:`~spectralbrain.core.base` -- :class:`SpectralDecomposition` (the
  central handoff object), the :class:`GeometricObject` protocol and shared
  vertex-array geometry.
- :mod:`~spectralbrain.core.meshes` -- :class:`BrainMesh`, the triangle-mesh
  geometry SpectralBrain is built around.
- :mod:`~spectralbrain.core.backends` -- NumPy / CuPy / JAX / Torch compute
  backends with a common ``eigsh`` interface.
"""

from spectralbrain.core.backends import (  # noqa: F401
    CupyBackend,
    JaxBackend,
    NumpyBackend,
    TorchBackend,
    get_gpu_backend,
)
from spectralbrain.core.base import (  # noqa: F401
    GeometricObject,
    SpectralDecomposition,
    align_to_pca,
    center_points,
    chamfer_distance,
    compute_bounding_box,
    compute_centroid,
    compute_pca_axes,
    convex_hull_area,
    convex_hull_volume,
    hausdorff_distance,
    knn_search,
    marching_cubes,
    normalize_scale,
    procrustes_align,
    radius_search,
)
from spectralbrain.core.meshes import BrainMesh, weld_mesh  # noqa: F401
