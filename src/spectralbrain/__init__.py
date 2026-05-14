"""SpectralBrain — spectral shape analysis for brain structures.

SpectralBrain provides spectral shape descriptors (ShapeDNA, HKS, WKS,
GPS, BKS, spectral graph wavelets) for brain surfaces and point clouds.
It supports multiple input modalities (T1, FreeSurfer, HippUnfold,
TractSeg), Bayesian statistical models, and publication-quality
visualizations.

Quick start::

    import spectralbrain as sb

    # Load a FreeSurfer surface
    vertices, faces = sb.load_freesurfer_surface("lh.pial")

    # Build a mesh object and compute eigenpairs
    mesh = sb.BrainMesh(vertices, faces)
    mesh.compute_eigenpairs(k=300)

    # Compute spectral descriptors
    hks = sb.compute_hks(mesh.eigenvalues, mesh.eigenvectors,
                         t_values=[1, 10, 100])

    # Visualize on a brain surface
    sb.plot_brain(data=hks[:, 0], atlas="schaefer_400")
"""

from __future__ import annotations

# Version: set dynamically by hatch-vcs from git tags,
# or fall back to a static string for editable installs.
try:
    from spectralbrain._version import __version__
except ImportError:
    __version__ = "0.0.2"

# ── Runtime configuration ──
from spectralbrain.runtime import (  # noqa: F401
    get_logger, set_log_level,
    GeometryFormat, AtlasScheme, DescriptorType,
    AnalysisObjective, BackendName,
    ContainerSpec, ContainerManager,
)

# ── Core geometric objects ──
from spectralbrain.core import (  # noqa: F401
    SpectralDecomposition,
    BrainMesh,
    BrainPointCloud,
    compute_centroid, compute_bounding_box, compute_pca_axes,
    center_points, normalize_scale, align_to_pca,
    procrustes_align, farthest_point_sampling,
    knn_search, radius_search,
    hausdorff_distance, chamfer_distance,
)

# ── I/O ──
from spectralbrain.io import (  # noqa: F401
    detect_format, load,
    load_freesurfer_surface, load_freesurfer_annot, load_freesurfer_morph,
    load_gifti_surface, load_gifti_func, load_gifti_label,
    load_nifti, load_mesh,
    labels_to_pointcloud, extract_submesh, apply_parcellation,
    save_hdf5, load_hdf5, save_mesh, save_gifti_func,
    save_npz, save_connectome,
    # Parcellation pipeline
    parcellate, parcellate_batch, list_atlases,
    ParcellationResult, ATLAS_REGISTRY,
)

# ── Spectral analysis (the core value of the library) ──
from spectralbrain.spectral import (  # noqa: F401
    compute_shapedna, compute_hks, compute_si_hks,
    compute_wks, compute_gps,
    compute_bates_signatures, compute_bks, compute_ibks,
    compute_all_descriptors,
    wesd, wesd_matrix, shapedna_distance,
    biharmonic_distance, commute_time_distance,
    diffusion_distance, descriptor_distance,
    build_geometric_connectome,
    sgw_transform, sgw_descriptor,
    anisotropic_laplacian,
    compute_functional_map, shape_difference_operator,
)

# ── Backends (lazy — heavy imports deferred) ──
from spectralbrain.backends import NumpyBackend  # noqa: F401

# ── Utilities ──
from spectralbrain.utils import (  # noqa: F401
    ASEG_LABELS, HIPPOCAMPAL_SUBFIELDS, THALAMIC_NUCLEI,
    get_label_name, get_label_id, list_labels,
    seed_everything, get_reproducibility_info,
    timer, Timer,
    parse_bids_filename, collect_subjects,
)

__all__ = [
    "__version__",
    # Runtime
    "get_logger", "set_log_level",
    "GeometryFormat", "AtlasScheme", "DescriptorType",
    "AnalysisObjective", "BackendName",
    # Core
    "SpectralDecomposition", "BrainMesh", "BrainPointCloud",
    "compute_centroid", "compute_bounding_box", "compute_pca_axes",
    "center_points", "normalize_scale", "align_to_pca",
    "procrustes_align", "farthest_point_sampling",
    "knn_search", "radius_search",
    "hausdorff_distance", "chamfer_distance",
    # I/O
    "detect_format", "load",
    "load_freesurfer_surface", "load_freesurfer_annot", "load_freesurfer_morph",
    "load_gifti_surface", "load_gifti_func", "load_gifti_label",
    "load_nifti", "load_mesh",
    "labels_to_pointcloud", "extract_submesh", "apply_parcellation",
    "save_hdf5", "load_hdf5", "save_mesh", "save_gifti_func",
    "save_npz", "save_connectome",
    # Parcellation
    "parcellate", "parcellate_batch", "list_atlases",
    "ParcellationResult", "ATLAS_REGISTRY",
    # Spectral
    "compute_shapedna", "compute_hks", "compute_si_hks",
    "compute_wks", "compute_gps",
    "compute_bates_signatures", "compute_bks", "compute_ibks",
    "compute_all_descriptors",
    "wesd", "wesd_matrix", "shapedna_distance",
    "biharmonic_distance", "commute_time_distance",
    "diffusion_distance", "descriptor_distance",
    "build_geometric_connectome",
    "sgw_transform", "sgw_descriptor",
    "anisotropic_laplacian",
    "compute_functional_map", "shape_difference_operator",
    # Backends
    "NumpyBackend",
    # Utils
    "ASEG_LABELS", "HIPPOCAMPAL_SUBFIELDS", "THALAMIC_NUCLEI",
    "get_label_name", "get_label_id", "list_labels",
    "seed_everything", "get_reproducibility_info",
    "timer", "Timer",
    "parse_bids_filename", "collect_subjects",
]
