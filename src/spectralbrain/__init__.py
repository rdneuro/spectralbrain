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

    # Build a mesh and compute its Laplace-Beltrami decomposition
    mesh = sb.BrainMesh(vertices, faces)
    decomp = mesh.decompose(k=300)        # -> SpectralDecomposition

    # Compute spectral descriptors from the decomposition
    hks = sb.compute_hks(decomp, t_values=[1, 10, 100])
    wks = sb.compute_wks(decomp, n_energies=100)
    shapedna = sb.compute_shapedna(decomp)

    # Visualize on a brain surface
    sb.plot_brain(data=hks[:, 0], atlas="schaefer_400")
"""

from __future__ import annotations

# Version: set dynamically by hatch-vcs from git tags,
# or fall back to a static string for editable installs.
try:
    from spectralbrain._version import __version__
except ImportError:
    __version__ = "0.1.0"

# ── Runtime configuration ──
# ── Backends (lazy — heavy imports deferred) ──
from spectralbrain.backends import NumpyBackend

# ── Core geometric objects ──
from spectralbrain.core import (
    BrainMesh,
    BrainPointCloud,
    SpectralDecomposition,
    align_to_pca,
    center_points,
    chamfer_distance,
    compute_bounding_box,
    compute_centroid,
    compute_pca_axes,
    farthest_point_sampling,
    hausdorff_distance,
    knn_search,
    normalize_scale,
    procrustes_align,
    radius_search,
)

# ── I/O ──
from spectralbrain.io import (
    ATLAS_REGISTRY,
    DESIKAN_LOBE_MAP,
    SCHAEFER_NETWORK_MAP,
    GroupData,
    ParcellationResult,
    aggregate_by_parcellation,
    apply_parcellation,
    detect_format,
    discover_bids,
    discover_freesurfer,
    discover_tractseg_bundles,
    discover_tractseg_subjects,
    extract_submesh,
    group_comparison,
    labels_to_pointcloud,
    list_atlases,
    load,
    load_freesurfer_annot,
    load_freesurfer_morph,
    load_freesurfer_surface,
    load_gifti_func,
    load_gifti_label,
    load_gifti_surface,
    load_group,
    load_group_freesurfer,
    load_hdf5,
    load_mesh,
    load_nifti,
    load_tractseg,
    load_tractseg_bundle,
    # Parcellation pipeline
    parcellate,
    parcellate_batch,
    remap_parcellation,
    resample_to_template,
    save_connectome,
    save_gifti_func,
    save_hdf5,
    save_mesh,
    save_npz,
)
from spectralbrain.runtime import (  # noqa: F401
    AnalysisObjective,
    AtlasScheme,
    BackendName,
    ContainerManager,
    ContainerSpec,
    DescriptorType,
    GeometryFormat,
    get_logger,
    set_log_level,
)

# ── Spectral analysis (the core value of the library) ──
from spectralbrain.spectral import (
    anisotropic_laplacian,
    biharmonic_distance,
    build_geometric_connectome,
    commute_time_distance,
    compute_all_descriptors,
    compute_bates_signatures,
    compute_bks,
    compute_functional_map,
    compute_gps,
    compute_hks,
    compute_ibks,
    compute_shapedna,
    compute_si_hks,
    compute_wks,
    descriptor_distance,
    diffusion_distance,
    sgw_descriptor,
    sgw_transform,
    shape_difference_operator,
    shapedna_distance,
    wesd,
    wesd_matrix,
)

# ── Utilities ──
from spectralbrain.utils import (
    ASEG_LABELS,
    HIPPOCAMPAL_SUBFIELDS,
    THALAMIC_NUCLEI,
    Timer,
    collect_subjects,
    get_label_id,
    get_label_name,
    get_reproducibility_info,
    list_labels,
    parse_bids_filename,
    seed_everything,
    timer,
)

__all__ = [
    # Utils
    "ASEG_LABELS",
    "ATLAS_REGISTRY",
    "DESIKAN_LOBE_MAP",
    "HIPPOCAMPAL_SUBFIELDS",
    "SCHAEFER_NETWORK_MAP",
    "THALAMIC_NUCLEI",
    "AnalysisObjective",
    "AtlasScheme",
    "BackendName",
    "BrainMesh",
    "BrainPointCloud",
    "DescriptorType",
    "GeometryFormat",
    "GroupData",
    # Backends
    "NumpyBackend",
    "ParcellationResult",
    # Core
    "SpectralDecomposition",
    "Timer",
    "__version__",
    "aggregate_by_parcellation",
    "align_to_pca",
    "anisotropic_laplacian",
    "apply_parcellation",
    "biharmonic_distance",
    "build_geometric_connectome",
    "center_points",
    "chamfer_distance",
    "collect_subjects",
    "commute_time_distance",
    "compute_all_descriptors",
    "compute_bates_signatures",
    "compute_bks",
    "compute_bounding_box",
    "compute_centroid",
    "compute_functional_map",
    "compute_gps",
    "compute_hks",
    "compute_ibks",
    "compute_pca_axes",
    # Spectral
    "compute_shapedna",
    "compute_si_hks",
    "compute_wks",
    "descriptor_distance",
    # I/O
    "detect_format",
    "diffusion_distance",
    "discover_bids",
    "discover_freesurfer",
    "discover_tractseg_bundles",
    "discover_tractseg_subjects",
    "extract_submesh",
    "farthest_point_sampling",
    "get_label_id",
    "get_label_name",
    # Runtime
    "get_logger",
    "get_reproducibility_info",
    "group_comparison",
    "hausdorff_distance",
    "knn_search",
    "labels_to_pointcloud",
    "list_atlases",
    "list_labels",
    "load",
    "load_freesurfer_annot",
    "load_freesurfer_morph",
    "load_freesurfer_surface",
    "load_gifti_func",
    "load_gifti_label",
    "load_gifti_surface",
    "load_group",
    "load_group_freesurfer",
    "load_hdf5",
    "load_mesh",
    "load_nifti",
    "load_tractseg",
    "load_tractseg_bundle",
    "normalize_scale",
    # Parcellation
    "parcellate",
    "parcellate_batch",
    "parse_bids_filename",
    "procrustes_align",
    "radius_search",
    "remap_parcellation",
    "resample_to_template",
    "save_connectome",
    "save_gifti_func",
    "save_hdf5",
    "save_mesh",
    "save_npz",
    "seed_everything",
    "set_log_level",
    "sgw_descriptor",
    "sgw_transform",
    "shape_difference_operator",
    "shapedna_distance",
    "timer",
    "wesd",
    "wesd_matrix",
]
