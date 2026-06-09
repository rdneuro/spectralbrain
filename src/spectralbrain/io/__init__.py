"""SpectralBrain I/O — loaders, exporters, preprocessing, parcellation, and GPU pipeline."""

from spectralbrain.io.export import (  # noqa: F401
    load_hdf5,
    save_connectome,
    save_gifti_func,
    save_hdf5,
    save_mesh,
    save_npz,
)
from spectralbrain.io.gpu_preprocess import (  # noqa: F401
    PreprocessResult,
    discover_nifti,
    enhance,
    enhance_bmex,
    enhance_deepn4,
    ensure_template,
    parcellate_brainparc,
    parcellate_openmap,
    preprocess_gpu,
    preprocess_gpu_batch,
    purge_vram,
    register,
    register_synthmorph_affine,
    register_unigradicon,
    segment,
    segment_fastsurfer,
    segment_synthseg,
    skull_strip,
    skull_strip_hdbet,
    skull_strip_synthstrip,
    vram_info,
)
from spectralbrain.io.group import (  # noqa: F401
    GroupData,
    discover_bids,
    discover_freesurfer,
    group_comparison,
    load_group,
    load_group_freesurfer,
    resample_to_template,
)
from spectralbrain.io.loaders import (  # noqa: F401
    DESIKAN_LOBE_MAP,
    SCHAEFER_NETWORK_MAP,
    aggregate_by_parcellation,
    apply_parcellation,
    detect_format,
    extract_submesh,
    labels_to_pointcloud,
    load,
    load_freesurfer_annot,
    load_freesurfer_morph,
    load_freesurfer_surface,
    load_gifti_func,
    load_gifti_label,
    load_gifti_surface,
    load_mesh,
    load_nifti,
    remap_parcellation,
)
from spectralbrain.io.parcellate import (  # noqa: F401
    ATLAS_REGISTRY,
    AtlasSpec,
    ParcellationResult,
    list_atlases,
    parcellate,
    parcellate_batch,
)
from spectralbrain.io.preprocess import (  # noqa: F401
    clean,
    raw_to_pointcloud,
    status,
)
from spectralbrain.io.preprocess import (  # noqa: F401
    run_fastsurfer as container_run_fastsurfer,
)
from spectralbrain.io.preprocess import (  # noqa: F401
    segment as container_segment,
)
from spectralbrain.io.preprocess import (  # noqa: F401
    skull_strip as container_skull_strip,
)
from spectralbrain.io.tractseg import (  # noqa: F401
    discover_tractseg_bundles,
    discover_tractseg_subjects,
    load_tractseg,
    load_tractseg_bundle,
)
