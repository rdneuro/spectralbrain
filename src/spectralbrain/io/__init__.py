"""SpectralBrain I/O — loaders, exporters, preprocessing, parcellation, and GPU pipeline."""

from spectralbrain.io.loaders import (  # noqa: F401
    detect_format, load,
    load_freesurfer_surface, load_freesurfer_annot, load_freesurfer_morph,
    load_gifti_surface, load_gifti_func, load_gifti_label,
    load_nifti, load_mesh,
    labels_to_pointcloud, extract_submesh, apply_parcellation,
)
from spectralbrain.io.export import (  # noqa: F401
    save_hdf5, load_hdf5, save_mesh, save_gifti_func,
    save_npz, save_connectome,
)
from spectralbrain.io.preprocess import (  # noqa: F401
    skull_strip as container_skull_strip,
    segment as container_segment,
    run_fastsurfer as container_run_fastsurfer,
    raw_to_pointcloud, status, clean,
)
from spectralbrain.io.parcellate import (  # noqa: F401
    AtlasSpec, ATLAS_REGISTRY, list_atlases,
    ParcellationResult,
    parcellate, parcellate_batch,
)
from spectralbrain.io.gpu_preprocess import (  # noqa: F401
    PreprocessResult,
    preprocess_gpu, preprocess_gpu_batch,
    enhance, enhance_bmex, enhance_deepn4,
    skull_strip, skull_strip_hdbet, skull_strip_synthstrip,
    register, register_synthmorph_affine, register_unigradicon,
    segment, segment_synthseg, segment_fastsurfer,
    parcellate_openmap, parcellate_brainparc,
    purge_vram, vram_info,
    discover_nifti, ensure_template,
)
