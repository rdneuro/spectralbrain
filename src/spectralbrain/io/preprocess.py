"""Container-based DL preprocessing for raw anatomical images.

This module provides high-level functions that orchestrate
Singularity/Apptainer containers for skull-stripping, tissue
segmentation, and structure extraction.  **No DL dependencies are
installed on the host** — each tool runs inside its own immutable
``.sif`` container, downloaded once on first use.

All functions delegate to :class:`spectralbrain.runtime.ContainerManager`.

.. note::
   This module requires Singularity or Apptainer to be installed on
   the system.  If neither is available, a clear error message is
   raised with installation instructions.

Examples
--------
>>> from spectralbrain.io.preprocess import skull_strip, segment
>>> brain_path = skull_strip("sub-01_T1w.nii.gz")
>>> seg_path = segment(brain_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from spectralbrain.runtime import (
    ContainerManager,
    PathLike,
    get_logger,
)

logger = get_logger(__name__)

# Module-level container manager (lazy singleton).
_manager: Optional[ContainerManager] = None


def _get_manager() -> ContainerManager:
    """Return (and cache) a module-level ContainerManager."""
    global _manager
    if _manager is None:
        _manager = ContainerManager()
    return _manager


# ======================================================================
# §1  HIGH-LEVEL PREPROCESSING FUNCTIONS
# ======================================================================

def skull_strip(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    gpu: Optional[bool] = None,
) -> Path:
    """Skull-strip a T1w image using HD-BET.

    Parameters
    ----------
    input_path : PathLike
        Raw T1-weighted NIfTI file.
    output_path : PathLike, optional
        Output brain-extracted NIfTI.  Defaults to
        ``<input_stem>_brain.nii.gz`` in the same directory.
    gpu : bool or None
        Force GPU on/off.  ``None`` = auto-detect.

    Returns
    -------
    Path
        Path to the brain-extracted image.

    Raises
    ------
    EnvironmentError
        If no container runtime is available.
    RuntimeError
        If the container exits with an error.

    Examples
    --------
    >>> brain = skull_strip("sub-01_T1w.nii.gz")
    >>> brain
    PosixPath('sub-01_T1w_brain.nii.gz')
    """
    inp = Path(input_path)
    if output_path is None:
        output_path = inp.parent / f"{inp.name.split('.')[0]}_brain.nii.gz"
    out = Path(output_path)

    cm = _get_manager()
    cm.run("hdbet", input_path=inp, output_path=out, gpu=gpu)
    logger.info("Skull-stripped → %s", out)
    return out


def segment(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    gpu: Optional[bool] = None,
) -> Path:
    """Segment a T1w (or T2w/FLAIR) image using SynthSeg.

    Produces a volumetric label map compatible with FreeSurfer's
    ``aseg`` conventions.  Works on **any MRI contrast** — SynthSeg
    is contrast-agnostic.

    Parameters
    ----------
    input_path : PathLike
        Anatomical NIfTI (skull-stripped or not — SynthSeg handles both).
    output_path : PathLike, optional
        Output segmentation NIfTI.  Defaults to
        ``<input_stem>_synthseg.nii.gz``.
    gpu : bool or None
        Force GPU on/off.

    Returns
    -------
    Path
        Path to the segmentation volume.

    Examples
    --------
    >>> seg = segment("sub-01_T1w_brain.nii.gz")
    >>> data, affine = sb.io.load_nifti(seg)
    >>> np.unique(data)  # FreeSurfer aseg labels
    """
    inp = Path(input_path)
    if output_path is None:
        output_path = inp.parent / f"{inp.name.split('.')[0]}_synthseg.nii.gz"
    out = Path(output_path)

    cm = _get_manager()
    cm.run("synthseg", input_path=inp, output_path=out, gpu=gpu)
    logger.info("Segmented → %s", out)
    return out


def run_fastsurfer(
    input_path: PathLike,
    output_dir: Optional[PathLike] = None,
    *,
    gpu: Optional[bool] = None,
) -> Path:
    """Run FastSurfer segmentation (seg_only mode).

    Produces a FreeSurfer-compatible segmentation and cortical
    parcellation in a ``$SUBJECTS_DIR``-like directory structure.

    Parameters
    ----------
    input_path : PathLike
        T1-weighted NIfTI.
    output_dir : PathLike, optional
        FastSurfer output directory.  Defaults to
        ``<input_dir>/fastsurfer/``.
    gpu : bool or None
        Force GPU on/off.

    Returns
    -------
    Path
        Path to the output directory.
    """
    inp = Path(input_path)
    if output_dir is None:
        output_dir = inp.parent / "fastsurfer"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cm = _get_manager()
    cm.run("fastsurfer", input_path=inp, output_path=out, gpu=gpu)
    logger.info("FastSurfer output → %s", out)
    return out


# ======================================================================
# §2  PIPELINE: RAW → GEOMETRY
# ======================================================================

def raw_to_pointcloud(
    input_path: PathLike,
    label_id: int,
    *,
    output_dir: Optional[PathLike] = None,
    gpu: Optional[bool] = None,
    jitter: bool = True,
    jitter_scale: float = 0.25,
    seed: Optional[int] = None,
) -> "np.ndarray":
    """End-to-end: raw T1w → skull-strip → segment → point cloud.

    Chains :func:`skull_strip`, :func:`segment`, and
    :func:`spectralbrain.io.labels_to_pointcloud` into a single call.

    Parameters
    ----------
    input_path : PathLike
        Raw T1w NIfTI.
    label_id : int
        Target structure label (e.g. 17 for left hippocampus).
    output_dir : PathLike, optional
        Working directory for intermediate files.
    gpu : bool or None
        GPU passthrough.
    jitter : bool
        Add sub-voxel jitter to the point cloud.
    jitter_scale : float
        Jitter magnitude in voxel units.
    seed : int, optional
        RNG seed.

    Returns
    -------
    points : ndarray, shape (N, 3)
        World-space point cloud for the target structure.

    Examples
    --------
    >>> hippo = raw_to_pointcloud("sub-01_T1w.nii.gz", label_id=17)
    >>> hippo.shape
    (4231, 3)
    """
    import numpy as np
    from spectralbrain.io.loaders import labels_to_pointcloud, load_nifti

    inp = Path(input_path)
    wdir = Path(output_dir) if output_dir else inp.parent
    wdir.mkdir(parents=True, exist_ok=True)

    # Step 1: skull-strip
    brain = skull_strip(inp, wdir / f"{inp.stem}_brain.nii.gz", gpu=gpu)

    # Step 2: segment
    seg = segment(brain, wdir / f"{inp.stem}_synthseg.nii.gz", gpu=gpu)

    # Step 3: extract point cloud
    data, affine = load_nifti(seg)
    points = labels_to_pointcloud(
        data, affine, label_id,
        jitter=jitter, jitter_scale=jitter_scale, seed=seed,
    )
    return points


# ======================================================================
# §3  CONTAINER STATUS / MANAGEMENT
# ======================================================================

def status() -> None:
    """Print the status of all preprocessing containers."""
    _get_manager().status()


def clean(tool: Optional[str] = None) -> None:
    """Remove cached container(s).

    Parameters
    ----------
    tool : str or None
        Specific tool (e.g. ``"hdbet"``), or ``None`` for all.
    """
    _get_manager().clean(tool)


# ======================================================================

__all__ = [
    "skull_strip",
    "segment",
    "run_fastsurfer",
    "raw_to_pointcloud",
    "status",
    "clean",
]
