"""Container-based DL preprocessing for raw anatomical images.

This module provides high-level functions that orchestrate
Singularity/Apptainer containers for skull-stripping, tissue
segmentation, and structure extraction.  **No DL dependencies are
installed on the host** -- each tool runs inside its own immutable
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
from typing import Any

import numpy as np

from spectralbrain.runtime import (
    ContainerManager,
    PathLike,
    get_logger,
)

logger = get_logger(__name__)

# Module-level container manager (lazy singleton).
_manager: ContainerManager | None = None


def _get_manager() -> ContainerManager:
    """Return (and cache) a module-level ContainerManager."""
    global _manager
    if _manager is None:
        _manager = ContainerManager()
    return _manager


# ======================================================================
# S1  HIGH-LEVEL PREPROCESSING FUNCTIONS
# ======================================================================


def skull_strip(
    input_path: PathLike,
    output_path: PathLike | None = None,
    *,
    gpu: bool | None = None,
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
    logger.info("Skull-stripped -> %s", out)
    return out


def segment(
    input_path: PathLike,
    output_path: PathLike | None = None,
    *,
    gpu: bool | None = None,
) -> Path:
    """Segment a T1w (or T2w/FLAIR) image using SynthSeg.

    Produces a volumetric label map compatible with FreeSurfer's
    ``aseg`` conventions.  Works on **any MRI contrast** -- SynthSeg
    is contrast-agnostic.

    Parameters
    ----------
    input_path : PathLike
        Anatomical NIfTI (skull-stripped or not -- SynthSeg handles both).
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
    logger.info("Segmented -> %s", out)
    return out


def run_fastsurfer(
    input_path: PathLike,
    output_dir: PathLike | None = None,
    *,
    gpu: bool | None = None,
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
    logger.info("FastSurfer output -> %s", out)
    return out


# ======================================================================
# S2  PIPELINE: RAW -> GEOMETRY
# ======================================================================


def raw_to_mesh(
    input_path: PathLike,
    label_id: int | list[int],
    *,
    output_dir: PathLike | None = None,
    gpu: bool | None = None,
    raw: bool = False,
    closed: bool = True,
    **mesh_kwargs: Any,
) -> Any:
    """End-to-end: raw T1w -> skull-strip -> segment -> LBO-ready mesh.

    Chains :func:`skull_strip`, :func:`segment` and
    :func:`spectralbrain.io.meshing.volume_to_mesh` into a single call.

    Parameters
    ----------
    input_path : PathLike
        Raw T1w NIfTI.
    label_id : int or list of int
        Target structure label(s) (e.g. 17 for the left hippocampus).
    output_dir : PathLike, optional
        Working directory for intermediate files.
    gpu : bool or None
        GPU passthrough.
    raw : bool
        If ``True``, plain marching cubes (legacy path); otherwise the
        improved, LBO-ready surface (default).
    closed : bool
        Treat the structure as a closed 2-manifold (see
        :func:`spectralbrain.io.meshing.refine_mesh`).
    **mesh_kwargs
        Forwarded to :func:`spectralbrain.io.meshing.volume_to_mesh`.

    Returns
    -------
    BrainMesh
        World-space surface of the target structure, with the meshing
        report and the source paths in ``metadata``.

    Examples
    --------
    >>> hippo = raw_to_mesh("sub-01_T1w.nii.gz", label_id=17)
    >>> decomp = hippo.decompose(k=100)
    """
    from spectralbrain.core.meshes import BrainMesh
    from spectralbrain.io.loaders import load_nifti
    from spectralbrain.io.meshing import volume_to_mesh

    inp = Path(input_path)
    wdir = Path(output_dir) if output_dir else inp.parent
    wdir.mkdir(parents=True, exist_ok=True)

    stem = inp.name.split(".")[0]  # "sub-01_T1w.nii.gz" -> "sub-01_T1w"

    # Step 1: skull-strip
    brain = skull_strip(inp, wdir / f"{stem}_brain.nii.gz", gpu=gpu)

    # Step 2: segment
    seg = segment(brain, wdir / f"{stem}_synthseg.nii.gz", gpu=gpu)

    # Step 3: label volume -> mesh
    data, affine = load_nifti(seg)
    verts, faces, info = volume_to_mesh(
        np.asarray(data),
        affine,
        label=label_id,
        raw=raw,
        closed=closed,
        return_info=True,
        **mesh_kwargs,
    )
    meta = {**info, "source": str(inp), "segmentation": str(seg), "label_id": label_id}
    return BrainMesh(verts, faces, metadata=meta)


# ======================================================================
# S3  CONTAINER STATUS / MANAGEMENT
# ======================================================================


def status() -> None:
    """Print the status of all preprocessing containers."""
    _get_manager().status()


def clean(tool: str | None = None) -> None:
    """Remove cached container(s).

    Parameters
    ----------
    tool : str or None
        Specific tool (e.g. ``"hdbet"``), or ``None`` for all.
    """
    _get_manager().clean(tool)


# ======================================================================

__all__ = [
    "clean",
    "raw_to_mesh",
    "run_fastsurfer",
    "segment",
    "skull_strip",
    "status",
]
