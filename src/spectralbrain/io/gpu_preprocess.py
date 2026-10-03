"""GPU-native preprocessing pipeline for brain MRI.

This module replaces the traditional CPU-bound FreeSurfer/ANTs chain
with a fully GPU-accelerated PyTorch pipeline.  Each step runs as
a separate model load → inference → VRAM purge cycle to keep peak
memory within consumer GPU limits (24 GB).

Pipeline stages
~~~~~~~~~~~~~~~

1. **Enhancement** — BME-X (Sun et al., 2025, Nat Biomed Eng) or
   DeepN4 (Kanakaraj et al., 2024) for bias field correction,
   denoising, and optional super-resolution.
2. **Skull stripping** — HD-BET (Isensee et al., 2019) or
   SynthStrip (Hoopes et al., 2022) for brain extraction.
3. **Registration** — SynthMorph affine (Hoffmann et al., 2024) +
   uniGradICON deformable (Tian et al., 2024) for MNI
   normalization.
4. **Tissue segmentation** — SynthSeg+ (Billot et al., 2023) for
   GM/WM/CSF + DKT cortical parcellation in a single pass.
5. **Parcellation** — OpenMAP-T1 (Nishimaki et al., 2024) for 280
   regions or BrainParc (Liu et al., 2026) for 106 lifespan
   regions.

Usage
-----
>>> from spectralbrain.io.gpu_preprocess import preprocess_gpu
>>> result = preprocess_gpu(
...     "sub-01_T1w.nii.gz",
...     output_dir="output/",
...     steps=["enhance", "skull_strip", "register", "segment"],
... )
>>> result.brain_path
PosixPath('output/sub-01_T1w_brain.nii.gz')
>>> result.segmentation_path
PosixPath('output/sub-01_T1w_synthseg.nii.gz')

Notes
-----
Based on the gpu_preproc.py pipeline by Rodrigo Dalvit (rdneuro),
refactored as a SpectralBrain library module with direct Python API
integration where possible and subprocess fallback where required.
"""

from __future__ import annotations

import gc
import inspect
import os
import re
import shlex
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)


# ======================================================================
# §0  Constants and configuration
# ======================================================================

NIFTI_EXTS = (".nii.gz", ".nii")

# BIDS-style sequence detection
BIDS_SEQ_RE = re.compile(r"_(T1w|T2w|FLAIR|PD|PDw)[\._]", re.IGNORECASE)

# TemplateFlow S3 for MNI152NLin2009cAsym 2mm
TEMPLATE_URLS = {
    "t1": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
    ),
    "t2": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_res-02_T2w.nii.gz"
    ),
}

TEMPLATE_CACHE = Path.home() / ".cache" / "spectralbrain" / "templates"


class Step(Enum):
    """Available preprocessing steps."""

    ENHANCE = auto()
    SKULL_STRIP = auto()
    REGISTER = auto()
    SEGMENT = auto()
    PARCELLATE = auto()


# Step name → Step enum mapping for string-based API
_STEP_MAP = {
    "enhance": Step.ENHANCE,
    "skull_strip": Step.SKULL_STRIP,
    "register": Step.REGISTER,
    "segment": Step.SEGMENT,
    "parcellate": Step.PARCELLATE,
}

# Default pipeline order
DEFAULT_STEPS = [
    Step.ENHANCE,
    Step.SKULL_STRIP,
    Step.REGISTER,
    Step.SEGMENT,
]


# ======================================================================
# §1  VRAM management — mirrors gpu_preproc.py + backends/gpu.py
# ======================================================================


def _has_cuda() -> bool:
    """Check CUDA availability without importing torch at module load."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def purge_vram() -> None:
    """Aggressively release all CUDA memory held by this process.

    Runs two garbage-collection passes to break reference cycles,
    empties the PyTorch caching allocator, and collects IPC handles.
    Call between GPU-heavy pipeline steps so each model gets full
    VRAM access.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    gc.collect()


def vram_info() -> dict[str, float]:
    """Return current VRAM usage in GB.

    Returns
    -------
    dict
        Keys: ``'allocated'``, ``'reserved'``, ``'total'``, ``'free'``.
        All values in GB.  Returns zeros if CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}
        alloc = torch.cuda.memory_allocated() / (1024**3)
        resrv = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            "allocated": round(alloc, 2),
            "reserved": round(resrv, 2),
            "total": round(total, 1),
            "free": round(total - alloc, 2),
        }
    except ImportError:
        return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}


# ======================================================================
# §2  Template management
# ======================================================================


def _detect_sequence(filepath: Path, fallback: str = "t1") -> str:
    """Detect MRI sequence from BIDS-style filename."""
    m = BIDS_SEQ_RE.search(filepath.name)
    if m:
        raw = m.group(1).lower()
        return {"t1w": "t1", "t2w": "t2", "flair": "t2", "pd": "t1", "pdw": "t1"}.get(raw, fallback)
    return fallback


def ensure_template(
    seq: str = "t1",
    user_template: PathLike | None = None,
) -> Path:
    """Resolve or download the MNI registration template.

    Parameters
    ----------
    seq : str
        MRI sequence type (``'t1'``, ``'t2'``).
    user_template : path or None
        Explicit template path; skips download if provided.

    Returns
    -------
    Path
        Absolute path to the template NIfTI.
    """
    if user_template is not None:
        tp = Path(user_template).resolve()
        if not tp.exists():
            raise FileNotFoundError(f"Template not found: {tp}")
        return tp

    url = TEMPLATE_URLS.get(seq, TEMPLATE_URLS["t1"])
    TEMPLATE_CACHE.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    local = TEMPLATE_CACHE / filename

    if local.exists():
        logger.info("Template cached: %s", local.name)
        return local

    logger.info("Downloading template: %s", filename)
    urllib.request.urlretrieve(url, str(local))
    logger.info("Template saved: %s", local)
    return local


def _q(path: PathLike) -> str:
    """Shell-quote a path for the ``shell=True`` command strings."""
    return shlex.quote(str(path))


def _route_kwargs(func, kwargs: dict, label: str) -> dict:
    """Keep only the keyword arguments *func* accepts (log the rest).

    The ``auto``/dispatch wrappers receive one set of options for several
    backends with different signatures (e.g. ``device`` is an HD-BET option,
    ``mask_path`` a SynthStrip one); forwarding everything would raise
    ``TypeError`` in the backend that does not take it.
    """
    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        logger.debug("%s: ignoring options not used by this backend: %s", label, dropped)
    return kept


def _snapshot(out_dir: Path) -> dict[Path, float]:
    """Map of existing files → mtime (to detect outputs of a later run)."""
    return {f: f.stat().st_mtime for f in out_dir.rglob("*") if f.is_file()}


def _pick_new_output(
    out_dir: Path,
    before: dict[Path, float],
    patterns: tuple[str, ...],
    label: str,
) -> Path:
    """Return the single NIfTI written/updated by the last run in *out_dir*.

    Candidates are files created or modified since *before*, tried against
    *patterns* in order (most specific first). Zero or several matches raise
    instead of silently returning an arbitrary (possibly stale) file.
    """
    fresh = sorted(
        f for f in out_dir.rglob("*.nii*") if f.is_file() and before.get(f) != f.stat().st_mtime
    )
    if not fresh:
        raise FileNotFoundError(f"{label} produced no new NIfTI output in {out_dir}")
    for pat in patterns:
        hits = [f for f in fresh if f.match(pat)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError(
                f"{label}: ambiguous outputs for pattern {pat!r} in {out_dir}: "
                f"{[h.name for h in hits]}"
            )
    if len(fresh) == 1:
        return fresh[0]
    raise RuntimeError(
        f"{label}: cannot tell which output is the parcellation: {[f.name for f in fresh]}"
    )


# ======================================================================
# §3  Subprocess runner with VRAM isolation
# ======================================================================


def _run_cmd(
    cmd: str,
    label: str,
    *,
    env_extras: dict[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Execute a shell command with CUDA memory configuration.

    Sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` in the
    child environment to reduce fragmentation during iterative
    optimization.

    Parameters
    ----------
    cmd : str
        Shell command string.
    label : str
        Human-readable label for logging.
    env_extras : dict or None
        Additional environment variables.
    timeout : int
        Maximum seconds before killing the subprocess.

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    RuntimeError
        If the command exits with non-zero code.
    """
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if env_extras:
        env.update(env_extras)

    logger.info("[%s] $ %s", label, cmd)
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )

    if result.returncode != 0:
        # Log last 500 chars of output for debugging
        tail = (result.stdout + result.stderr)[-500:]
        logger.error("[%s] failed (exit %d): %s", label, result.returncode, tail)
        raise RuntimeError(f"{label} exited with code {result.returncode}.\nLast output: {tail}")

    logger.info("[%s] completed successfully", label)
    return result


# ======================================================================
# §4  Step 1 — Image enhancement (BME-X or DeepN4)
# ======================================================================


def enhance_bmex(
    input_path: PathLike,
    output_path: PathLike,
    *,
    age_group: str = "adult",
    mode: str = "enhance",
    overwrite: bool = False,
) -> Path:
    """Enhance a T1w image using BME-X (Sun et al., 2025).

    BME-X performs bias field correction, denoising, and optional
    super-resolution in a single forward pass using a tissue-aware
    foundation model.

    Parameters
    ----------
    input_path : PathLike
        Raw T1w NIfTI.
    output_path : PathLike
        Enhanced output NIfTI.
    age_group : str
        BME-X age model: ``'fetal'``, ``'infant'``, ``'adult'``
        (24+ months, valid through 100 years).
    mode : str
        Enhancement mode: ``'enhance'`` (bias+denoise),
        ``'super_resolution'``, ``'harmonize'``.

    Returns
    -------
    Path
        Path to the enhanced image.

    Notes
    -----
    Requires the ``brain-mri-enhancement`` package:
    ``pip install brain-mri-enhancement``
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        logger.info("BME-X output exists, skipping: %s", out.name)
        return out

    t0 = time.time()
    logger.info("BME-X enhancement (%s, %s): %s", age_group, mode, inp.name)

    try:
        # Direct Python API (BME-X PyTorch v1.0.2+)
        from brain_mri_enhancement import enhance as bmex_enhance

        bmex_enhance(
            input_path=str(inp),
            output_path=str(out),
            age_group=age_group,
            mode=mode,
        )
    except ImportError:
        # Fallback: Docker CLI
        logger.info("BME-X Python not found, trying Docker CLI")
        cmd = (
            f"docker run --gpus all --rm "
            f"-v {_q(f'{inp.parent}:/input')} -v {_q(f'{out.parent}:/output')} "
            f"yuesun814/bme-x:v1.0.5 "
            f"--input {_q('/input/' + inp.name)} "
            f"--output {_q('/output/' + out.name)} "
            f"--age_group {_q(age_group)} --mode {_q(mode)}"
        )
        _run_cmd(cmd, "BME-X (Docker)")

    purge_vram()
    elapsed = time.time() - t0
    logger.info("BME-X → %s (%.1fs)", out.name, elapsed)
    return out


def enhance_deepn4(
    input_path: PathLike,
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """GPU bias field correction via DeepN4 (DIPY).

    Includes the monkey-patch for DIPY's upstream CUDA bug where
    ``__predict`` calls ``.numpy()`` on a CUDA tensor without
    ``.cpu()`` first.

    Parameters
    ----------
    input_path : PathLike
        Raw T1w NIfTI.
    output_path : PathLike
        Bias-corrected output NIfTI.

    Returns
    -------
    Path
        Path to the corrected image.
    """
    import nibabel as nib
    import torch
    from dipy.nn.torch.deepn4 import DeepN4

    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        logger.info("DeepN4 output exists, skipping: %s", out.name)
        return out

    t0 = time.time()
    logger.info("DeepN4 bias correction: %s", inp.name)

    img = nib.load(str(inp))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    model = DeepN4(use_cuda=True)
    model.fetch_default_weights()

    # Monkey-patch for DIPY CUDA bug (missing .cpu() before .numpy())
    def _patched_predict(x_test):
        """Patched predict method for GPU-accelerated surface extraction."""
        with torch.no_grad():
            out_tensor = model.model(x_test)[:, 0].detach()
        return out_tensor.cpu().numpy() if out_tensor.is_cuda else out_tensor.numpy()

    model._DeepN4__predict = _patched_predict
    logger.debug("Patched DeepN4.__predict() for CUDA → cpu().numpy()")

    corrected = np.asarray(model.predict(data, affine), dtype=np.float32)
    # Store as float32: reusing the input header verbatim would keep e.g. a
    # uint8/int16 on-disk dtype and quantise the corrected intensities.
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(None, None)
    nib.save(nib.Nifti1Image(corrected, affine, hdr), str(out))

    # Free all GPU memory
    del data, corrected, img  # model freed on return
    purge_vram()

    elapsed = time.time() - t0
    logger.info("DeepN4 → %s (%.1fs)", out.name, elapsed)
    return out


def enhance(
    input_path: PathLike,
    output_path: PathLike,
    *,
    method: Literal["bmex", "deepn4", "auto"] = "auto",
    **kwargs,
) -> Path:
    """Enhance a T1w image using the best available method.

    Tries BME-X first (foundation model with bias+denoise+harmonize),
    falls back to DeepN4 (bias correction only).

    Parameters
    ----------
    input_path : PathLike
        Raw T1w NIfTI.
    output_path : PathLike
        Enhanced output.
    method : {'bmex', 'deepn4', 'auto'}
        ``'auto'`` tries BME-X, falls back to DeepN4.
    **kwargs
        Passed to the chosen method.

    Returns
    -------
    Path
        Path to the enhanced image.
    """
    return _enhance_impl(input_path, output_path, method=method, **kwargs)[0]


def _enhance_impl(
    input_path: PathLike,
    output_path: PathLike,
    *,
    method: str = "auto",
    **kwargs,
) -> tuple[Path, str]:
    """:func:`enhance` that also reports the backend actually used."""
    if method == "bmex":
        return enhance_bmex(
            input_path, output_path, **_route_kwargs(enhance_bmex, kwargs, "BME-X")
        ), "bmex"
    if method == "deepn4":
        return (
            enhance_deepn4(
                input_path, output_path, **_route_kwargs(enhance_deepn4, kwargs, "DeepN4")
            ),
            "deepn4",
        )
    if method != "auto":
        raise ValueError(f"Unknown enhance method {method!r}")

    # Auto: try BME-X first
    try:
        out = enhance_bmex(input_path, output_path, **_route_kwargs(enhance_bmex, kwargs, "BME-X"))
        return out, "bmex"
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        logger.warning("BME-X unavailable (%s), falling back to DeepN4", exc)
        out = enhance_deepn4(
            input_path, output_path, **_route_kwargs(enhance_deepn4, kwargs, "DeepN4")
        )
        return out, "deepn4"


# ======================================================================
# §5  Step 2 — Skull stripping (HD-BET or SynthStrip)
# ======================================================================


def skull_strip_hdbet(
    input_path: PathLike,
    output_path: PathLike,
    *,
    device: str = "cuda:0",
    save_mask: bool = True,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Brain extraction via HD-BET (Isensee et al., 2019).

    Parameters
    ----------
    input_path : PathLike
        Enhanced T1w NIfTI.
    output_path : PathLike
        Brain-extracted output NIfTI.
    device : str
        CUDA device string.
    save_mask : bool
        Whether to save the brain mask.

    Returns
    -------
    (Path, Path or None)
        Paths to the brain image and mask (if saved).
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        logger.info("HD-BET output exists, skipping: %s", out.name)
        mask = out.parent / out.name.replace(".nii.gz", "_mask.nii.gz")
        return out, mask if mask.exists() else None

    t0 = time.time()
    purge_vram()

    mask_flag = "--save_bet_mask" if save_mask else ""
    cmd = f"hd-bet -i {_q(inp)} -o {_q(out)} -device {_q(device)} {mask_flag}"
    _run_cmd(cmd, "HD-BET")

    if not out.exists():
        raise FileNotFoundError(f"HD-BET output missing: {out}")

    purge_vram()
    mask_path = out.parent / out.name.replace(".nii.gz", "_mask.nii.gz")
    elapsed = time.time() - t0
    logger.info("HD-BET → %s (%.1fs)", out.name, elapsed)
    return out, mask_path if mask_path.exists() else None


def skull_strip_synthstrip(
    input_path: PathLike,
    output_path: PathLike,
    mask_path: PathLike | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Brain extraction via SynthStrip (Hoopes et al., 2022).

    Contrast-agnostic skull stripping trained with domain
    randomization. Works on T1, T2, FLAIR, EPI, DWI, CT.

    Parameters
    ----------
    input_path : PathLike
        Input NIfTI (any contrast).
    output_path : PathLike
        Brain-extracted output.
    mask_path : PathLike or None
        Brain mask output path.

    Returns
    -------
    (Path, Path or None)
        Brain image and mask paths.
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        logger.info("SynthStrip output exists, skipping: %s", out.name)
        return out, Path(mask_path) if mask_path and Path(mask_path).exists() else None

    t0 = time.time()
    mask_flag = f"--mask {_q(mask_path)}" if mask_path else ""
    cmd = f"mri_synthstrip -i {_q(inp)} -o {_q(out)} {mask_flag} --gpu"
    _run_cmd(cmd, "SynthStrip")

    purge_vram()
    elapsed = time.time() - t0
    logger.info("SynthStrip → %s (%.1fs)", out.name, elapsed)
    return out, Path(mask_path) if mask_path and Path(mask_path).exists() else None


def skull_strip(
    input_path: PathLike,
    output_path: PathLike,
    *,
    method: Literal["hdbet", "synthstrip", "auto"] = "auto",
    **kwargs,
) -> tuple[Path, Path | None]:
    """Brain extraction using the best available method.

    Parameters
    ----------
    input_path : PathLike
        Enhanced or raw T1w NIfTI.
    output_path : PathLike
        Brain-extracted output.
    method : {'hdbet', 'synthstrip', 'auto'}
        ``'auto'`` tries HD-BET, falls back to SynthStrip.
    **kwargs
        Passed to the chosen method.

    Returns
    -------
    (Path, Path or None)
        Brain image and mask paths.
    """
    brain, mask, _method = _skull_strip_impl(input_path, output_path, method=method, **kwargs)
    return brain, mask


def _skull_strip_impl(
    input_path: PathLike,
    output_path: PathLike,
    *,
    method: str = "auto",
    **kwargs,
) -> tuple[Path, Path | None, str]:
    """:func:`skull_strip` that routes per-backend kwargs and reports the method."""

    def _hdbet() -> tuple[Path, Path | None]:
        return skull_strip_hdbet(
            input_path, output_path, **_route_kwargs(skull_strip_hdbet, kwargs, "HD-BET")
        )

    def _synth() -> tuple[Path, Path | None]:
        return skull_strip_synthstrip(
            input_path, output_path, **_route_kwargs(skull_strip_synthstrip, kwargs, "SynthStrip")
        )

    if method == "hdbet":
        return (*_hdbet(), "hdbet")
    if method == "synthstrip":
        return (*_synth(), "synthstrip")
    if method != "auto":
        raise ValueError(f"Unknown skull-strip method {method!r}")

    # Auto: try HD-BET first
    try:
        return (*_hdbet(), "hdbet")
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("HD-BET failed (%s), trying SynthStrip", exc)
        return (*_synth(), "synthstrip")


# ======================================================================
# §6  Step 3 — Registration (SynthMorph affine + uniGradICON deformable)
# ======================================================================


def register_synthmorph_affine(
    moving_path: PathLike,
    fixed_path: PathLike,
    output_path: PathLike,
    transform_path: PathLike | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path | None]:
    """Affine registration via SynthMorph (Hoffmann et al., 2024).

    Contrast-invariant affine alignment trained with domain
    randomization.  Ideal as a robust pre-step before deformable
    registration, especially for large initial misalignments.

    Parameters
    ----------
    moving_path : PathLike
        Moving (subject) brain image.
    fixed_path : PathLike
        Fixed (template) image.
    output_path : PathLike
        Affine-registered output.
    transform_path : PathLike or None
        Output transform file (.lta or .txt).

    Returns
    -------
    (Path, Path or None)
        Registered image and transform file.
    """
    mov = Path(moving_path)
    fix = Path(fixed_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        logger.info("SynthMorph affine output exists, skipping: %s", out.name)
        xfm = Path(transform_path) if transform_path and Path(transform_path).exists() else None
        return out, xfm

    t0 = time.time()
    purge_vram()

    xfm_flag = f"--trans {_q(transform_path)}" if transform_path else ""
    cmd = f"mri_synthmorph -m affine -i {_q(mov)} -t {_q(fix)} -o {_q(out)} {xfm_flag} --gpu"
    _run_cmd(cmd, "SynthMorph affine", timeout=120)

    purge_vram()
    elapsed = time.time() - t0
    logger.info("SynthMorph affine → %s (%.1fs)", out.name, elapsed)
    xfm_out = Path(transform_path) if transform_path and Path(transform_path).exists() else None
    return out, xfm_out


def register_unigradicon(
    moving_path: PathLike,
    fixed_path: PathLike,
    warped_path: PathLike,
    transform_path: PathLike,
    *,
    io_iterations: int | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Deformable registration via uniGradICON (Tian et al., 2024).

    Foundation model for medical image registration.  Produces a
    dense displacement field stored as HDF5.

    Parameters
    ----------
    moving_path : PathLike
        Moving (subject) brain image (ideally affine-pre-aligned).
    fixed_path : PathLike
        Fixed (template) image.
    warped_path : PathLike
        Deformably warped output.
    transform_path : PathLike
        Output HDF5 transform.
    io_iterations : int or None
        Instance optimization iterations.  ``None`` = feedforward
        only (fast); ``50`` = gradient refinement (more accurate).

    Returns
    -------
    (Path, Path)
        Warped image and transform paths.
    """
    mov = Path(moving_path)
    fix = Path(fixed_path)
    warp = Path(warped_path)
    xfm = Path(transform_path)
    warp.parent.mkdir(parents=True, exist_ok=True)

    if warp.exists() and not overwrite:
        logger.info("uniGradICON output exists, skipping: %s", warp.name)
        return warp, xfm

    t0 = time.time()
    purge_vram()

    io_arg = "None" if io_iterations is None else str(io_iterations)
    cmd = (
        f"unigradicon-register "
        f"--fixed={_q(fix)} --fixed_modality=mri "
        f"--moving={_q(mov)} --moving_modality=mri "
        f"--transform_out={_q(xfm)} "
        f"--warped_moving_out={_q(warp)} "
        f"--io_iterations {io_arg}"
    )
    _run_cmd(cmd, "uniGradICON")

    if not warp.exists():
        raise FileNotFoundError(f"uniGradICON output missing: {warp}")

    purge_vram()
    elapsed = time.time() - t0
    logger.info("uniGradICON → %s (%.1fs)", warp.name, elapsed)
    return warp, xfm


def register(
    moving_path: PathLike,
    fixed_path: PathLike,
    output_dir: PathLike,
    stem: str,
    *,
    affine_method: Literal["synthmorph", "none"] = "synthmorph",
    io_iterations: int | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Full registration pipeline: optional affine + deformable.

    Parameters
    ----------
    moving_path : PathLike
        Brain-extracted moving image.
    fixed_path : PathLike
        Template (fixed) image.
    output_dir : PathLike
        Output directory.
    stem : str
        Filename stem for outputs.
    affine_method : {'synthmorph', 'none'}
        Whether to run SynthMorph affine pre-alignment.
    io_iterations : int or None
        uniGradICON IO iterations.

    Returns
    -------
    dict
        Keys: ``'affine'``, ``'affine_xfm'``, ``'warped'``,
        ``'deformable_xfm'``.  Values are Paths (None if skipped).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path | None] = {
        "affine": None,
        "affine_xfm": None,
        "warped": None,
        "deformable_xfm": None,
    }

    # Step A: Affine pre-alignment (optional but recommended)
    if affine_method == "synthmorph":
        affine_out = out_dir / f"{stem}_affine.nii.gz"
        affine_xfm = out_dir / f"{stem}_affine.lta"
        try:
            aff, axfm = register_synthmorph_affine(
                moving_path,
                fixed_path,
                affine_out,
                affine_xfm,
                overwrite=overwrite,
            )
            paths["affine"] = aff
            paths["affine_xfm"] = axfm
            # Use affine-aligned as input to deformable
            moving_path = aff
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning(
                "SynthMorph affine failed (%s), proceeding directly to uniGradICON deformable.",
                exc,
            )

    # Step B: Deformable registration
    warped = out_dir / f"{stem}_MNI.nii.gz"
    deform_xfm = out_dir / f"{stem}_transform.hdf5"
    warp, dxfm = register_unigradicon(
        moving_path,
        fixed_path,
        warped,
        deform_xfm,
        io_iterations=io_iterations,
        overwrite=overwrite,
    )
    paths["warped"] = warp
    paths["deformable_xfm"] = dxfm

    return paths


# ======================================================================
# §7  Step 4 — Tissue segmentation (SynthSeg+)
# ======================================================================


def segment_synthseg(
    input_path: PathLike,
    output_path: PathLike,
    *,
    parc: bool = True,
    robust: bool = False,
    vol_path: PathLike | None = None,
    qc_path: PathLike | None = None,
    overwrite: bool = False,
) -> Path:
    """Tissue segmentation + optional DKT parcellation via SynthSeg+.

    Contrast-agnostic segmentation producing FreeSurfer-compatible
    aseg labels (32 structures) with optional DKT cortical
    parcellation (31 labels per hemisphere).

    Parameters
    ----------
    input_path : PathLike
        T1w NIfTI (raw or preprocessed — SynthSeg is contrast-agnostic).
    output_path : PathLike
        Segmentation output NIfTI.
    parc : bool
        Include DKT cortical parcellation (aparc+DKTatlas).
    robust : bool
        Use SynthSeg-robust mode (slower, better for clinical scans).
    vol_path : PathLike or None
        Output CSV with structure volumes.
    qc_path : PathLike or None
        Output CSV with QC scores.

    Returns
    -------
    Path
        Path to the segmentation volume.
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        logger.info("SynthSeg output exists, skipping: %s", out.name)
        return out

    t0 = time.time()
    purge_vram()

    # Build command
    parts = [f"mri_synthseg --i {_q(inp)} --o {_q(out)}"]
    if parc:
        parts.append("--parc")
    if robust:
        parts.append("--robust")
    if vol_path:
        parts.append(f"--vol {_q(vol_path)}")
    if qc_path:
        parts.append(f"--qc {_q(qc_path)}")
    # GPU flag
    parts.append("--threads 1")  # SynthSeg handles GPU internally

    cmd = " ".join(parts)
    _run_cmd(cmd, "SynthSeg+")

    if not out.exists():
        raise FileNotFoundError(f"SynthSeg output missing: {out}")

    purge_vram()
    elapsed = time.time() - t0
    logger.info("SynthSeg → %s (%.1fs)", out.name, elapsed)
    return out


def segment_fastsurfer(
    input_path: PathLike,
    output_dir: PathLike,
    subject_id: str,
    *,
    device: str = "cuda:0",
) -> Path:
    """Tissue segmentation via FastSurferCNN (Henschel et al., 2020).

    Produces FreeSurfer-compatible aseg + DKT in < 1 minute on GPU.

    Parameters
    ----------
    input_path : PathLike
        T1w NIfTI.
    output_dir : PathLike
        FastSurfer output directory (SUBJECTS_DIR-like).
    subject_id : str
        Subject ID for the output directory structure.
    device : str
        CUDA device.

    Returns
    -------
    Path
        Path to the output directory containing mri/aparc.DKTatlas+aseg.mgz.
    """
    inp = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    purge_vram()

    cmd = (
        f"run_fastsurfer.sh "
        f"--t1 {_q(inp)} --sd {_q(out)} --sid {_q(subject_id)} "
        f"--seg_only --no_cereb --no_biasfield "
        f"--device {_q(device)} --parallel --threads 4"
    )
    _run_cmd(cmd, "FastSurfer", timeout=300)

    purge_vram()
    elapsed = time.time() - t0
    logger.info("FastSurfer → %s/%s (%.1fs)", out, subject_id, elapsed)
    return out / subject_id


def segment(
    input_path: PathLike,
    output_path: PathLike,
    *,
    method: Literal["synthseg", "fastsurfer", "auto"] = "auto",
    **kwargs,
) -> Path:
    """Tissue segmentation using the best available method.

    Parameters
    ----------
    input_path : PathLike
        T1w NIfTI.
    output_path : PathLike
        Segmentation output.
    method : {'synthseg', 'fastsurfer', 'auto'}
        ``'auto'`` tries SynthSeg first.
    **kwargs
        Passed to the chosen method; options a backend does not take are
        ignored (e.g. ``parc``/``vol_path`` for FastSurfer, ``device`` /
        ``subject_id`` for SynthSeg).

    Returns
    -------
    Path
        Segmentation output path. When *output_path* is a NIfTI/MGZ file
        path, FastSurfer's ``aparc.DKTatlas+aseg`` volume is written there,
        so both backends return the same kind of file. (Passing a directory
        with ``method="fastsurfer"`` keeps the legacy behaviour and returns
        the FastSurfer subject directory.)
    """
    return _segment_impl(input_path, output_path, method=method, **kwargs)[0]


def _is_volume_path(p: Path) -> bool:
    return any(p.name.endswith(ext) for ext in (*NIFTI_EXTS, ".mgz", ".mgh"))


def _segment_with_fastsurfer(input_path: PathLike, output_path: PathLike, **kwargs) -> Path:
    """Run FastSurfer and, for a file *output_path*, export its aseg+DKT there."""
    out = Path(output_path)
    inp = Path(input_path)
    stem = inp.name
    for ext in NIFTI_EXTS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    fs_kwargs = _route_kwargs(segment_fastsurfer, kwargs, "FastSurfer")
    subject_id = fs_kwargs.pop("subject_id", None) or stem
    if not _is_volume_path(out):
        return segment_fastsurfer(inp, out, subject_id, **fs_kwargs)

    if out.exists() and not kwargs.get("overwrite", False):
        logger.info("FastSurfer segmentation exists, skipping: %s", out.name)
        return out
    subj_dir = segment_fastsurfer(inp, out.parent / "fastsurfer", subject_id, **fs_kwargs)
    candidates = [
        subj_dir / "mri" / name
        for name in (
            "aparc.DKTatlas+aseg.deep.mgz",
            "aparc.DKTatlas+aseg.mgz",
            "aseg.auto_noCCseg.mgz",
        )
    ]
    seg_src = next((c for c in candidates if c.exists()), None)
    if seg_src is None:
        raise FileNotFoundError(f"FastSurfer segmentation not found in {subj_dir / 'mri'}")
    import nibabel as nib

    img = nib.load(str(seg_src))
    data = np.rint(np.asarray(img.dataobj)).astype(np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.name.endswith((".mgz", ".mgh")):
        nib.save(nib.MGHImage(data, img.affine), str(out))
    else:
        nib.save(nib.Nifti1Image(data, img.affine), str(out))
    logger.info("FastSurfer %s → %s", seg_src.name, out)
    return out


def _segment_impl(
    input_path: PathLike,
    output_path: PathLike,
    *,
    method: str = "auto",
    **kwargs,
) -> tuple[Path, str]:
    """:func:`segment` that routes kwargs per backend and reports the method."""
    if method == "fastsurfer":
        return _segment_with_fastsurfer(input_path, output_path, **kwargs), "fastsurfer"
    if method not in ("synthseg", "auto"):
        raise ValueError(f"Unknown segment method {method!r}")

    # SynthSeg is the default (contrast-agnostic, no FreeSurfer needed)
    try:
        out = segment_synthseg(
            input_path, output_path, **_route_kwargs(segment_synthseg, kwargs, "SynthSeg")
        )
        return out, "synthseg"
    except (FileNotFoundError, RuntimeError) as exc:
        if method == "synthseg":
            raise
        logger.warning("SynthSeg failed (%s), trying FastSurfer", exc)
        return _segment_with_fastsurfer(input_path, output_path, **kwargs), "fastsurfer"


# ======================================================================
# §8  Step 5 — Parcellation (OpenMAP-T1 or BrainParc)
# ======================================================================


def parcellate_openmap(
    input_path: PathLike,
    output_dir: PathLike,
) -> Path:
    """280-region parcellation via OpenMAP-T1 (Nishimaki et al., 2024).

    Covers cortical and subcortical gray matter plus white matter
    tracts (JHU-MNI atlas).  Includes internal skull-stripping,
    cropping, and hemispheric segmentation.

    Parameters
    ----------
    input_path : PathLike
        Raw or enhanced T1w NIfTI.
    output_dir : PathLike
        Output directory.

    Returns
    -------
    Path
        Path to the parcellation NIfTI (280 labels).
    """
    inp = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    purge_vram()

    # OpenMAP-T1 expects: python run_openmap.py --input ... --output ...
    before = _snapshot(out_dir)
    cmd = f"python -m openmap_t1 --input {_q(inp)} --output {_q(out_dir)}"
    _run_cmd(cmd, "OpenMAP-T1", timeout=300)

    # Find the parcellation output written by *this* run (no stale files,
    # no arbitrary pick among several candidates).
    result = _pick_new_output(
        out_dir, before, ("*parcellation*.nii*", "*280*.nii*", "*.nii*"), "OpenMAP-T1"
    )
    purge_vram()
    elapsed = time.time() - t0
    logger.info("OpenMAP-T1 → %s (%.1fs)", result.name, elapsed)
    return result


def parcellate_brainparc(
    input_path: PathLike,
    output_dir: PathLike,
) -> Path:
    """106-region lifespan parcellation via BrainParc (Liu et al., 2026).

    Edge-guided progressive parcellation that works consistently
    across neonates to elderly without retraining.

    Parameters
    ----------
    input_path : PathLike
        Raw or enhanced T1w NIfTI.
    output_dir : PathLike
        Output directory.

    Returns
    -------
    Path
        Path to the parcellation NIfTI (106 labels).
    """
    inp = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    purge_vram()

    before = _snapshot(out_dir)
    cmd = f"python -m brainparc --input {_q(inp)} --output {_q(out_dir)}"
    _run_cmd(cmd, "BrainParc", timeout=300)

    result = _pick_new_output(out_dir, before, ("*parc*.nii*", "*.nii*"), "BrainParc")
    purge_vram()
    elapsed = time.time() - t0
    logger.info("BrainParc → %s (%.1fs)", result.name, elapsed)
    return result


# ======================================================================
# §9  Pipeline result container
# ======================================================================


@dataclass
class PreprocessResult:
    """Container for GPU preprocessing pipeline outputs.

    Attributes
    ----------
    input_path : Path
        Original input NIfTI.
    enhanced_path : Path or None
        Bias-corrected / enhanced image.
    brain_path : Path or None
        Skull-stripped brain image.
    brain_mask_path : Path or None
        Brain mask.
    registered_paths : dict or None
        Registration outputs (affine, warped, transforms).
    segmentation_path : Path or None
        Tissue segmentation (aseg labels).
    parcellation_path : Path or None
        Full brain parcellation.
    template_path : Path or None
        Registration template used.
    timings : dict
        Per-step wall-clock timings in seconds.
    methods : dict
        Which method was used for each step.
    """

    input_path: Path
    enhanced_path: Path | None = None
    brain_path: Path | None = None
    brain_mask_path: Path | None = None
    registered_paths: dict[str, Path] | None = None
    segmentation_path: Path | None = None
    parcellation_path: Path | None = None
    template_path: Path | None = None
    timings: dict[str, float] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Total processing time in seconds."""
        return sum(self.timings.values())

    def summary(self) -> str:
        """Human-readable summary of the pipeline run."""
        lines = [
            f"PreprocessResult for {self.input_path.name}",
            f"  Total time: {self.total_time:.1f}s",
        ]
        for step, t in self.timings.items():
            method = self.methods.get(step, "?")
            lines.append(f"  {step}: {t:.1f}s ({method})")
        return "\n".join(lines)


# ======================================================================
# §10  End-to-end pipeline orchestrator
# ======================================================================


def preprocess_gpu(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    steps: list[str] | None = None,
    enhance_method: Literal["bmex", "deepn4", "auto"] = "auto",
    strip_method: Literal["hdbet", "synthstrip", "auto"] = "auto",
    segment_method: Literal["synthseg", "fastsurfer", "auto"] = "auto",
    parcellate_method: Literal["openmap", "brainparc"] | None = None,
    template: PathLike | None = None,
    io_iterations: int | None = None,
    affine_pre: bool = True,
    device: str = "cuda:0",
    skip_existing: bool = True,
    segment_space: Literal["native", "template"] = "native",
) -> PreprocessResult:
    """End-to-end GPU-native preprocessing pipeline.

    Chains enhancement → skull stripping → registration → segmentation
    (→ parcellation) with VRAM purge between each step.  Replaces the
    traditional FreeSurfer ``recon-all`` + ANTs pipeline with a fully
    GPU-accelerated chain that runs in 2–4 minutes per subject on a
    24 GB consumer GPU.

    Parameters
    ----------
    input_path : PathLike
        Raw T1-weighted NIfTI file.
    output_dir : PathLike
        Output directory for all products.
    steps : list of str or None
        Steps to run: ``['enhance', 'skull_strip', 'register',
        'segment', 'parcellate']``.  None = default (all except
        parcellate).
    enhance_method : {'bmex', 'deepn4', 'auto'}
        Enhancement method.
    strip_method : {'hdbet', 'synthstrip', 'auto'}
        Skull stripping method.
    segment_method : {'synthseg', 'fastsurfer', 'auto'}
        Tissue segmentation method.
    parcellate_method : {'openmap', 'brainparc'} or None
        Full brain parcellation method.  None = skip.
    template : PathLike or None
        Registration template.  None = auto-download MNI.
    io_iterations : int or None
        uniGradICON instance optimization iterations.
    affine_pre : bool
        Run SynthMorph affine before uniGradICON.
    device : str
        CUDA device for HD-BET / FastSurfer.
    skip_existing : bool
        Skip steps whose outputs already exist. ``False`` recomputes every
        step and overwrites its outputs.
    segment_space : {"native", "template"}
        Image that is segmented. ``"native"`` (default) segments the
        subject-space brain (after enhancement/skull-stripping), so label
        maps keep the individual anatomy that shape descriptors measure.
        ``"template"`` segments the registration output (the MNI-warped image,
        whose shape has been normalised by the deformable warp).

    Returns
    -------
    PreprocessResult
        Container with all output paths and timings.

    Examples
    --------
    >>> result = preprocess_gpu(
    ...     "sub-01_T1w.nii.gz", "output/",
    ...     steps=["enhance", "skull_strip", "register", "segment"],
    ... )
    >>> print(result.summary())
    PreprocessResult for sub-01_T1w.nii.gz
      Total time: 142.3s
      enhance: 12.5s (bmex)
      skull_strip: 8.2s (hdbet)
      register: 18.7s (synthmorph_affine+unigradicon)
      segment: 14.3s (synthseg)
    """
    inp = Path(input_path).resolve()
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _strip_nifti_ext(inp.name)

    # Parse steps
    if steps is None:
        active_steps = set(DEFAULT_STEPS)
    else:
        active_steps = set()
        for s in steps:
            if s.lower() in _STEP_MAP:
                active_steps.add(_STEP_MAP[s.lower()])
            else:
                raise ValueError(f"Unknown step '{s}'.  Available: {list(_STEP_MAP.keys())}")

    # Add parcellate if explicitly requested
    if parcellate_method is not None:
        active_steps.add(Step.PARCELLATE)

    if segment_space not in ("native", "template"):
        raise ValueError(f"segment_space must be 'native' or 'template', got {segment_space!r}")
    overwrite = not skip_existing

    result = PreprocessResult(input_path=inp)
    current = inp  # tracks the "current" image through the chain

    # ── Step 1: Enhance ──
    if Step.ENHANCE in active_steps:
        t0 = time.time()
        enhanced = out_dir / f"{stem}_enhanced.nii.gz"
        current, used = _enhance_impl(current, enhanced, method=enhance_method, overwrite=overwrite)
        result.enhanced_path = current
        result.timings["enhance"] = time.time() - t0
        result.methods["enhance"] = used
        logger.info("VRAM after enhance: %s", vram_info())

    # ── Step 2: Skull strip ──
    if Step.SKULL_STRIP in active_steps:
        t0 = time.time()
        brain = out_dir / f"{stem}_brain.nii.gz"
        brain_path, mask_path, used = _skull_strip_impl(
            current,
            brain,
            method=strip_method,
            device=device,
            overwrite=overwrite,
        )
        current = brain_path
        result.brain_path = brain_path
        result.brain_mask_path = mask_path
        result.timings["skull_strip"] = time.time() - t0
        result.methods["skull_strip"] = used
        logger.info("VRAM after skull_strip: %s", vram_info())

    native = current  # last subject-space image (segmentation input by default)

    # ── Step 3: Register ──
    if Step.REGISTER in active_steps:
        t0 = time.time()
        seq = _detect_sequence(inp)
        tmpl = ensure_template(seq, user_template=template)
        result.template_path = tmpl

        reg_paths = register(
            current,
            tmpl,
            out_dir,
            stem,
            affine_method="synthmorph" if affine_pre else "none",
            io_iterations=io_iterations,
            overwrite=overwrite,
        )
        result.registered_paths = reg_paths
        if reg_paths.get("warped"):
            current = reg_paths["warped"]
        result.timings["register"] = time.time() - t0
        aff_tag = "+synthmorph_affine" if affine_pre else ""
        result.methods["register"] = f"unigradicon{aff_tag}"
        logger.info("VRAM after register: %s", vram_info())

    # ── Step 4: Segment ──
    if Step.SEGMENT in active_steps:
        t0 = time.time()
        seg = out_dir / f"{stem}_synthseg.nii.gz"
        vol_csv = out_dir / f"{stem}_volumes.csv"
        qc_csv = out_dir / f"{stem}_qc.csv"
        seg_input = native if segment_space == "native" else current
        seg_path, used = _segment_impl(
            seg_input,
            seg,
            method=segment_method,
            parc=True,
            vol_path=vol_csv,
            qc_path=qc_csv,
            device=device,
            subject_id=stem,
            overwrite=overwrite,
        )
        result.segmentation_path = seg_path
        result.timings["segment"] = time.time() - t0
        result.methods["segment"] = f"{used} ({segment_space})"
        logger.info("VRAM after segment: %s", vram_info())

    # ── Step 5: Parcellate ──
    if Step.PARCELLATE in active_steps and parcellate_method:
        t0 = time.time()
        parc_dir = out_dir / "parcellation"
        if parcellate_method == "openmap":
            parc_path = parcellate_openmap(inp, parc_dir)
        elif parcellate_method == "brainparc":
            parc_path = parcellate_brainparc(inp, parc_dir)
        else:
            raise ValueError(f"Unknown parcellate method: {parcellate_method}")
        result.parcellation_path = parc_path
        result.timings["parcellate"] = time.time() - t0
        result.methods["parcellate"] = parcellate_method
        logger.info("VRAM after parcellate: %s", vram_info())

    logger.info(
        "Pipeline complete for %s in %.1fs",
        inp.name,
        result.total_time,
    )
    return result


def _strip_nifti_ext(name: str) -> str:
    for ext in NIFTI_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _unique_subject_keys(paths: list[Path]) -> list[str]:
    """Filename stems, disambiguated with parent dirs when they collide."""
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same input file is listed more than once.")
    depth = 0
    while True:
        keys = []
        for p in resolved:
            parts = [
                *(q.name for q in reversed(p.parents[:depth] if depth else [])),
                _strip_nifti_ext(p.name),
            ]
            keys.append("_".join(x for x in parts if x))
        if len(set(keys)) == len(keys) or depth >= max(len(p.parents) for p in resolved):
            break
        depth += 1
    if len(set(keys)) != len(keys):  # pragma: no cover - only for identical paths
        keys = [f"{k}_{i}" for i, k in enumerate(keys)]
    if depth:
        logger.warning("Duplicate input stems; using directory-qualified keys: %s", keys)
    return keys


# ======================================================================
# §11  Batch processing
# ======================================================================


def preprocess_gpu_batch(
    input_paths: list[PathLike],
    output_dir: PathLike,
    *,
    steps: list[str] | None = None,
    enhance_method: Literal["bmex", "deepn4", "auto"] = "auto",
    strip_method: Literal["hdbet", "synthstrip", "auto"] = "auto",
    segment_method: Literal["synthseg", "fastsurfer", "auto"] = "auto",
    parcellate_method: Literal["openmap", "brainparc"] | None = None,
    template: PathLike | None = None,
    io_iterations: int | None = None,
    affine_pre: bool = True,
    device: str = "cuda:0",
) -> dict[str, PreprocessResult]:
    """Process multiple subjects sequentially with VRAM isolation.

    Parameters
    ----------
    input_paths : list of PathLike
        NIfTI files to process.
    output_dir : PathLike
        Base output directory (per-subject subdirs created).
    steps, enhance_method, strip_method, segment_method,
    parcellate_method, template, io_iterations, affine_pre, device
        Passed to :func:`preprocess_gpu` for each subject.

    Returns
    -------
    dict of {subject_stem: PreprocessResult}
        Results keyed by filename stem. When several inputs share a stem
        (e.g. ``sub-01/T1w.nii.gz`` and ``sub-02/T1w.nii.gz``) the key — and
        the per-subject output directory — is prefixed with the parent
        directory names until unique, so subjects never share (and silently
        reuse) each other's outputs.
    """
    out_base = Path(output_dir).resolve()
    results: dict[str, PreprocessResult] = {}
    n = len(input_paths)
    keys = _unique_subject_keys([Path(p) for p in input_paths])
    failed: dict[str, str] = {}

    for i, (inp, stem) in enumerate(zip((Path(p) for p in input_paths), keys), 1):
        logger.info(
            "Processing %d/%d: %s",
            i,
            n,
            inp.name,
        )

        subj_dir = out_base / stem
        try:
            result = preprocess_gpu(
                inp,
                subj_dir,
                steps=steps,
                enhance_method=enhance_method,
                strip_method=strip_method,
                segment_method=segment_method,
                parcellate_method=parcellate_method,
                template=template,
                io_iterations=io_iterations,
                affine_pre=affine_pre,
                device=device,
            )
            results[stem] = result
            logger.info("✓ %s: %.1fs total", stem, result.total_time)
        except (TypeError, AttributeError, NameError, NotImplementedError, AssertionError):
            raise  # programming errors would otherwise fail every subject silently
        except Exception as exc:
            logger.error("✗ %s: %s", stem, exc)
            failed[stem] = f"{type(exc).__name__}: {exc}"
            continue

    if failed:
        logger.warning("%d subject(s) failed: %s", len(failed), ", ".join(failed))
    successful = len(results)
    logger.info(
        "Batch complete: %d/%d subjects succeeded.",
        successful,
        n,
    )
    return results


# ======================================================================
# §12  File discovery utility
# ======================================================================


def discover_nifti(input_path: PathLike) -> list[Path]:
    """Find NIfTI files from a file or directory.

    If a file, returns ``[file]``.  If a directory, searches root
    and first-level subdirectories for ``.nii.gz`` and ``.nii``.

    Parameters
    ----------
    input_path : PathLike
        File or directory path.

    Returns
    -------
    list of Path
        Sorted list of NIfTI paths.

    Raises
    ------
    FileNotFoundError
        If path doesn't exist or no NIfTI files found.
    """
    p = Path(input_path).resolve()

    if p.is_file():
        if not any(p.name.endswith(ext) for ext in NIFTI_EXTS):
            raise ValueError(f"Not a NIfTI file: {p}")
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {p}")

    found = []
    for ext in NIFTI_EXTS:
        found.extend(p.glob(f"*{ext}"))
        for subdir in sorted(p.iterdir()):
            if subdir.is_dir():
                found.extend(subdir.glob(f"*{ext}"))

    found = sorted(set(found))
    if not found:
        raise FileNotFoundError(f"No NIfTI files in {p}")
    return found


# ======================================================================
# __all__
# ======================================================================

__all__ = [
    "DEFAULT_STEPS",
    # Pipeline
    "PreprocessResult",
    # Constants
    "Step",
    # Utilities
    "discover_nifti",
    # Individual steps
    "enhance",
    "enhance_bmex",
    "enhance_deepn4",
    # Template management
    "ensure_template",
    "parcellate_brainparc",
    "parcellate_openmap",
    "preprocess_gpu",
    "preprocess_gpu_batch",
    # VRAM management
    "purge_vram",
    "register",
    "register_synthmorph_affine",
    "register_unigradicon",
    "segment",
    "segment_fastsurfer",
    "segment_synthseg",
    "skull_strip",
    "skull_strip_hdbet",
    "skull_strip_synthstrip",
    "vram_info",
]
