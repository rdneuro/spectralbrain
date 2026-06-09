"""Import TractSeg outputs for spectral shape analysis.

`TractSeg <https://github.com/MIC-DKFZ/TractSeg>`_ writes one binary
NIfTI mask per white-matter bundle into a ``bundle_segmentations/``
directory (72 bundles in the standard atlas, e.g. ``CST_left.nii.gz``,
``AF_left.nii.gz``). This module turns those masks into the geometric
objects SpectralBrain operates on:

- ``output="pointcloud"`` → a :class:`~spectralbrain.core.pointclouds.BrainPointCloud`
  of the mask's world-space voxel coordinates (ready for point-cloud
  Laplacian spectral analysis).
- ``output="mesh"`` → a :class:`~spectralbrain.core.meshes.BrainMesh`
  isosurface (marching cubes on the binary mask), ready for
  ``.decompose()`` and the mesh descriptors.

Both carry the bundle name and source path in their metadata.

Examples
--------
>>> bundles = load_tractseg("/data/sub-01/tractseg_output", output="mesh")
>>> cst = bundles["CST_left"]
>>> decomp = cst.decompose(k=80)
>>> hks = sb.compute_hks(decomp, t_values=[1, 10, 100])

>>> # A single bundle across a cohort, as point clouds:
>>> files = discover_tractseg_subjects("/data/derivatives/tractseg", "CST_left")
>>> clouds = {sid: load_tractseg_bundle(p) for sid, p in files.items()}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

#: Subdirectory TractSeg writes bundle masks into.
_DEFAULT_SUBDIR = "bundle_segmentations"


# ======================================================================
# §1  DISCOVERY
# ======================================================================


def discover_tractseg_bundles(
    tractseg_dir: PathLike,
    *,
    bundles: list[str] | None = None,
    subdir: str | None = _DEFAULT_SUBDIR,
) -> dict[str, Path]:
    """Find bundle-segmentation masks in one subject's TractSeg output.

    Parameters
    ----------
    tractseg_dir : PathLike
        TractSeg output directory for a subject.
    bundles : list of str, optional
        Restrict to these bundle names (without extension, e.g.
        ``"CST_left"``). Defaults to every ``*.nii.gz`` mask found.
    subdir : str, optional
        Subdirectory holding the masks (default ``"bundle_segmentations"``).
        Pass ``None`` to search *tractseg_dir* directly.

    Returns
    -------
    dict of {bundle_name: Path}
    """
    root = Path(tractseg_dir)
    search_dir = root / subdir if subdir else root
    if not search_dir.is_dir():
        raise FileNotFoundError(f"No TractSeg mask directory at {search_dir}")

    found: dict[str, Path] = {}
    for p in sorted(search_dir.glob("*.nii*")):
        name = p.name.split(".")[0]
        found[name] = p

    if bundles is not None:
        missing = [b for b in bundles if b not in found]
        for b in missing:
            logger.warning("Bundle %r not found in %s", b, search_dir)
        found = {b: found[b] for b in bundles if b in found}

    logger.info("TractSeg: %d bundle masks in %s", len(found), search_dir)
    return found


def discover_tractseg_subjects(
    root: PathLike,
    bundle: str,
    *,
    pattern: str = "sub-{sub}/tractseg_output",
    subdir: str | None = _DEFAULT_SUBDIR,
    subjects: list[str] | None = None,
) -> dict[str, Path]:
    """Find one bundle's mask across subjects in a derivatives tree.

    Parameters
    ----------
    root : PathLike
        Derivatives root containing per-subject TractSeg outputs.
    bundle : str
        Bundle name (e.g. ``"CST_left"``).
    pattern : str
        Per-subject TractSeg directory relative to *root*, with a
        ``{sub}`` placeholder (bare label).
    subdir : str, optional
        Mask subdirectory within each subject's TractSeg output.
    subjects : list of str, optional
        Restrict to these subjects.

    Returns
    -------
    dict of {subject_id: Path}
        One bundle mask per subject.
    """
    root = Path(root)
    if subjects is None:
        labels = sorted(p.name[4:] for p in root.glob("sub-*") if p.is_dir())
    else:
        labels = [s[4:] if s.startswith("sub-") else s for s in subjects]

    found: dict[str, Path] = {}
    for label in labels:
        ts_dir = root / pattern.replace("{sub}", label)
        search_dir = ts_dir / subdir if subdir else ts_dir
        matches = sorted(search_dir.glob(f"{bundle}.nii*")) if search_dir.is_dir() else []
        if not matches:
            logger.warning("No %s mask for sub-%s", bundle, label)
            continue
        found[f"sub-{label}"] = matches[0]
    logger.info("TractSeg cohort: %d subjects with bundle %r", len(found), bundle)
    return found


# ======================================================================
# §2  LOADING
# ======================================================================


def load_tractseg_bundle(
    mask_path: PathLike,
    *,
    output: str = "pointcloud",
    level: float = 0.5,
    jitter: bool = False,
    jitter_scale: float = 0.25,
    seed: int | None = None,
    step_size: int = 1,
) -> Any:
    """Load one TractSeg bundle mask as a point cloud or isosurface mesh.

    Parameters
    ----------
    mask_path : PathLike
        A binary (or probabilistic) bundle mask NIfTI.
    output : ``"pointcloud"`` or ``"mesh"``
        ``"pointcloud"`` returns a :class:`BrainPointCloud` of the
        mask's world-space voxel coordinates; ``"mesh"`` returns a
        :class:`BrainMesh` isosurface via marching cubes.
    level : float
        Threshold separating inside/outside the bundle (default ``0.5``;
        appropriate for binary masks and TractSeg probability maps).
    jitter, jitter_scale, seed :
        Point-cloud only — optional sub-voxel jitter to break the regular
        grid (helps point-cloud Laplacian estimation).
    step_size : int
        Mesh only — marching-cubes step (larger = coarser/faster).

    Returns
    -------
    BrainPointCloud or BrainMesh
        With ``metadata["bundle"]`` and ``metadata["source"]`` set.
    """
    from spectralbrain.io.loaders import labels_to_pointcloud, load_nifti

    path = Path(mask_path)
    vol, affine = load_nifti(path)
    binary = (np.asarray(vol) > level).astype(np.int16)
    if binary.sum() == 0:
        raise ValueError(f"Empty mask (no voxels above {level}) in {path.name}")

    bundle = path.name.split(".")[0]
    meta = {"bundle": bundle, "source": str(path)}

    if output == "pointcloud":
        from spectralbrain.core.pointclouds import BrainPointCloud

        pts = labels_to_pointcloud(
            binary,
            affine,
            label_id=1,
            jitter=jitter,
            jitter_scale=jitter_scale,
            seed=seed,
        )
        return BrainPointCloud(pts, metadata={**meta, "n_voxels": int(binary.sum())})

    if output == "mesh":
        from spectralbrain.core.base import marching_cubes
        from spectralbrain.core.meshes import BrainMesh

        verts, faces = marching_cubes(
            binary.astype(np.float32), affine, level=0.5, step_size=step_size
        )
        return BrainMesh(verts, faces, metadata=meta)

    raise ValueError(f"Unknown output {output!r}; use 'pointcloud' or 'mesh'.")


def load_tractseg(
    tractseg_dir: PathLike,
    *,
    bundles: list[str] | None = None,
    output: str = "pointcloud",
    subdir: str | None = _DEFAULT_SUBDIR,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load all (or selected) bundles from one subject's TractSeg output.

    Parameters
    ----------
    tractseg_dir : PathLike
        TractSeg output directory for a subject.
    bundles : list of str, optional
        Restrict to these bundle names. Defaults to all masks found.
    output : ``"pointcloud"`` or ``"mesh"``
        Geometric representation per bundle.
    subdir : str, optional
        Mask subdirectory (default ``"bundle_segmentations"``).
    **kwargs
        Forwarded to :func:`load_tractseg_bundle` (``level``, ``jitter``,
        ``step_size``, …).

    Returns
    -------
    dict of {bundle_name: BrainPointCloud or BrainMesh}
        Bundles that fail to load (e.g. empty masks) are logged and skipped.
    """
    files = discover_tractseg_bundles(tractseg_dir, bundles=bundles, subdir=subdir)
    out: dict[str, Any] = {}
    for name, path in files.items():
        try:
            out[name] = load_tractseg_bundle(path, output=output, **kwargs)
        except Exception as exc:
            logger.error("✗ %s: %s", name, exc)
    logger.info("Loaded %d/%d TractSeg bundles as %s.", len(out), len(files), output)
    return out


__all__ = [
    "discover_tractseg_bundles",
    "discover_tractseg_subjects",
    "load_tractseg",
    "load_tractseg_bundle",
]
