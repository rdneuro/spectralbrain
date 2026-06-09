"""Unified loaders for neuroimaging and geometry formats.

Every loader returns plain NumPy arrays using the canonical type
aliases from :mod:`spectralbrain.runtime`.  No loader returns
library-specific objects (nibabel images, PyVista meshes, etc.) —
downstream modules receive only arrays and dicts.

The auto-detection function :func:`load` inspects the file extension
(and, when ambiguous, magic bytes) to dispatch to the correct
format-specific loader.

Dependencies
------------
- **nibabel** — required for FreeSurfer, GIfTI, NIfTI, MGZ.
  Lazy-imported so ``import spectralbrain`` works without it.
- **pyvista** — core dependency; reads generic meshes
  (.ply / .obj / .stl / .vtk / .vtp) natively via VTK.
- **h5py** — core dependency, for HDF5 cache files.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from spectralbrain.runtime import (
    DescriptorMatrix,
    Faces,
    GeometryFormat,
    LabelArray,
    PathLike,
    Points,
    ScalarMap,
    Vertices,
    get_logger,
)

logger = get_logger(__name__)

# ======================================================================
# Lazy imports — keep ``import spectralbrain`` fast
# ======================================================================


def _require_nibabel():
    """Import nibabel or raise a helpful error."""
    try:
        import nibabel as nib

        return nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for neuroimaging I/O.\n  pip install nibabel"
        ) from exc


def _require_h5py():
    """Import h5py or raise a helpful error."""
    try:
        import h5py

        return h5py
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 I/O.\n  pip install h5py") from exc


# ======================================================================
# §1  FORMAT DETECTION
# ======================================================================

# Extension → GeometryFormat mapping.  Checked in order; first match
# wins.  Compound extensions (.surf.gii) are tried before simple ones.

_EXT_MAP: dict[str, GeometryFormat] = {
    # GIfTI (compound extensions first)
    ".surf.gii": GeometryFormat.GIFTI_SURFACE,
    ".func.gii": GeometryFormat.GIFTI_FUNC,
    ".shape.gii": GeometryFormat.GIFTI_FUNC,
    ".label.gii": GeometryFormat.GIFTI_LABEL,
    # NIfTI
    ".nii.gz": GeometryFormat.NIFTI_VOLUME,
    ".nii": GeometryFormat.NIFTI_VOLUME,
    # FreeSurfer volumes
    ".mgz": GeometryFormat.MGZ_VOLUME,
    ".mgh": GeometryFormat.MGZ_VOLUME,
    # FreeSurfer overlays
    ".annot": GeometryFormat.FREESURFER_ANNOT,
    ".thickness": GeometryFormat.FREESURFER_MORPH,
    ".curv": GeometryFormat.FREESURFER_MORPH,
    ".sulc": GeometryFormat.FREESURFER_MORPH,
    ".area": GeometryFormat.FREESURFER_MORPH,
    ".label": GeometryFormat.FREESURFER_LABEL,
    # Generic meshes
    ".ply": GeometryFormat.PLY,
    ".obj": GeometryFormat.OBJ,
    ".stl": GeometryFormat.STL,
    ".vtk": GeometryFormat.VTK,
    ".vtp": GeometryFormat.VTK,
    # Cache / raw
    ".h5": GeometryFormat.HDF5,
    ".hdf5": GeometryFormat.HDF5,
    ".npz": GeometryFormat.NUMPY,
}

# FreeSurfer surface files have no extension but a magic number.
_FS_SURFACE_MAGIC = b"\xff\xff\xfe"  # 3 bytes: 255, 255, 254


def detect_format(path: PathLike) -> GeometryFormat:
    """Identify the geometry format of a file.

    Parameters
    ----------
    path : PathLike
        File to inspect.

    Returns
    -------
    GeometryFormat

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    p = Path(path)
    name = p.name.lower()

    # Try compound extensions first (longest match).
    for ext, fmt in _EXT_MAP.items():
        if name.endswith(ext):
            return fmt

    # FreeSurfer surfaces have no extension — check magic bytes.
    if p.is_file():
        try:
            with open(p, "rb") as fh:
                magic = fh.read(3)
            if magic == _FS_SURFACE_MAGIC:
                return GeometryFormat.FREESURFER_SURFACE
        except OSError:
            pass

    # Heuristic: common FS surface names without extensions.
    stem = p.name.lower()
    fs_surf_names = {
        "lh.white",
        "rh.white",
        "lh.pial",
        "rh.pial",
        "lh.inflated",
        "rh.inflated",
        "lh.sphere",
        "rh.sphere",
        "lh.midthickness",
        "rh.midthickness",
        "lh.smoothwm",
        "rh.smoothwm",
    }
    if stem in fs_surf_names:
        return GeometryFormat.FREESURFER_SURFACE

    raise ValueError(f"Cannot detect format of '{p}'.  Known extensions: {sorted(_EXT_MAP.keys())}")


# ======================================================================
# §2  UNIFIED LOADER
# ======================================================================


def load(
    path: PathLike,
    *,
    fmt: GeometryFormat | None = None,
) -> dict[str, Any]:
    """Auto-detect format and load a neuroimaging / geometry file.

    This is the recommended entry point for users who don't want to
    think about file formats.  The returned dict always contains a
    ``"format"`` key; other keys depend on the format.

    Parameters
    ----------
    path : PathLike
        File to load.
    fmt : GeometryFormat, optional
        Force a specific format (skip auto-detection).

    Returns
    -------
    dict
        Contents vary by format.  Guaranteed keys:

        - ``"format"`` : :class:`GeometryFormat`

        Surface files add ``"vertices"`` and ``"faces"``.
        Scalar overlays add ``"scalars"``.
        Annotations add ``"labels"``, ``"ctab"``, ``"names"``.
        Volumes add ``"data"``, ``"affine"``.

    Raises
    ------
    ValueError
        Unknown format or failed auto-detection.
    FileNotFoundError
        Path does not exist.

    Examples
    --------
    >>> result = sb.io.load("lh.white")
    >>> verts, faces = result["vertices"], result["faces"]

    >>> result = sb.io.load("lh.aparc.annot")
    >>> labels, names = result["labels"], result["names"]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: '{p}'")

    if fmt is None:
        fmt = detect_format(p)

    dispatch = {
        GeometryFormat.FREESURFER_SURFACE: _load_fs_surface,
        GeometryFormat.FREESURFER_ANNOT: _load_fs_annot,
        GeometryFormat.FREESURFER_MORPH: _load_fs_morph,
        GeometryFormat.FREESURFER_LABEL: _load_fs_label,
        GeometryFormat.GIFTI_SURFACE: _load_gifti_surface,
        GeometryFormat.GIFTI_FUNC: _load_gifti_func,
        GeometryFormat.GIFTI_LABEL: _load_gifti_label,
        GeometryFormat.NIFTI_VOLUME: _load_nifti,
        GeometryFormat.MGZ_VOLUME: _load_nifti,  # nibabel handles MGZ
        GeometryFormat.PLY: _load_generic_mesh,
        GeometryFormat.OBJ: _load_generic_mesh,
        GeometryFormat.STL: _load_generic_mesh,
        GeometryFormat.VTK: _load_generic_mesh,
        GeometryFormat.HDF5: _load_hdf5,
        GeometryFormat.NUMPY: _load_npz,
    }

    loader = dispatch.get(fmt)
    if loader is None:
        raise ValueError(f"No loader for format {fmt}")

    result = loader(p)
    result["format"] = fmt
    logger.debug("Loaded %s as %s", p.name, fmt.name)
    return result


# ======================================================================
# §3  FORMAT-SPECIFIC LOADERS
# ======================================================================

# ── FreeSurfer surface (.white, .pial, …) ────────────────────────────


def _load_fs_surface(path: Path) -> dict[str, Any]:
    """Load a FreeSurfer binary surface file (.white, .pial, .inflated)."""
    nib = _require_nibabel()
    vertices, faces = nib.freesurfer.read_geometry(str(path))
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
    }


def load_freesurfer_surface(path: PathLike) -> tuple[Vertices, Faces]:
    """Load a FreeSurfer surface file.

    Parameters
    ----------
    path : PathLike
        Path to a FreeSurfer surface (``.white``, ``.pial``,
        ``.inflated``, ``.sphere``, …).

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates in TkRAS mm.
    faces : ndarray, shape (F, 3)
        Triangle indices, 0-indexed.

    Examples
    --------
    >>> verts, faces = sb.io.load_freesurfer_surface("lh.white")
    >>> verts.shape
    (163842, 3)
    """
    result = _load_fs_surface(Path(path))
    return result["vertices"], result["faces"]


# ── FreeSurfer annotation (.annot) ────────────────────────────────────


def _load_fs_annot(path: Path) -> dict[str, Any]:
    """Load a FreeSurfer annotation file (.annot)."""
    nib = _require_nibabel()
    labels, ctab, names = nib.freesurfer.read_annot(str(path))
    # names come as bytes in some nibabel versions; decode.
    decoded_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]
    return {
        "labels": np.asarray(labels, dtype=np.int32),
        "ctab": np.asarray(ctab),
        "names": decoded_names,
    }


def load_freesurfer_annot(
    path: PathLike,
) -> tuple[LabelArray, np.ndarray, list[str]]:
    """Load a FreeSurfer annotation (parcellation overlay).

    Parameters
    ----------
    path : PathLike
        Path to ``.annot`` file (e.g. ``lh.aparc.annot``).

    Returns
    -------
    labels : ndarray, shape (N,)
        Per-vertex parcel index.
    ctab : ndarray, shape (n_labels, 5)
        Colour table (RGBT + label ID).
    names : list of str
        Region names, one per row of *ctab*.

    Examples
    --------
    >>> labels, ctab, names = sb.io.load_freesurfer_annot(
    ...     "lh.aparc.a2009s.annot")
    >>> set(labels)  # unique parcel IDs
    """
    result = _load_fs_annot(Path(path))
    return result["labels"], result["ctab"], result["names"]


# ── FreeSurfer morphometry (.thickness, .curv, .sulc, .area) ─────────


def _load_fs_morph(path: Path) -> dict[str, Any]:
    """Load a FreeSurfer morphometry overlay (.thickness, .curv, .sulc)."""
    nib = _require_nibabel()
    scalars = nib.freesurfer.read_morph_data(str(path))
    return {
        "scalars": np.asarray(scalars, dtype=np.float64),
    }


def load_freesurfer_morph(path: PathLike) -> ScalarMap:
    """Load a FreeSurfer per-vertex scalar overlay.

    Parameters
    ----------
    path : PathLike
        Path to ``.thickness``, ``.curv``, ``.sulc``, or ``.area``.

    Returns
    -------
    ndarray, shape (N,)
        Per-vertex scalar values.
    """
    return _load_fs_morph(Path(path))["scalars"]


# ── FreeSurfer label (.label) ─────────────────────────────────────────


def _load_fs_label(path: Path) -> dict[str, Any]:
    """Load a FreeSurfer label file (.label)."""
    nib = _require_nibabel()
    label_array, scalar_values = nib.freesurfer.read_label(
        str(path),
        read_scalars=True,
    )
    return {
        "indices": np.asarray(label_array, dtype=np.int64),
        "scalars": (
            np.asarray(scalar_values, dtype=np.float64) if scalar_values is not None else None
        ),
    }


# ── GIfTI surface (.surf.gii) ────────────────────────────────────────


def _load_gifti_surface(path: Path) -> dict[str, Any]:
    """Load a GIFTI surface file (.surf.gii)."""
    nib = _require_nibabel()
    img = nib.load(str(path))
    vertices = img.darrays[0].data
    faces = img.darrays[1].data
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
    }


def load_gifti_surface(path: PathLike) -> tuple[Vertices, Faces]:
    """Load a GIfTI surface file.

    Parameters
    ----------
    path : PathLike
        Path to ``.surf.gii``.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    """
    r = _load_gifti_surface(Path(path))
    return r["vertices"], r["faces"]


# ── GIfTI functional / shape (.func.gii, .shape.gii) ─────────────────


def _load_gifti_func(path: Path) -> dict[str, Any]:
    """Load a GIFTI functional/metric file (.func.gii, .shape.gii)."""
    nib = _require_nibabel()
    img = nib.load(str(path))
    # May contain one or more data arrays (e.g. multi-frame).
    arrays = [np.asarray(da.data, dtype=np.float64) for da in img.darrays]
    if len(arrays) == 1:
        return {"scalars": arrays[0]}
    # Stack frames → (N, T) descriptor matrix.
    return {"scalars": np.column_stack(arrays)}


def load_gifti_func(path: PathLike) -> ScalarMap | DescriptorMatrix:
    """Load a GIfTI functional / shape overlay.

    Parameters
    ----------
    path : PathLike
        Path to ``.func.gii`` or ``.shape.gii``.

    Returns
    -------
    ndarray, shape (N,) or (N, T)
        Scalar map (single frame) or descriptor matrix (multi-frame).
    """
    return _load_gifti_func(Path(path))["scalars"]


# ── GIfTI label (.label.gii) ─────────────────────────────────────────


def _load_gifti_label(path: Path) -> dict[str, Any]:
    """Load a GIFTI label file (.label.gii)."""
    nib = _require_nibabel()
    img = nib.load(str(path))
    labels = np.asarray(img.darrays[0].data, dtype=np.int32)
    # Extract label table if present.
    names: list[str] = []
    lt = img.labeltable
    if lt is not None and hasattr(lt, "labels"):
        names = [lbl.label if hasattr(lbl, "label") else str(lbl) for lbl in lt.labels]
    return {"labels": labels, "names": names}


def load_gifti_label(path: PathLike) -> tuple[LabelArray, list[str]]:
    """Load a GIfTI label overlay.

    Parameters
    ----------
    path : PathLike
        Path to ``.label.gii``.

    Returns
    -------
    labels : ndarray, shape (N,)
    names : list of str
    """
    r = _load_gifti_label(Path(path))
    return r["labels"], r["names"]


# ── NIfTI / MGZ volume ────────────────────────────────────────────────


def _load_nifti(path: Path) -> dict[str, Any]:
    """Load a NIfTI volume file (.nii, .nii.gz)."""
    nib = _require_nibabel()
    img = nib.load(str(path))
    return {
        "data": np.asarray(img.dataobj),
        "affine": np.asarray(img.affine, dtype=np.float64),
        "header": img.header,
    }


def load_nifti(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI or MGZ volume.

    Parameters
    ----------
    path : PathLike
        Path to ``.nii``, ``.nii.gz``, ``.mgz``, or ``.mgh``.

    Returns
    -------
    data : ndarray
        Volume data (3D or 4D).
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine.
    """
    r = _load_nifti(Path(path))
    return r["data"], r["affine"]


# ── Generic mesh (.ply, .obj, .stl, .vtk) ────────────────────────────


def _load_generic_mesh(path: Path) -> dict[str, Any]:
    """Load a generic mesh file (.ply, .obj, .stl, .vtk, .vtp).

    Uses PyVista — a core dependency, VTK-backed — which reads all of
    these formats natively.  The mesh is triangulated on load, so the
    returned ``faces`` array is always ``(F, 3)`` even if the source
    stored quads or triangle strips.
    """
    import pyvista as pv

    mesh = pv.read(str(path))
    # Non-PolyData (e.g. UnstructuredGrid from some .vtk files) → surface.
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()

    verts = np.asarray(mesh.points, dtype=np.float64)
    faces_flat = np.asarray(mesh.faces, dtype=np.int64)
    if faces_flat.size == 0:
        faces = np.empty((0, 3), dtype=np.int64)
    else:
        # PyVista packs faces as [3, i, j, k, 3, i, j, k, ...].
        faces = faces_flat.reshape(-1, 4)[:, 1:]
    return {"vertices": verts, "faces": faces}


def load_mesh(path: PathLike) -> tuple[Vertices, Faces]:
    """Load a mesh from a generic format (.ply, .obj, .stl, .vtk, .vtp).

    Backed by PyVista, so every listed format works with a default
    install (no optional dependency required).

    Parameters
    ----------
    path : PathLike
        Mesh file.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    """
    r = _load_generic_mesh(Path(path))
    return r["vertices"], r["faces"]


# ── HDF5 (.h5) ───────────────────────────────────────────────────────


def _load_hdf5(path: Path) -> dict[str, Any]:
    """Load data from an HDF5 file (.h5, .hdf5)."""
    h5py = _require_h5py()
    result: dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            ds = f[key]
            if hasattr(ds, "shape"):
                result[key] = np.asarray(ds)
            else:
                # Group — recurse one level.
                result[key] = {k: np.asarray(v) for k, v in ds.items()}
    return result


# ── NumPy archive (.npz) ─────────────────────────────────────────────


def _load_npz(path: Path) -> dict[str, Any]:
    """Load data from a NumPy compressed archive (.npz)."""
    data = np.load(str(path), allow_pickle=False)
    return dict(data)


# ======================================================================
# §4  VOLUMETRIC LABEL → POINT CLOUD
# ======================================================================


def labels_to_pointcloud(
    label_volume: np.ndarray,
    affine: np.ndarray,
    label_id: int,
    *,
    jitter: bool = False,
    jitter_scale: float = 0.25,
    seed: int | None = None,
) -> Points:
    """Extract a point cloud from a volumetric segmentation.

    Given a 3D integer label volume (e.g. FreeSurfer ``aseg.mgz``)
    and a target label ID, returns the world-space (RAS) coordinates
    of all voxels with that label.

    This is the pathway ③ from the SpectralBrain I/O diagram:
    volumetric segmentation → point cloud → spectral descriptors.

    Parameters
    ----------
    label_volume : ndarray, shape (X, Y, Z)
        Integer label volume.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine matrix.
    label_id : int
        Target label (e.g. 17 for left hippocampus in aseg).
    jitter : bool
        Add sub-voxel Gaussian jitter to break the grid pattern.
        Useful for point-cloud Laplacian estimation, where a
        regular grid causes degenerate eigenvalues.
    jitter_scale : float
        Standard deviation of the jitter in voxel units (default
        0.25 — i.e. ±0.25 voxels).
    seed : int, optional
        RNG seed for reproducible jitter.

    Returns
    -------
    points : ndarray, shape (N, 3)
        World-space coordinates of the extracted voxels.

    Raises
    ------
    ValueError
        If *label_id* is not found in the volume.

    Examples
    --------
    >>> data, affine = sb.io.load_nifti("aseg.mgz")
    >>> hippo_L = sb.io.labels_to_pointcloud(data, affine, label_id=17)
    >>> hippo_L.shape
    (4231, 3)
    """
    label_volume = np.asarray(label_volume)
    affine = np.asarray(affine, dtype=np.float64)

    if label_volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {label_volume.shape}")

    mask = label_volume == label_id
    if not mask.any():
        raise ValueError(
            f"Label {label_id} not found in volume.  "
            f"Unique labels: {np.unique(label_volume[:20])!r}…"
        )

    # Voxel indices → (N, 3) int array.
    ijk = np.argwhere(mask).astype(np.float64)  # (N, 3)

    if jitter:
        rng = np.random.default_rng(seed)
        ijk += rng.normal(scale=jitter_scale, size=ijk.shape)

    # Apply affine: world = affine @ [i, j, k, 1]ᵀ
    ones = np.ones((ijk.shape[0], 1), dtype=np.float64)
    ijk_h = np.hstack([ijk, ones])  # (N, 4)
    xyz = (affine @ ijk_h.T).T[:, :3]  # (N, 3)

    logger.info("Extracted %d points for label %d", xyz.shape[0], label_id)
    return xyz


# ======================================================================
# §5  PARCELLATION UTILITIES
# ======================================================================


def extract_submesh(
    vertices: Vertices,
    faces: Faces,
    vertex_mask: np.ndarray,
) -> tuple[Vertices, Faces]:
    """Extract the sub-mesh defined by a vertex mask.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Full mesh vertices.
    faces : ndarray, shape (F, 3)
        Full mesh faces.
    vertex_mask : ndarray, shape (N,), bool
        ``True`` for vertices to keep.

    Returns
    -------
    sub_vertices : ndarray, shape (M, 3)
        Subset of vertices.
    sub_faces : ndarray, shape (G, 3)
        Re-indexed faces referencing *sub_vertices*.
    """
    vertex_mask = np.asarray(vertex_mask, dtype=bool)
    if vertex_mask.shape[0] != vertices.shape[0]:
        raise ValueError(f"Mask length {vertex_mask.shape[0]} != n_vertices {vertices.shape[0]}")

    # Only keep faces where ALL three vertices are in the mask.
    face_mask = vertex_mask[faces].all(axis=1)  # (F,) bool
    kept_faces = faces[face_mask]  # (G, 3)

    # Build old→new index map.
    old_to_new = np.full(vertices.shape[0], -1, dtype=np.int64)
    new_indices = np.where(vertex_mask)[0]
    old_to_new[new_indices] = np.arange(new_indices.size, dtype=np.int64)

    sub_vertices = vertices[new_indices]  # (M, 3)
    sub_faces = old_to_new[kept_faces]  # (G, 3)

    # Sanity: no unmapped indices.
    assert (sub_faces >= 0).all(), "Bug in extract_submesh: unmapped index"

    return sub_vertices, sub_faces


def apply_parcellation(
    vertices: Vertices,
    faces: Faces,
    labels: LabelArray,
    *,
    ignore_labels: list[int] | None = None,
) -> dict[int, tuple[Vertices, Faces]]:
    """Split a surface into sub-meshes according to a parcellation.

    Given a cortical mesh and a per-vertex label array (e.g. from a
    Schaefer ``.annot``), extracts one sub-mesh per parcel.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Full mesh vertices.
    faces : ndarray, shape (F, 3)
        Full mesh faces.
    labels : ndarray, shape (N,)
        Per-vertex parcel labels.
    ignore_labels : list of int, optional
        Labels to skip (e.g. ``[0]`` for the medial wall in Schaefer).

    Returns
    -------
    dict of {int: (vertices, faces)}
        Mapping from label ID to the corresponding sub-mesh.
        Each sub-mesh has re-indexed faces starting from 0.

    Notes
    -----
    This is the building block for the geometric connectome:
    apply a Schaefer-200 parcellation, compute spectral descriptors
    per parcel, and build a 200×200 similarity matrix.

    Examples
    --------
    >>> verts, faces = sb.io.load_freesurfer_surface("lh.white")
    >>> labels, _, names = sb.io.load_freesurfer_annot(
    ...     "lh.Schaefer2018_200Parcels_7Networks_order.annot")
    >>> parcels = sb.io.apply_parcellation(verts, faces, labels,
    ...                                     ignore_labels=[0])
    >>> len(parcels)
    100  # 100 left-hemisphere parcels
    >>> parcels[1][0].shape  # vertices of parcel 1
    (823, 3)
    """
    labels = np.asarray(labels)
    if labels.shape[0] != vertices.shape[0]:
        raise ValueError(f"Label length {labels.shape[0]} != n_vertices {vertices.shape[0]}")

    ignore = set(ignore_labels or [])
    unique_labels = sorted(set(np.unique(labels).tolist()) - ignore)

    parcels: dict[int, tuple[Vertices, Faces]] = {}
    for lab in unique_labels:
        mask = labels == lab
        n_verts = mask.sum()
        if n_verts < 3:
            logger.warning("Label %d has only %d vertices — skipping.", lab, n_verts)
            continue
        sub_v, sub_f = extract_submesh(vertices, faces, mask)
        if sub_f.shape[0] == 0:
            logger.warning(
                "Label %d has vertices but no complete triangles — skipping.",
                lab,
            )
            continue
        parcels[lab] = (sub_v, sub_f)

    logger.info(
        "Parcellated surface into %d regions (%d labels ignored)",
        len(parcels),
        len(ignore),
    )
    return parcels


# ======================================================================
# §5b  PARCELLATION REMAPPING AND AGGREGATION
# ======================================================================

# ── Predefined lobe/network groupings ────────────────────────────────

DESIKAN_LOBE_MAP: dict[str, str] = {
    # Frontal
    "superiorfrontal": "frontal",
    "rostralmiddlefrontal": "frontal",
    "caudalmiddlefrontal": "frontal",
    "parsopercularis": "frontal",
    "parstriangularis": "frontal",
    "parsorbitalis": "frontal",
    "lateralorbitofrontal": "frontal",
    "medialorbitofrontal": "frontal",
    "precentral": "frontal",
    "paracentral": "frontal",
    "frontalpole": "frontal",
    "rostralanteriorcingulate": "frontal",
    "caudalanteriorcingulate": "frontal",
    # Parietal
    "superiorparietal": "parietal",
    "inferiorparietal": "parietal",
    "supramarginal": "parietal",
    "postcentral": "parietal",
    "precuneus": "parietal",
    "posteriorcingulate": "parietal",
    "isthmuscingulate": "parietal",
    # Temporal
    "superiortemporal": "temporal",
    "middletemporal": "temporal",
    "inferiortemporal": "temporal",
    "bankssts": "temporal",
    "fusiform": "temporal",
    "transversetemporal": "temporal",
    "entorhinal": "temporal",
    "temporalpole": "temporal",
    "parahippocampal": "temporal",
    # Occipital
    "lateraloccipital": "occipital",
    "lingual": "occipital",
    "cuneus": "occipital",
    "pericalcarine": "occipital",
    # Insular
    "insula": "insular",
}
"""Map from Desikan-Killiany (aparc) region names to lobe labels.

Works with both ``aparc.annot`` and ``aparc.DKTatlas.annot``
(Desikan-Killiany-Tourville).  The mapping follows standard
neuroanatomical conventions.

Examples
--------
>>> labels, ctab, names = sb.io.load_freesurfer_annot("lh.aparc.annot")
>>> lobe_labels, lobe_names = sb.io.remap_parcellation(
...     labels, names, DESIKAN_LOBE_MAP,
... )
"""

SCHAEFER_NETWORK_MAP: dict[str, str] = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "DorsalAttention",
    "SalVentAttn": "SalVentAttn",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
    "TempPar": "TempPar",
}
"""Map from Schaefer parcel network prefixes to Yeo 7/17 network names.

Used with :func:`remap_parcellation` when ``match="contains"``
to group Schaefer-200/400/600/800/1000 parcels by their parent
network.

Examples
--------
>>> labels, ctab, names = sb.io.load_freesurfer_annot(
...     "lh.Schaefer2018_200Parcels_7Networks_order.annot"
... )
>>> net_labels, net_names = sb.io.remap_parcellation(
...     labels, names, SCHAEFER_NETWORK_MAP, match="contains",
... )
"""


def remap_parcellation(
    labels: np.ndarray,
    names: Sequence[str | bytes],
    mapping: dict[str, str],
    *,
    match: Literal["exact", "contains"] = "exact",
    unmapped: str = "unmapped",
) -> tuple[np.ndarray, dict[int, str]]:
    """Remap per-vertex parcellation labels to a coarser grouping.

    Takes the label array and region names from a FreeSurfer ``.annot``
    (or any atlas) and remaps each region to a new group according to
    *mapping*.

    Parameters
    ----------
    labels : ndarray, shape (V,)
        Per-vertex integer labels (as returned by
        :func:`load_freesurfer_annot`).
    names : sequence of str or bytes
        Region names, one per unique label in the annotation colour
        table.  Index *i* corresponds to label integer *i*.
    mapping : dict of {str: str}
        Source region name → target group name.
    match : ``"exact"`` or ``"contains"``
        How to match region names to mapping keys.

        * ``"exact"`` — the region name (lowered, stripped) must equal
          a mapping key.
        * ``"contains"`` — the region name is assigned to the first
          mapping key that appears as a substring.  Useful for
          Schaefer parcels whose names embed the network prefix
          (e.g. ``"7Networks_LH_Vis_1"`` matches key ``"Vis"``).
    unmapped : str
        Group name for regions that do not match any mapping key.

    Returns
    -------
    new_labels : ndarray, shape (V,)
        Remapped per-vertex labels (contiguous integers starting at 0
        for *unmapped*, 1 for the first group, etc.).
    new_names : dict of {int: str}
        Mapping from new integer label to group name.

    Examples
    --------
    Group Desikan parcels into lobes:

    >>> labels, ctab, names = sb.io.load_freesurfer_annot("lh.aparc.annot")
    >>> lobe_labels, lobe_names = sb.io.remap_parcellation(
    ...     labels, names, sb.io.DESIKAN_LOBE_MAP,
    ... )
    >>> set(lobe_names.values())
    {'frontal', 'parietal', 'temporal', 'occipital', 'insular', 'unmapped'}

    Group Schaefer parcels into Yeo networks:

    >>> labels, _, names = sb.io.load_freesurfer_annot(
    ...     "lh.Schaefer2018_400Parcels_7Networks_order.annot"
    ... )
    >>> net_labels, net_names = sb.io.remap_parcellation(
    ...     labels, names, sb.io.SCHAEFER_NETWORK_MAP, match="contains",
    ... )
    """
    labels = np.asarray(labels, dtype=np.int64)

    # Decode bytes names.
    decoded: list[str] = []
    for n in names:
        decoded.append(n.decode("utf-8") if isinstance(n, bytes) else str(n))

    # Build old-label → group-name map.
    label_to_group: dict[int, str] = {}
    mapping_lower = {k.lower().strip(): v for k, v in mapping.items()}

    for idx, name in enumerate(decoded):
        name_clean = name.lower().strip()
        if match == "exact":
            label_to_group[idx] = mapping_lower.get(name_clean, unmapped)
        else:
            found = False
            for key, group in mapping.items():
                if key.lower() in name_clean or key in name:
                    label_to_group[idx] = group
                    found = True
                    break
            if not found:
                label_to_group[idx] = unmapped

    # Build contiguous integer labels for groups.
    unique_groups = sorted(set(label_to_group.values()))
    # Ensure "unmapped" is index 0.
    if unmapped in unique_groups:
        unique_groups.remove(unmapped)
        unique_groups = [unmapped, *unique_groups]
    group_to_int: dict[str, int] = {g: i for i, g in enumerate(unique_groups)}

    new_labels = np.array(
        [group_to_int[label_to_group.get(l, unmapped)] for l in labels],
        dtype=np.int64,
    )
    new_names = {v: k for k, v in group_to_int.items()}

    logger.info(
        "Remapped %d labels → %d groups (%s)",
        len(set(labels.tolist())),
        len(unique_groups),
        ", ".join(unique_groups),
    )
    return new_labels, new_names


def aggregate_by_parcellation(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    stat: str | Callable = "mean",
    ignore_labels: list[int] | None = None,
    label_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Aggregate vertex-wise data per parcellation region.

    Compute summary statistics of a vertex-wise array (e.g. thickness,
    HKS, z-score) within each parcel defined by *labels*.

    Parameters
    ----------
    data : ndarray, shape (V,) or (V, D)
        Vertex-wise data.  If 2-D, each column is aggregated
        independently.
    labels : ndarray, shape (V,)
        Per-vertex integer labels.
    stat : str or callable
        Aggregation function.  Built-in options: ``"mean"``,
        ``"median"``, ``"std"``, ``"min"``, ``"max"``, ``"sum"``,
        ``"count"``, ``"iqr"`` (interquartile range).  Or pass a
        callable that accepts an array and returns a scalar.
    ignore_labels : list of int, optional
        Labels to exclude (e.g. ``[0]`` for medial wall).
    label_names : dict of {int: str}, optional
        Mapping from label integer to name.  If provided, the
        returned DataFrame uses region names as the index.

    Returns
    -------
    pandas.DataFrame
        One row per parcel, columns are ``"label"`` (or region name)
        plus one column per data dimension (``"d0"``, ``"d1"``, …
        or ``"value"`` for 1-D input).

    Examples
    --------
    Mean cortical thickness per Desikan region:

    >>> thickness = sb.io.load_freesurfer_morph("lh.thickness")
    >>> labels, _, names = sb.io.load_freesurfer_annot("lh.aparc.annot")
    >>> df = sb.io.aggregate_by_parcellation(
    ...     thickness, labels, stat="mean", ignore_labels=[0],
    ... )
    >>> df.head()

    Mean HKS per Yeo network (after remapping):

    >>> hks = sb.spectral.compute_hks(decomp, n_times=16)
    >>> net_labels, net_names = sb.io.remap_parcellation(
    ...     labels, names, sb.io.SCHAEFER_NETWORK_MAP, match="contains",
    ... )
    >>> df = sb.io.aggregate_by_parcellation(
    ...     hks, net_labels, stat="mean",
    ...     ignore_labels=[0], label_names=net_names,
    ... )
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas required: pip install pandas") from exc

    data = np.asarray(data)
    labels = np.asarray(labels, dtype=np.int64)

    if data.ndim == 1:
        data = data[:, np.newaxis]
        col_names = ["value"]
    else:
        col_names = [f"d{i}" for i in range(data.shape[1])]

    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"data rows {data.shape[0]} != labels length {labels.shape[0]}")

    ignore = set(ignore_labels or [])
    unique_labels = sorted(set(np.unique(labels).tolist()) - ignore)

    # Resolve aggregation function.
    _stat_funcs: dict[str, Callable] = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "std": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
        "sum": np.nansum,
        "count": lambda x, axis=0: np.sum(~np.isnan(x), axis=axis),
        "iqr": lambda x, axis=0: (
            np.nanpercentile(x, 75, axis=axis) - np.nanpercentile(x, 25, axis=axis)
        ),
    }
    if isinstance(stat, str):
        if stat not in _stat_funcs:
            raise ValueError(
                f"Unknown stat '{stat}'. Use one of {list(_stat_funcs.keys())} or pass a callable."
            )
        func = _stat_funcs[stat]
    else:
        func = stat

    rows = []
    for lab in unique_labels:
        mask = labels == lab
        region_data = data[mask]
        agg = func(region_data, axis=0)
        agg = np.atleast_1d(agg)
        rows.append([lab, *agg.tolist()])

    df = pd.DataFrame(rows, columns=["label", *col_names])

    if label_names is not None:
        df["region"] = df["label"].map(label_names).fillna("unknown")
        df = df.set_index("region")
    else:
        df = df.set_index("label")

    return df


# ======================================================================
# §6  __all__
# ======================================================================

__all__: list[str] = [
    "DESIKAN_LOBE_MAP",
    "SCHAEFER_NETWORK_MAP",
    "aggregate_by_parcellation",
    "apply_parcellation",
    "detect_format",
    # Parcellation
    "extract_submesh",
    # Volumetric → point cloud
    "labels_to_pointcloud",
    # Auto-detection
    "load",
    "load_freesurfer_annot",
    "load_freesurfer_morph",
    # FreeSurfer
    "load_freesurfer_surface",
    "load_gifti_func",
    "load_gifti_label",
    # GIfTI
    "load_gifti_surface",
    # Generic mesh
    "load_mesh",
    # NIfTI / MGZ
    "load_nifti",
    # Remapping & aggregation
    "remap_parcellation",
]
