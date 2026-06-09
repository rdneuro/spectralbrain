"""Export spectral decompositions, meshes, scalar maps, and connectomes.

All export functions accept the canonical type aliases from
:mod:`spectralbrain.runtime` and write to standard neuroimaging or
geometry formats.  The primary cache format is HDF5 (via h5py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectralbrain.runtime import (
    ConnectomeMatrix,
    DescriptorMatrix,
    Eigenvalues,
    Eigenvectors,
    Faces,
    PathLike,
    ScalarMap,
    Vertices,
    get_logger,
)

logger = get_logger(__name__)


def _require_h5py():
    """Lazy-import h5py for HDF5 I/O."""
    try:
        import h5py

        return h5py
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 export.\n  pip install h5py") from exc


def _require_nibabel():
    """Lazy-import nibabel for neuroimaging I/O."""
    try:
        import nibabel as nib

        return nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for neuroimaging export.\n  pip install nibabel"
        ) from exc


# ======================================================================
# §1  HDF5 CACHE  (SpectralDecomposition persistence)
# ======================================================================


def save_hdf5(
    path: PathLike,
    *,
    eigenvalues: Eigenvalues | None = None,
    eigenvectors: Eigenvectors | None = None,
    vertices: Vertices | None = None,
    faces: np.ndarray | None = None,
    descriptors: dict[str, np.ndarray] | None = None,
    metadata: dict[str, Any] | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Path:
    """Save spectral decomposition and descriptors to HDF5.

    This is the primary caching mechanism.  A full eigendecomposition
    for a 160 k-vertex cortical surface (~300 eigenpairs) takes minutes;
    saving to HDF5 allows instant reload.

    Parameters
    ----------
    path : PathLike
        Output ``.h5`` file.
    eigenvalues : ndarray, shape (k,), optional
    eigenvectors : ndarray, shape (N, k), optional
    vertices : ndarray, shape (N, 3), optional
    faces : ndarray, shape (F, 3), optional
    descriptors : dict of {str: ndarray}, optional
        Named descriptor arrays (e.g. ``{"hks": hks_matrix}``).
    metadata : dict, optional
        Scalar metadata stored as HDF5 attributes (version, atlas,
        subject ID, structure name, backend used, …).
    compression : str
        HDF5 compression filter.
    compression_opts : int
        Compression level (1–9).

    Returns
    -------
    Path
        The written file path.

    Examples
    --------
    >>> sb.io.export.save_hdf5(
    ...     "sub-01_lh_white_spectral.h5",
    ...     eigenvalues=evals,
    ...     eigenvectors=evecs,
    ...     vertices=verts,
    ...     faces=faces,
    ...     descriptors={"hks": hks, "wks": wks},
    ...     metadata={"subject": "sub-01", "hemi": "lh",
    ...               "n_eigenvalues": 100},
    ... )
    """
    h5py = _require_h5py()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kw = dict(compression=compression, compression_opts=compression_opts)

    with h5py.File(str(out), "w") as f:
        if eigenvalues is not None:
            f.create_dataset("eigenvalues", data=eigenvalues, **kw)
        if eigenvectors is not None:
            f.create_dataset("eigenvectors", data=eigenvectors, **kw)
        if vertices is not None:
            f.create_dataset("vertices", data=vertices, **kw)
        if faces is not None:
            f.create_dataset("faces", data=faces, **kw)

        if descriptors:
            grp = f.create_group("descriptors")
            for name, arr in descriptors.items():
                grp.create_dataset(name, data=arr, **kw)

        # Store metadata as root attributes.
        if metadata:
            for key, val in metadata.items():
                f.attrs[key] = val

        # Always stamp the SpectralBrain version.
        from spectralbrain.runtime import __version__

        f.attrs["spectralbrain_version"] = __version__

    logger.info("Saved HDF5 → %s", out)
    return out


def load_hdf5(path: PathLike) -> dict[str, Any]:
    """Load a SpectralBrain HDF5 cache file.

    Parameters
    ----------
    path : PathLike

    Returns
    -------
    dict
        Keys mirror what was passed to :func:`save_hdf5`.
    """
    h5py = _require_h5py()
    result: dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        for key in ("eigenvalues", "eigenvectors", "vertices", "faces"):
            if key in f:
                result[key] = np.asarray(f[key])
        if "descriptors" in f:
            result["descriptors"] = {k: np.asarray(v) for k, v in f["descriptors"].items()}
        result["metadata"] = dict(f.attrs)
    return result


# ======================================================================
# §2  MESH EXPORT
# ======================================================================


def save_mesh(
    path: PathLike,
    vertices: Vertices,
    faces: Faces,
) -> Path:
    """Save a mesh to .ply, .obj, .stl, .vtk, or .vtp.

    Backed by PyVista (a core dependency), so every listed format works
    with a default install and the geometry round-trips with
    :func:`spectralbrain.io.load_mesh`.

    Parameters
    ----------
    path : PathLike
        Output file — format inferred from extension.
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    Path
    """
    import pyvista as pv

    out = Path(path)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    # PyVista packs faces as [3, i, j, k, 3, i, j, k, ...].
    faces_pv = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    mesh = pv.PolyData(v, faces_pv)
    mesh.save(str(out))
    logger.info("Saved mesh → %s", out)
    return out


# ======================================================================
# §3  GIFTI SCALAR OVERLAY EXPORT
# ======================================================================


def save_gifti_func(
    path: PathLike,
    scalars: ScalarMap | DescriptorMatrix,
) -> Path:
    """Save a scalar map or descriptor matrix as .func.gii.

    Parameters
    ----------
    path : PathLike
        Output ``.func.gii``.
    scalars : ndarray, shape (N,) or (N, T)
        Scalar overlay(s).

    Returns
    -------
    Path
    """
    nib = _require_nibabel()
    out = Path(path)
    scalars = np.asarray(scalars, dtype=np.float32)

    darrays = []
    if scalars.ndim == 1:
        scalars = scalars[:, np.newaxis]
    for col in range(scalars.shape[1]):
        da = nib.gifti.GiftiDataArray(
            data=scalars[:, col],
            intent="NIFTI_INTENT_SHAPE",
            datatype="NIFTI_TYPE_FLOAT32",
        )
        darrays.append(da)

    img = nib.gifti.GiftiImage(darrays=darrays)
    nib.save(img, str(out))
    logger.info("Saved GIfTI func → %s", out)
    return out


# ======================================================================
# §4  NUMPY ARCHIVE EXPORT
# ======================================================================


def save_npz(
    path: PathLike,
    **arrays: np.ndarray,
) -> Path:
    """Save named arrays to a compressed .npz archive.

    Parameters
    ----------
    path : PathLike
        Output ``.npz``.
    **arrays
        Keyword → ndarray pairs.

    Returns
    -------
    Path
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **arrays)
    logger.info("Saved npz → %s", out)
    return out


# ======================================================================
# §5  CONNECTOME MATRIX EXPORT
# ======================================================================


def save_connectome(
    path: PathLike,
    matrix: ConnectomeMatrix,
    *,
    labels: list[str] | None = None,
) -> Path:
    """Save a connectome matrix to .tsv (BIDS-compatible).

    Parameters
    ----------
    path : PathLike
        Output ``.tsv``.
    matrix : ndarray, shape (R, R)
    labels : list of str, optional
        Region names for the header row/column.

    Returns
    -------
    Path
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix)

    header = ""
    if labels is not None:
        header = "\t".join(["", *labels]) + "\n"

    with open(out, "w") as fh:
        fh.write(header)
        for i in range(matrix.shape[0]):
            row_label = labels[i] if labels else str(i)
            row_vals = "\t".join(f"{v:.6f}" for v in matrix[i])
            fh.write(f"{row_label}\t{row_vals}\n")

    logger.info("Saved connectome → %s", out)
    return out


# ======================================================================

__all__ = [
    "load_hdf5",
    "save_connectome",
    "save_gifti_func",
    "save_hdf5",
    "save_mesh",
    "save_npz",
]
