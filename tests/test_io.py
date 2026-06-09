"""Regression tests for the file loaders in :mod:`spectralbrain.io`.

These guard the input pathways that a default install must support. In
particular, generic meshes (.ply / .obj / .stl / .vtk / .vtp) are read
through PyVista — a *core* dependency — so every listed format must work
without any optional extra. The VTK/VTP cases are explicit regression
guards: an earlier version routed them through an optional package that
could not actually read VTK, leaving the declared format broken.
"""

import numpy as np
import pytest

import spectralbrain as sb


# ----------------------------------------------------------------------
# Synthetic geometry (small icosphere, built directly in PyVista format)
# ----------------------------------------------------------------------
def _small_mesh():
    """Return (vertices, faces) of a coarse unit icosahedron."""
    phi = (1.0 + 5.0**0.5) / 2.0
    v = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return v, f


# ----------------------------------------------------------------------
# Generic meshes via PyVista
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ext", ["ply", "obj", "stl", "vtk", "vtp"])
def test_load_generic_mesh_roundtrip(tmp_path, ext):
    """Every PyVista-backed mesh format round-trips to (N, 3) / (F, 3)."""
    pv = pytest.importorskip("pyvista")
    v, f = _small_mesh()
    faces_pv = np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64).ravel()
    mesh = pv.PolyData(v, faces_pv)

    path = tmp_path / f"mesh.{ext}"
    mesh.save(str(path))

    result = sb.io.load(str(path))
    verts, faces = result["vertices"], result["faces"]

    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert verts.shape[0] == len(v)
    assert faces.shape[0] >= 1
    # All face indices must reference valid vertices.
    assert faces.max() < verts.shape[0]
    assert faces.min() >= 0


def test_load_mesh_helper_matches_load(tmp_path):
    """``load_mesh`` returns the same geometry as the generic dispatcher."""
    pv = pytest.importorskip("pyvista")
    v, f = _small_mesh()
    faces_pv = np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64).ravel()
    path = tmp_path / "mesh.ply"
    pv.PolyData(v, faces_pv).save(str(path))

    verts, faces = sb.io.load_mesh(str(path))
    bundle = sb.io.load(str(path))
    np.testing.assert_allclose(verts, bundle["vertices"])
    np.testing.assert_array_equal(faces, bundle["faces"])


# ----------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------
def test_load_npz_roundtrip(tmp_path):
    """A NumPy archive exposes its arrays plus the format tag."""
    v, f = _small_mesh()
    path = tmp_path / "mesh.npz"
    np.savez(str(path), vertices=v, faces=f)

    result = sb.io.load(str(path))
    assert "vertices" in result and "faces" in result
    np.testing.assert_allclose(result["vertices"], v)


def test_load_hdf5_roundtrip(tmp_path):
    """An HDF5 cache exposes its datasets plus the format tag."""
    h5py = pytest.importorskip("h5py")
    v, f = _small_mesh()
    path = tmp_path / "cache.h5"
    with h5py.File(str(path), "w") as fh:
        fh["vertices"] = v
        fh["faces"] = f

    result = sb.io.load(str(path))
    assert "vertices" in result and "faces" in result
    np.testing.assert_allclose(np.asarray(result["vertices"]), v)


# ----------------------------------------------------------------------
# Export round-trip
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ext", ["ply", "obj", "stl", "vtk", "vtp"])
def test_save_mesh_roundtrip(tmp_path, ext):
    """``save_mesh`` writes every PyVista format and ``load_mesh`` reads it back."""
    pytest.importorskip("pyvista")
    v, f = _small_mesh()
    path = tmp_path / f"out.{ext}"

    returned = sb.io.save_mesh(str(path), v, f)
    assert returned.exists()

    verts, faces = sb.io.load_mesh(str(path))
    assert verts.shape == v.shape
    assert faces.shape[1] == 3
    assert faces.shape[0] == f.shape[0]


# ----------------------------------------------------------------------
# Format auto-detection
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mesh.ply", "PLY"),
        ("mesh.obj", "OBJ"),
        ("mesh.stl", "STL"),
        ("mesh.vtk", "VTK"),
        ("mesh.vtp", "VTK"),
        ("data.surf.gii", "GIFTI_SURFACE"),
        ("data.func.gii", "GIFTI_FUNC"),
        ("vol.nii.gz", "NIFTI_VOLUME"),
        ("vol.mgz", "MGZ_VOLUME"),
        ("lh.aparc.annot", "FREESURFER_ANNOT"),
        ("lh.thickness", "FREESURFER_MORPH"),
        ("cache.h5", "HDF5"),
        ("mesh.npz", "NUMPY"),
    ],
)
def test_detect_format_by_extension(name, expected):
    """Extension-based detection maps to the right GeometryFormat."""
    from spectralbrain.io.loaders import detect_format

    assert detect_format(name).name == expected
