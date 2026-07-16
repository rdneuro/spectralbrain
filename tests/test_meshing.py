"""Tests for :mod:`spectralbrain.io.meshing` (volume → LBO-ready mesh).

The improvement pipeline needs ``trimesh`` (core) and, for remeshing,
``pyacvd``; tests ``importorskip`` those so a minimal install skips rather than
fails. Correctness is anchored on a solid ball: its improved surface must be a
closed, single-component, genus-0 2-manifold, and marching-cubes parity
(``raw=True``) must reproduce :func:`spectralbrain.core.marching_cubes` exactly.
"""

import numpy as np
import pytest

import spectralbrain as sb
from spectralbrain.core import marching_cubes

trimesh = pytest.importorskip("trimesh")


LABEL = 17  # arbitrary integer label used to tag the ball


def _ball_label_volume(radius_vox: int = 12, pad: int = 8, label: int = LABEL):
    """Solid ball as an integer-label volume + 1 mm isotropic affine."""
    n = 2 * (radius_vox + pad)
    c = n / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    inside = (xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2 <= radius_vox ** 2
    vol = np.zeros((n, n, n), dtype=np.int16)
    vol[inside] = label
    affine = np.eye(4)
    affine[:3, 3] = -c  # centre the ball on the world origin
    return vol, affine


def _min_angle_median(verts, faces):
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    la = np.linalg.norm(c - b, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(b - a, axis=1)

    def ang(o1, o2, opp):
        cosv = (o1 ** 2 + o2 ** 2 - opp ** 2) / (2 * o1 * o2 + 1e-12)
        return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

    angles = np.stack([ang(lb, lc, la), ang(la, lc, lb), ang(la, lb, lc)], axis=1)
    return float(np.median(angles.min(axis=1)))


def test_raw_matches_core_marching_cubes():
    """raw=True reproduces the legacy plain marching-cubes mesh exactly."""
    vol, affine = _ball_label_volume()
    mask = (vol == LABEL).astype(np.float64)
    exp_v, exp_f = marching_cubes(mask, affine, level=0.5)

    got_v, got_f = sb.volume_to_mesh(vol, affine, raw=True, label=LABEL)
    np.testing.assert_allclose(got_v, exp_v)
    np.testing.assert_array_equal(got_f, exp_f)


def test_improved_is_closed_manifold_genus0():
    """The improved closed-structure mesh is watertight, single-component, genus 0."""
    pytest.importorskip("pyacvd")
    vol, affine = _ball_label_volume()
    v, f, info = sb.volume_to_mesh(vol, affine, label=LABEL, return_info=True)

    m = trimesh.Trimesh(v, f, process=False)
    assert m.is_watertight
    assert m.is_winding_consistent
    assert len(m.split(only_watertight=False)) == 1
    # genus 0 for a topological sphere: V - E + F = 2
    euler = len(m.vertices) - len(np.unique(np.sort(m.edges, axis=1), axis=0)) + len(m.faces)
    assert euler == 2
    assert info["method"] == "improved"


def test_improved_beats_raw_on_triangle_quality():
    """Remeshing raises the median minimum triangle angle vs raw marching cubes."""
    pytest.importorskip("pyacvd")
    vol, affine = _ball_label_volume()
    rv, rf = sb.volume_to_mesh(vol, affine, raw=True, label=LABEL)
    iv, if_ = sb.volume_to_mesh(vol, affine, label=LABEL)
    assert _min_angle_median(iv, if_) > _min_angle_median(rv, rf)


def test_from_volume_decomposes_like_a_sphere():
    """BrainMesh.from_volume yields a mesh whose LBO spectrum starts near 0."""
    pytest.importorskip("pyacvd")
    vol, affine = _ball_label_volume(radius_vox=12)
    mesh = sb.BrainMesh.from_volume(vol, affine, label=LABEL)
    assert mesh.is_closed()
    decomp = mesh.decompose(k=6)
    evals = np.sort(decomp.eigenvalues)
    # first eigenvalue ~ 0 (constant eigenfunction)
    assert abs(evals[0]) < 1e-2 * (evals[-1] + 1e-9)
    # sphere of radius r: first non-zero eigenvalue ~ 2 / r^2 (r = 12 mm here)
    lam1 = evals[1]
    assert 0.5 * (2 / 12.0 ** 2) < lam1 < 2.0 * (2 / 12.0 ** 2)


def test_refine_mesh_improves_raw_surface():
    """refine_mesh cleans a raw surface (closed=True → watertight genus-0)."""
    pytest.importorskip("pyacvd")
    vol, affine = _ball_label_volume()
    rv, rf = sb.volume_to_mesh(vol, affine, raw=True, label=LABEL)
    v, f = sb.refine_mesh(rv, rf, closed=True)
    m = trimesh.Trimesh(v, f, process=False)
    assert m.is_watertight
    assert _min_angle_median(v, f) > _min_angle_median(rv, rf)


def test_open_path_does_not_force_watertight():
    """closed=False keeps components and does not force a watertight surface."""
    # two separate balls -> a legitimately multi-component (open-handling) mask
    vol, affine = _ball_label_volume(radius_vox=8, pad=6)
    n = vol.shape[0]
    vol2 = np.zeros((n, n + 20, n), dtype=np.int16)
    vol2[:, :n, :] = vol
    vol2[:, 20:, :] = np.maximum(vol2[:, 20:, :], vol)  # second ball, shifted
    v, f, info = sb.volume_to_mesh(
        vol2, affine, label=LABEL, closed=False, remesh=False, return_info=True
    )
    assert info["closed"] is False
    assert info["keep_largest"] is False
    m = trimesh.Trimesh(v, f, process=False)
    # both balls survive: more than one connected component
    assert len(m.split(only_watertight=False)) >= 2


def test_empty_mask_raises():
    vol, affine = _ball_label_volume()
    with pytest.raises(ValueError):
        sb.volume_to_mesh(vol, affine, label=999)  # label absent -> empty mask


def test_bad_shapes_raise():
    _, affine = _ball_label_volume()
    with pytest.raises(ValueError):
        sb.volume_to_mesh(np.zeros((4, 4)), affine)          # not 3-D
    with pytest.raises(ValueError):
        sb.volume_to_mesh(np.zeros((4, 4, 4)), np.eye(3))    # bad affine
