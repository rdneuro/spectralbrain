"""Regression tests for bugs fixed in core/ (incl. backends), runtime and utils/."""

from __future__ import annotations

import logging
import sys
import types
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import ConvexHull

import spectralbrain as sb
from spectralbrain.core import meshes as meshes_mod
from spectralbrain.core.base import marching_cubes
from spectralbrain.core.meshes import BrainMesh, _voronoi_areas, weld_mesh

# -- fixtures -----------------------------------------------------------


def _fib_sphere(n: int = 1500, r: float = 1.0) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _sphere_mesh(n: int = 1500, r: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Closed sphere mesh with outward (CCW) winding."""
    v = _fib_sphere(n, r)
    f = ConvexHull(v).simplices.copy()
    c = v[f].mean(axis=1)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    inward = np.sum(fn * c, axis=1) < 0
    f[inward] = f[inward][:, ::-1]
    return v, f


def _signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    return float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6)


# -- #2 mesh Laplacian cache --------------------------------------------


def test_mesh_decompose_rebuilds_on_method_change(monkeypatch):
    v, f = _sphere_mesh(400)

    def fake_robust(vertices, faces, *, mollify_factor=1e-5):
        L, M = meshes_mod._cotangent_laplacian(vertices, faces)
        return 2.0 * L, M

    monkeypatch.setattr(meshes_mod, "_robust_laplacian_mesh", fake_robust)
    m = BrainMesh(v, f)
    m.mean_curvature()  # builds and caches the cotangent Laplacian
    cot = m.decompose(k=5)
    rob = m.decompose(k=5, laplacian_method="robust")
    assert cot.metadata["laplacian_method"] == "cotangent"
    assert rob.metadata["laplacian_method"] == "robust"
    np.testing.assert_allclose(rob.eigenvalues[1:], 2 * cot.eigenvalues[1:], rtol=1e-6)


def test_mesh_vertex_assignment_invalidates_cache():
    v, f = _sphere_mesh(400, r=1.0)
    m = BrainMesh(v, f)
    a = m.surface_area()
    m.vertices = 2 * v
    assert m.surface_area() == pytest.approx(4 * a)
    lam = m.decompose(k=4).eigenvalues[1]
    assert lam == pytest.approx(2.0 / 4.0, rel=0.05)


# -- #3 / #4 shape index range and signed mean curvature ---------------


def test_shape_index_in_range_and_mean_curvature_signed():
    v, f = _sphere_mesh(1500, r=10.0)
    m = BrainMesh(v, f)
    H = m.mean_curvature()
    assert np.median(H) == pytest.approx(0.1, rel=0.02)
    si = m.shape_index()
    assert si.min() >= -1 - 1e-12 and si.max() <= 1 + 1e-12
    assert np.median(si) == pytest.approx(1.0, abs=0.05)

    flipped = BrainMesh(v, f[:, ::-1].copy())
    assert np.median(flipped.mean_curvature()) == pytest.approx(-0.1, rel=0.02)
    assert np.median(flipped.shape_index()) == pytest.approx(-1.0, abs=0.05)


# -- #5 degenerate triangles --------------------------------------------


def test_degenerate_faces_do_not_corrupt_spectrum():
    v, f = _sphere_mesh(800, r=1.0)
    ref = BrainMesh(v, f).decompose(k=5).eigenvalues
    # add a zero-area sliver (collinear: repeated vertex) and a repeated-index face
    f_bad = np.vstack([f, [[f[0, 0], f[0, 1], f[0, 0]], [f[1, 0], f[1, 0], f[1, 1]]]])
    got = BrainMesh(v, f_bad).decompose(k=5).eigenvalues
    np.testing.assert_allclose(got, ref, atol=1e-8)
    L, _M = meshes_mod._cotangent_laplacian(v, f_bad)
    assert abs(L).max() < 1e6


def test_example_sphere_is_welded_and_spectrum_correct():
    v, f = sb.utils.example_sphere(20, 40, 1.0)
    assert len(np.unique(np.round(v, 8), axis=0)) == len(v)
    lam = BrainMesh(v, f).decompose(k=5).eigenvalues
    np.testing.assert_allclose(lam[1:4], 2.0, rtol=0.05)


def test_weld_mesh_merges_duplicates():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0.0]])
    f = np.array([[0, 1, 2], [0, 3, 2], [0, 1, 1]])
    vw, fw = weld_mesh(v, f)
    assert len(vw) == 3 and len(fw) == 1


# -- #10 curvature / areas ----------------------------------------------


def test_mixed_voronoi_areas_sum_to_area_and_differ_from_barycentric():
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(size=(60, 2)), np.zeros(60)]
    from scipy.spatial import Delaunay

    f = Delaunay(pts[:, :2]).simplices
    m = BrainMesh(pts, f)
    vor = m.vertex_areas("voronoi")
    bary = m.vertex_areas("barycentric")
    assert vor.sum() == pytest.approx(m.surface_area())
    assert not np.allclose(vor, bary)


def test_gaussian_curvature_flat_open_patch_is_zero_off_corners():
    xx, yy = np.meshgrid(np.arange(5.0), np.arange(5.0))
    P = np.c_[xx.ravel(), yy.ravel(), np.zeros(25)]
    from scipy.spatial import Delaunay

    K = BrainMesh(P, Delaunay(P[:, :2]).simplices).gaussian_curvature()
    corners = [0, 4, 20, 24]
    mask = np.ones(25, bool)
    mask[corners] = False
    np.testing.assert_allclose(K[mask], 0.0, atol=1e-10)


def test_heat_geodesic_matches_great_circle():
    v, f = _sphere_mesh(1500, r=10.0)
    d = BrainMesh(v, f).geodesic_distance(np.array([0]))
    true = 10 * np.arccos(np.clip(v @ v[0] / 100, -1, 1))
    assert np.corrcoef(d, true)[0, 1] > 0.999


def test_smoothing_does_not_alias_metadata():
    v, f = _sphere_mesh(200)
    m = BrainMesh(v, f, metadata={"structure": "x"})
    s = m.laplacian_smooth(n_iterations=1)
    s.metadata["structure"] = "y"
    assert m.metadata["structure"] == "x"


# -- #17 marching cubes -------------------------------------------------


def _ball(label: int = 1) -> np.ndarray:
    zz, yy, xx = np.mgrid[0:20, 0:20, 0:20]
    return np.where(((xx - 10) ** 2 + (yy - 9) ** 2 + (zz - 10) ** 2) <= 25, label, 0)


@pytest.mark.parametrize(
    "affine",
    [
        np.eye(4),
        np.array([[-1.0, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128], [0, 0, 0, 1]]),  # LIA
    ],
)
def test_marching_cubes_outward_winding(affine):
    pytest.importorskip("skimage")
    v, f = marching_cubes(_ball(), affine)
    assert _signed_volume(v, f) > 0


def test_marching_cubes_binarises_labels():
    pytest.importorskip("skimage")
    v1, f1 = marching_cubes(_ball(1), np.eye(4))
    v17, f17 = marching_cubes(_ball(17), np.eye(4))
    np.testing.assert_allclose(v1, v17)
    np.testing.assert_array_equal(f1, f17)


# -- #6 / #21 atlas -----------------------------------------------------


def test_schaefer_to_yeo_exact_with_names_and_warns_without():
    names = [f"7Networks_LH_Vis_{i}" for i in range(1, 10)] + ["7Networks_LH_SomMot_1"]
    assert sb.utils.schaefer_to_yeo(8, 100, 7, parcel_names=names) == "Visual"
    assert sb.utils.schaefer_to_yeo(10, 100, 7, parcel_names=names) == "Somatomotor"
    with pytest.warns(UserWarning):
        sb.utils.schaefer_to_yeo(8, 100, 7)


def test_get_label_id_exact_and_ambiguous():
    from spectralbrain.utils import get_label_id, get_structure_ids

    assert get_label_id("aseg", "Right-Hippocampus") == 53
    with pytest.warns(UserWarning):
        assert get_label_id("aseg", "Hippocampus") == 17
    right = get_structure_ids("hippocampal_subfields", "right")
    assert right == get_structure_ids("hippocampal_subfields", "left") and right
    assert all(i < 1000 for i in right)


# -- #9 logging ---------------------------------------------------------


def test_set_log_level_controls_module_loggers():
    from spectralbrain.runtime import set_log_level

    child = logging.getLogger("spectralbrain.core.meshes")
    try:
        set_log_level("DEBUG")
        assert child.isEnabledFor(logging.DEBUG)
        set_log_level("ERROR")
        assert not child.isEnabledFor(logging.WARNING)
    finally:
        set_log_level("INFO")


# -- #13 / #16 reproducibility ------------------------------------------


def test_seed_everything_makes_library_rngs_deterministic():
    from spectralbrain.runtime import resolve_seed, set_global_seed

    try:
        sb.seed_everything(7)
        a = np.random.default_rng(resolve_seed(None)).integers(0, 10**9, size=10)
        sb.seed_everything(7)
        b = np.random.default_rng(resolve_seed(None)).integers(0, 10**9, size=10)
        assert resolve_seed(None) == 7
        np.testing.assert_array_equal(a, b)
    finally:
        set_global_seed(None)


def test_reproducibility_version_matches_package():
    assert sb.get_reproducibility_info()["spectralbrain"] == sb.__version__


# -- #15 container integrity --------------------------------------------


def test_container_placeholder_digest_warns(tmp_path):
    from spectralbrain.runtime import ContainerManager, ContainerSpec

    spec = ContainerSpec("Fake", "fake.sif", "https://example.invalid", "placeholder", 1, "x")
    cm = ContainerManager(cache_dir=tmp_path, registry={"fake": spec})
    (tmp_path / "fake.sif").write_bytes(b"data")
    with pytest.warns(UserWarning, match="CANNOT be verified"):
        cm.ensure("fake")


def test_container_cached_digest_mismatch_raises(tmp_path):
    from spectralbrain.runtime import ContainerManager, ContainerSpec

    spec = ContainerSpec("Fake", "fake.sif", "https://example.invalid", "0" * 64, 1, "x")
    cm = ContainerManager(cache_dir=tmp_path, registry={"fake": spec})
    (tmp_path / "fake.sif").write_bytes(b"data")
    with pytest.raises(RuntimeError, match="SHA-256"):
        cm.ensure("fake")


# -- #7 GPU fallback helpers (no GPU deps needed) -----------------------


def test_gpu_fallback_reason_and_diagonal_check():
    from spectralbrain.core.backends import _cpu_fallback_reason, _is_diagonal

    D = sp.diags(np.ones(5))
    assert _is_diagonal(D)
    assert not _is_diagonal(D + sp.eye(5, k=1))
    assert _cpu_fallback_reason(10, D, 20000) is None
    assert "dense_max" in _cpu_fallback_reason(30000, D, 20000)
    assert "not diagonal" in _cpu_fallback_reason(10, D + sp.eye(5, k=1), 20000)


# -- #18 / #19 / #20 small utilities ------------------------------------


def test_nutpie_sampler_forwards_config_and_overrides(monkeypatch):
    from spectralbrain.statistics import bayesian as cpu

    seen: dict = {}
    fake = types.SimpleNamespace(
        compile_pymc_model=lambda m: m,
        sample=lambda compiled, **kw: seen.update(kw) or "trace",
    )
    monkeypatch.setattr(cpu, "_require_nutpie", lambda: fake)
    s = cpu.NutpieSampler(cpu.SamplerConfig(draws=10, cores=3, target_accept=0.9))
    assert s.sample(object(), draws=5) == "trace"
    assert seen["draws"] == 5 and seen["cores"] == 3 and seen["target_accept"] == 0.9


def test_memoryinfo_repr_has_no_docstring_text():
    from spectralbrain.runtime import MemoryInfo

    r = repr(MemoryInfo(16.0, 8.0, 8.0, 50.0))
    assert "Return" not in r and "50% used" in r


def test_odd_subject_counts_give_matching_labels():
    d = sb.utils.make_connectome_example(n_subjects=41)
    assert d["labels"].shape[0] == d["connectomes"].shape[0] == 41
    d = sb.utils.make_laterality_example(n_subjects=41)
    assert d["labels"].shape[0] == d["left"].shape[0] == 41


def test_fetch_fsaverage_uses_nilearn_keys(monkeypatch, tmp_path):
    nib = pytest.importorskip("nibabel")
    v, f = _sphere_mesh(50)
    img = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(v.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(f.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"),
        ]
    )
    path = tmp_path / "pial_left.gii"
    nib.save(img, path)
    fake_ds = types.ModuleType("nilearn.datasets")
    fake_ds.fetch_surf_fsaverage = lambda mesh="fsaverage": {"pial_left": str(path)}
    monkeypatch.setitem(sys.modules, "nilearn", types.ModuleType("nilearn"))
    monkeypatch.setitem(sys.modules, "nilearn.datasets", fake_ds)
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    from spectralbrain.utils.datasets import fetch_fsaverage

    gv, gf = fetch_fsaverage("pial", "lh")
    assert gv.shape == v.shape and gf.shape == f.shape


def test_voronoi_helper_obtuse_triangle():
    v = np.array([[0.0, 0, 0], [4, 0, 0], [2, 0.5, 0]])  # obtuse at vertex 2
    a = _voronoi_areas(v, np.array([[0, 1, 2]]))
    area = 0.5 * 4 * 0.5
    np.testing.assert_allclose(a, [area / 4, area / 4, area / 2])


def test_no_warnings_on_clean_mesh():
    v, f = _sphere_mesh(300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BrainMesh(v, f).decompose(k=3)
