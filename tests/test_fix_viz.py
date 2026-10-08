"""Regression tests for the viz bug-fix pass (silent-failure audit).

Each test pins one previously-silent failure in ``spectralbrain.viz``. Tests
that need an optional renderer (vedo / fury / yabplot) are skipped when it is
not installed; the rest exercise the pure-numpy / matplotlib logic.
"""

from __future__ import annotations

import types
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _write_png(path):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# #1 / #2  camera conventions (RAS) -- vedo, PyVista and FURY
# ----------------------------------------------------------------------


def test_camera_ras_convention():
    from spectralbrain.viz._camera import camera_for_view

    pts = np.random.default_rng(0).random((200, 3)) * [10, 20, 5]
    center = 0.5 * (pts.min(0) + pts.max(0))
    expected = {
        "left_lateral": [-1, 0, 0],
        "right_lateral": [1, 0, 0],
        "anterior": [0, 1, 0],
        "posterior": [0, -1, 0],
        "superior": [0, 0, 1],
        "inferior": [0, 0, -1],
        "left_medial": [1, 0, 0],
        "right_medial": [-1, 0, 0],
    }
    for view, direction in expected.items():
        cam = camera_for_view(view, pts)
        d = np.asarray(cam["position"]) - center
        np.testing.assert_allclose(d / np.linalg.norm(d), direction, atol=1e-9)
        np.testing.assert_allclose(cam["focal_point"], center)


def test_camera_presets_consistent_and_unknown_view_raises():
    from spectralbrain.viz.clusters import CAMERA_PRESETS, _view_camera
    from spectralbrain.viz.render3d import CAMERA_PRESETS as MESH_PRESETS

    assert CAMERA_PRESETS == MESH_PRESETS
    pts = np.random.default_rng(1).random((50, 3))
    with pytest.raises(ValueError, match="Unknown view"):
        _view_camera("left_lateal", pts)  # typo must not silently render


def test_pyvista_camera_uses_ras_and_cluster_names():
    from spectralbrain.viz.render3d import _set_pv_camera

    class FakePlotter:
        camera_position = None

    V = np.random.default_rng(2).normal(size=(100, 3))
    for view, axis in [
        ("left", [-1, 0, 0]),
        ("left_lateral", [-1, 0, 0]),
        ("anterior", [0, 1, 0]),
        ("superior", [0, 0, 1]),
    ]:
        p = FakePlotter()
        _set_pv_camera(p, V, view)
        pos, foc, _up = p.camera_position
        d = np.asarray(pos) - np.asarray(foc)
        np.testing.assert_allclose(d / np.linalg.norm(d), axis, atol=1e-9)
    with pytest.raises(ValueError):
        _set_pv_camera(FakePlotter(), V, "sideways")


def test_fury_camera_is_explicit_ras():
    from spectralbrain.viz.render3d import _set_fury_camera

    class FakeScene:
        def set_camera(self, position, focal_point, view_up):
            self.pos, self.foc, self.up = position, focal_point, view_up

        def reset_clipping_range(self):
            pass

    pts = np.random.default_rng(3).normal(size=(300, 3))
    for view, axis in [
        ("left", [-1, 0, 0]),
        ("right", [1, 0, 0]),
        ("anterior", [0, 1, 0]),
        ("superior", [0, 0, 1]),
    ]:
        sc = FakeScene()
        _set_fury_camera(sc, view, pts)
        d = np.asarray(sc.pos) - np.asarray(sc.foc)
        np.testing.assert_allclose(d / np.linalg.norm(d), axis, atol=1e-9)


# ----------------------------------------------------------------------
# #3  tract panel colorbar == render clim / cmap / norm
# ----------------------------------------------------------------------


def test_streamlines_multiview_colorbar_matches_renders(tmp_path, monkeypatch):
    from spectralbrain.viz import render3d as nt

    calls = []

    def fake_render(streamlines, **kw):
        calls.append(kw)
        return _write_png(kw["out_path"])

    monkeypatch.setattr(nt, "render_streamlines", fake_render)
    rng = np.random.default_rng(0)
    sl = [np.cumsum(rng.normal(size=(20, 3)), axis=0) for _ in range(5)]
    sc = [rng.normal(loc=3.0, size=20) for _ in range(5)]
    fig, _ = nt.streamlines_multiview(
        sl,
        views=("left", "anterior"),
        color_by="scalar",
        scalars=sc,
        colorbar={"kind": "sequential", "label": "FA"},
    )
    clims = {tuple(c["clim"]) for c in calls}
    assert len(clims) == 1  # every view uses one resolved clim
    lo, hi = clims.pop()
    expected = nt.robust_clim(np.concatenate(sc))
    np.testing.assert_allclose((lo, hi), expected)
    cbar = next(ax for ax in fig.axes if hasattr(ax, "_colorbar"))._colorbar
    np.testing.assert_allclose((cbar.norm.vmin, cbar.norm.vmax), (lo, hi))
    assert cbar.cmap.name == calls[0]["cmap"].name


def test_compose_tract_panel_uses_linear_norm(tmp_path):
    from spectralbrain.viz import render3d as nt

    pngs = [_write_png(tmp_path / f"{i}.png") for i in range(2)]
    fig = nt.compose_tract_panel(pngs, colorbar={"kind": "diverging", "clim": (-1.0, 3.0)})
    cbar = next(ax for ax in fig.axes if hasattr(ax, "_colorbar"))._colorbar
    assert type(cbar.norm) is mcolors.Normalize  # same mapping as the renders
    assert (cbar.norm.vmin, cbar.norm.vmax) == (-1.0, 3.0)


# ----------------------------------------------------------------------
# #4  medial-wall mask
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask",
    [
        np.array([0.0, 0, 0, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([False, False, False, True, True, True]),
        np.array([3, 4, 5]),
    ],
)
def test_medial_mask_binary_and_indices(mask):
    from spectralbrain.viz.spectral import _apply_medial_mask

    out = _apply_medial_mask(np.arange(6.0), mask)
    assert np.isnan(out[3:]).all()
    np.testing.assert_array_equal(out[:3], [0.0, 1.0, 2.0])


# ----------------------------------------------------------------------
# #6  savefig
# ----------------------------------------------------------------------


def test_savefig_keeps_dotted_stem_and_requested_suffix(tmp_path):
    from spectralbrain.viz.graphics import savefig

    fig, _ = plt.subplots()
    out = savefig(fig, tmp_path / "hks_t0.5.png")
    assert [p.name for p in out] == ["hks_t0.5.png"]
    out = savefig(fig, tmp_path / "fig.pdf")
    assert sorted(p.name for p in out) == ["fig.pdf", "fig.png"]
    assert (tmp_path / "fig.pdf").exists()
    out = savefig(fig, tmp_path / "wks_e2.5", formats=["svg"])
    assert sorted(p.name for p in out) == ["wks_e2.5.png", "wks_e2.5.svg"]


# ----------------------------------------------------------------------
# #7  kymograph
# ----------------------------------------------------------------------


def test_kymograph_runs_and_line_position_is_normalised():
    from spectralbrain.viz.clusters import plot_kymograph

    rng = np.random.default_rng(0)
    uv = rng.random((600, 2)) * [10.0, 4.0] + [5.0, 2.0]  # not in [0, 1]
    T = 5
    # value = PD coordinate (column 1) for every scale
    H = np.repeat(uv[:, 1:2], T, axis=1) + 1.0
    _, ax = plot_kymograph(uv, None, H, line_axis="AP", line_position=0.5, log_norm=False)
    kymo = ax.collections[0].get_array().reshape(200, T)
    mid_pd = 2.0 + 0.5 * 4.0
    finite = np.isfinite(kymo)
    assert finite.mean() > 0.8
    np.testing.assert_allclose(kymo[finite], mid_pd + 1.0, atol=0.05)
    with pytest.raises(ValueError):
        plot_kymograph(uv, None, H, line_position=2.0)


def test_kymograph_outside_hull_is_nan():
    from spectralbrain.viz.clusters import plot_kymograph

    # triangle-shaped domain: the AP line at PD=0.9 leaves the hull
    rng = np.random.default_rng(1)
    pts = rng.random((3000, 2))
    uv = pts[pts[:, 1] <= pts[:, 0]]
    H = np.ones((len(uv), 3))
    _, ax = plot_kymograph(uv, None, H, line_position=0.9, log_norm=False)
    assert np.isnan(np.asarray(ax.collections[0].get_array(), float)).any()


# ----------------------------------------------------------------------
# #8  thresholded rows
# ----------------------------------------------------------------------


def test_brain_normative_vertexwise_thresholded(monkeypatch, tmp_path):
    from spectralbrain.viz import render3d as bp

    specs = []

    def fake_render(spec, out_png, **kw):
        specs.append(spec)
        return _write_png(out_png)

    monkeypatch.setattr(bp, "_render_row", fake_render)
    lh = np.array([0.5, 2.5, -3.0, 1.0])
    rh = np.array([-2.1, 0.0, 4.0, np.nan])
    bp.plot_normative_map((lh, rh), plot_kind="vertexwise", threshold=2.0)
    thr_lh, thr_rh = specs[1].data
    np.testing.assert_array_equal(np.isnan(thr_lh), [True, False, False, True])
    np.testing.assert_array_equal(np.isnan(thr_rh), [False, True, False, True])
    assert lh[0] == 0.5  # caller data untouched


def test_hipp_normative_no_global_mutation_and_path_input(monkeypatch, tmp_path):
    nib = pytest.importorskip("nibabel")
    from spectralbrain.viz import hipp

    before = {k: dict(v) for k, v in hipp.HIPP_DESCRIPTOR_STYLES.items()}
    captured = {}

    def fake_gallery(descriptors, **kw):
        captured["descriptors"] = descriptors
        captured["styles"] = kw.get("styles")
        return None, None

    monkeypatch.setattr(hipp, "plot_hippocampus_gallery", fake_gallery)
    z = np.array([0.5, 2.5, -3.0, 1.9], dtype=np.float32)
    gii = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(z)])
    path = tmp_path / "z.shape.gii"
    nib.save(gii, str(path))
    hipp.plot_hippocampus_normative(str(path), threshold=2.0)
    thr = captured["descriptors"]["|Z| > 2.0"]
    np.testing.assert_array_equal(np.isnan(thr), [True, False, False, True])
    assert hipp.HIPP_DESCRIPTOR_STYLES == before
    assert captured["styles"]["Z-score"]["vmin"] == -3.0


# ----------------------------------------------------------------------
# #9  Hovmoller
# ----------------------------------------------------------------------


def test_hovmoller_empty_bins_nan_and_shape_check():
    from spectralbrain.viz.clusters import plot_hovmoller

    rng = np.random.default_rng(0)
    ap = np.r_[rng.random(100) * 0.3, 0.7 + rng.random(100) * 0.3]
    uv = np.column_stack([ap, rng.random(200)])
    H = 5.0 + rng.random((200, 4))
    _, ax = plot_hovmoller(uv, H, np.array([1.0, 2, 3, 4]), log_norm=False, n_bins=10)
    mesh = ax.collections[0]
    arr = np.ma.getdata(mesh.get_array()).reshape(10, 4)
    assert np.isnan(arr[4:6]).all()  # empty middle bins are NaN, not 0
    assert mesh.norm.vmin >= 5.0
    with pytest.raises(ValueError, match="vertices"):
        plot_hovmoller(rng.random((500, 2)), H)


# ----------------------------------------------------------------------
# #10  cluster colours independent of noise
# ----------------------------------------------------------------------


def test_cluster_sizes_colours_ignore_noise():
    from spectralbrain.viz.clusters import CLUSTER_COLORS, plot_cluster_sizes

    _, ax = plot_cluster_sizes(np.array([-1, -1, 0, 0, 1, 2]))
    cols = [mcolors.to_hex(p.get_facecolor()) for p in ax.patches]
    assert cols[1].lower() == CLUSTER_COLORS[0].lower()
    assert cols[2].lower() == CLUSTER_COLORS[1].lower()


# ----------------------------------------------------------------------
# #12  shared colour limits for comparisons
# ----------------------------------------------------------------------


def test_brainplots_shared_vminmax():
    from spectralbrain.viz.render3d import BrainPlotSpec, _shared_vminmax

    a = BrainPlotSpec(data={"r1": 1.0, "r2": 2.0}, cmap="viridis")
    b = BrainPlotSpec(data={"r1": 3.0, "r2": 5.0}, cmap="viridis", vminmax=[None, 4.0])
    sa, sb = _shared_vminmax([a, b])
    assert sa.vminmax == [1.0, 5.0]
    assert sb.vminmax == [1.0, 4.0]  # explicit limit kept
    assert a.vminmax == [None, None]  # caller spec not mutated


def test_hipp_shared_range():
    from spectralbrain.viz.hipp import _shared_range

    assert _shared_range([np.array([1.0, 2.0]), np.array([0.5, np.nan, 4.0])], None, None) == (
        0.5,
        4.0,
    )
    assert _shared_range([np.array([1.0])], 0.0, None) == (0.0, 1.0)


# ----------------------------------------------------------------------
# #13  log axis does not drop the first scale
# ----------------------------------------------------------------------


def test_cluster_profiles_default_t_starts_at_one():
    from spectralbrain.viz.clusters import plot_cluster_profiles

    H = np.random.default_rng(0).random((40, 6))
    lab = np.r_[np.zeros(20, int), np.ones(20, int)]
    _, ax = plot_cluster_profiles(H, lab)
    assert ax.lines[0].get_xdata()[0] == 1.0
    assert ax.get_xlim()[0] <= 1.0


# ----------------------------------------------------------------------
# #14 / #15  colormap selection: z_score, signed, categorical
# ----------------------------------------------------------------------


def test_resolve_cmap_z_score():
    from spectralbrain.viz.render3d import _resolve_cmap as mesh_cmap

    assert mesh_cmap("z_score", None) == "RdBu_r"
    assert mesh_cmap("hks_t10", None) == "inferno"


def test_signed_descriptor_centred_and_labels_discrete():
    from spectralbrain.viz.render3d import BrainPlotSpec, _resolve_spec

    spec = BrainPlotSpec.from_descriptor("gaussian_k", data={"a": -0.2, "b": 0.6})
    assert _resolve_spec(spec).vminmax == [-0.6, 0.6]
    lab = BrainPlotSpec.from_descriptor("clusters", data={"a": 0, "b": 7, "c": 3, "d": -1})
    r = _resolve_spec(lab)
    assert r.vminmax == [-0.5, 2.5]
    assert r.data == {"a": 0.0, "b": 2.0, "c": 1.0, "d": r.data["d"]}
    assert np.isnan(r.data["d"])  # noise -> nan_color
    assert r.cmap.N == 3


# ----------------------------------------------------------------------
# #16 / #17  Bayesian summaries
# ----------------------------------------------------------------------


def _fake_trace(**arrays):
    post = {k: types.SimpleNamespace(values=np.asarray(v)) for k, v in arrays.items()}
    return types.SimpleNamespace(posterior=post)


def test_horseshoe_kappa_includes_tau():
    from spectralbrain.viz.bayes import plot_horseshoe_coefficients

    rng = np.random.default_rng(0)
    beta = rng.normal(size=(1, 50, 2))
    lam = np.ones((1, 50, 2))
    tau = np.full((1, 50), 0.1)
    _, (_, ax_shrink) = plot_horseshoe_coefficients(
        _fake_trace(beta=beta, **{"lambda": lam}, tau=tau)
    )
    vals = sorted(float(t.get_text()) for t in ax_shrink.texts)
    np.testing.assert_allclose(vals, [round(1 / (1 + 0.01), 2)] * 2)  # not 0.5


def test_centroid_norm_interval_contains_mean():
    from spectralbrain.viz.clusters import _centroid_norm_interval

    ci = {
        "mean": np.array([0.1, -0.1]),
        "hdi_3": np.array([-1.0, -1.0]),
        "hdi_97": np.array([1.0, 1.0]),
    }
    lo, m, hi = _centroid_norm_interval(ci)
    assert lo <= m <= hi
    assert lo == 0.0
    samples = np.random.default_rng(0).normal(size=(1000, 3)) + 5
    lo, m, hi = _centroid_norm_interval({**ci, "samples": samples})
    norms = np.linalg.norm(samples, axis=1)
    np.testing.assert_allclose((lo, hi), np.percentile(norms, [3, 97]))


def test_hdi_spans_requested_mass():
    from spectralbrain.viz.bayes import _hdi

    lo, hi = _hdi(np.arange(100.0), 0.9)
    assert hi - lo == 89.0  # exactly 90 samples
    lo, hi = _hdi(np.r_[np.arange(10.0), np.nan], 0.5)
    assert np.isfinite([lo, hi]).all()


def test_connectome_posterior_ignores_nan_in_vmax():
    from spectralbrain.viz.bayes import plot_connectome_posterior

    M = np.array([[0.0, 2.0], [np.nan, -1.0]])
    _, ax = plot_connectome_posterior(M)
    im = ax.images[0]
    assert im.norm.vmax == 2.0 and im.norm.vmin == -2.0


def test_forest_length_mismatch_raises():
    from spectralbrain.viz.bayes import plot_forest

    with pytest.raises(ValueError):
        plot_forest(["a", "b"], [np.zeros(10)])


# ----------------------------------------------------------------------
# #18 / #19  panels NaN, unknown views, distplot
# ----------------------------------------------------------------------


def test_panels_nan_gets_noise_colour():
    from spectralbrain.viz.clusters import _label_rgba

    rgba = _label_rgba(
        np.array([0.0, 1.0, np.nan, 2.0]), noise_color="lightgray", categorical=False
    )
    grey = (np.array(mcolors.to_rgba("lightgray")) * 255).astype(np.uint8)
    np.testing.assert_array_equal(rgba[2], grey)
    rgba = _label_rgba(
        np.array([0.0, np.nan, -1.0, 1.0]), noise_color="lightgray", categorical=True
    )
    np.testing.assert_array_equal(rgba[1], grey)
    np.testing.assert_array_equal(rgba[2], grey)


def test_panels_unknown_view_raises():
    from spectralbrain.viz.clusters import plot_parcellation_cluster_grid

    V = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValueError, match="Unknown view"):
        plot_parcellation_cluster_grid(
            V, np.array([[0, 1, 2]]), {"a": np.zeros(10)}, views=["left_lateal"]
        )


def test_distplot_keeps_all_groups():
    from spectralbrain.viz.graphics import distplot

    rng = np.random.default_rng(0)
    _, ax = distplot([rng.normal(size=50) for _ in range(12)])
    assert len(ax.lines) == 12


# ----------------------------------------------------------------------
# #20  parameters no longer silently ignored
# ----------------------------------------------------------------------


def test_vertexwise_extra_kwargs_forwarded(monkeypatch, tmp_path):
    from spectralbrain.viz import render3d as bp

    seen = {}

    def fake_fn(lh, rh, **kw):
        seen.update(kw)

    monkeypatch.setattr(bp, "_get_plot_fn", lambda kind: fake_fn)
    spec = bp.BrainPlotSpec(
        data=(np.zeros(3), np.ones(3)),
        plot_kind="vertexwise",
        cmap="viridis",
        extra_kwargs={"zoom": 1.4},
    )
    bp._render_row(spec, tmp_path / "x.png", views=["left_lateral"])
    assert seen["zoom"] == 1.4


def test_top10_keeps_documented_keys_and_warns_unknown(monkeypatch):
    from spectralbrain.viz import render3d as bp

    captured = {}
    monkeypatch.setattr(
        bp, "plot_morphometric_gallery", lambda specs, **kw: captured.setdefault("specs", specs)
    )
    with pytest.warns(UserWarning, match="mystery"):
        bp.plot_top10_morphometrics(
            {"hks": {"a": 1.0}, "shapedna": {"a": 2.0}, "mystery": {"a": 3.0}}
        )
    labels = [s.label for s in captured["specs"]]
    assert labels[:2] == ["HKS", "ShapeDNA"] and "mystery" in labels


def test_hipp_warns_on_unsupported_nan_color():
    from spectralbrain.viz.hipp import _warn_ignored

    with pytest.warns(UserWarning, match="nan_color"):
        _warn_ignored(nan_color="red")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_ignored()  # defaults: silent


# ----------------------------------------------------------------------
# #21  misc edge paths
# ----------------------------------------------------------------------


def test_silhouette_precomputed_with_noise():
    pytest.importorskip("sklearn")
    from scipy.spatial.distance import cdist

    from spectralbrain.viz.clusters import plot_silhouette_diagram

    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (20, 2)), rng.normal(6, 1, (20, 2)), rng.normal(3, 5, (5, 2))]
    lab = np.r_[np.zeros(20, int), np.ones(20, int), -np.ones(5, int)]
    D = cdist(X, X)
    _, ax = plot_silhouette_diagram(D, lab, metric="precomputed")
    assert len(ax.patches) == 40


# ----------------------------------------------------------------------
# vedo-dependent regressions (skipped when vedo is absent)
# ----------------------------------------------------------------------


def test_hipp3d_nan_not_painted_as_clim_min():
    pytest.importorskip("vedo")
    from spectralbrain.viz.hipp import plot_hippocampus_sixview

    zz, yy, xx = np.mgrid[0:16, 0:16, 0:16]
    from spectralbrain.core.base import marching_cubes

    vol = (((xx - 8) ** 2 + (yy - 8) ** 2 + (zz - 8) ** 2) <= 25).astype(np.float32)
    v, f = marching_cubes(vol, np.eye(4), level=0.5)
    s = v[:, 0] - v[:, 0].mean()
    s[:10] = np.nan
    fig = plot_hippocampus_sixview((v, f), s, signed=True, window=(120, 110))
    assert fig is not None
    with pytest.raises(ValueError):
        plot_hippocampus_sixview((v, f), s, views=("left_medial",), window=(120, 110))


def test_mesh_comparison_respects_zero_vmin():
    pytest.importorskip("vedo")
    from spectralbrain.viz.render3d import plot_mesh_comparison

    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    _, meta = plot_mesh_comparison(
        [{"vertices": V, "faces": F, "scalars": np.arange(4.0) + 1, "vmin": 0.0}]
    )
    assert meta["scalar_ranges"][0][0] == 0.0
