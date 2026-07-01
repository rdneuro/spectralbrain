"""Tests for the new visualization modules: tracts3d (3D tractography) and
panels (parcellation-vs-clustering grid). The GL-dependent renderers
(FURY/PyVista/vedo) are exercised elsewhere; here we test the non-GL compute and
composition logic plus the clean ImportError guards.
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from spectralbrain.viz import panels as pn
from spectralbrain.viz import tracts3d as nt

skimage = pytest.importorskip("skimage")
trimesh = pytest.importorskip("trimesh")


def _ellipsoid_mask():
    zz, yy, xx = np.mgrid[0:36, 0:44, 0:52]
    return (((xx - 26) / 20) ** 2 + ((yy - 22) / 16) ** 2
            + ((zz - 18) / 13) ** 2 <= 1.0).astype(float)


def test_mask_to_mesh_clean_manifold():
    V, F = nt.mask_to_mesh(_ellipsoid_mask(), affine=np.eye(4),
                           smooth_sigma=1.0, taubin_iter=10)
    assert V.ndim == 2 and V.shape[1] == 3
    assert F.ndim == 2 and F.shape[1] == 3
    # closed genus-0 surface: Euler characteristic 2
    e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]],
                                     F[:, [0, 2]]]), axis=1), axis=0).shape[0]
    assert V.shape[0] - e + F.shape[0] == 2


def test_spectral_overlay_hks_finite():
    V, F = nt.mask_to_mesh(_ellipsoid_mask(), affine=np.eye(4), taubin_iter=5)
    hks = nt.spectral_overlay(V, F, kind="hks", n_eigen=40, t_index=10)
    assert hks.shape == (V.shape[0],)
    assert np.isfinite(hks).all()


def test_robust_clim_symmetric():
    v = np.array([-5.0, -1, 0, 1, 8])
    lo, hi = nt.robust_clim(v, symmetric=True)
    assert lo == -hi and hi > 0


def test_compose_tract_panel(tmp_path):
    import matplotlib.pyplot as plt
    pngs = []
    for c in ("r", "g", "b"):
        fig, ax = plt.subplots()
        ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color=c)); ax.axis("off")
        p = tmp_path / f"{c}.png"
        fig.savefig(p); plt.close(fig); pngs.append(p)
    out = tmp_path / "panel.pdf"
    fig = nt.compose_tract_panel(pngs, titles=["L", "A", "S"],
                                 colorbar={"kind": "diverging", "clim": (-1, 1),
                                           "label": "d"}, out_path=out)
    assert out.exists() and out.with_suffix(".png").exists()


def test_parcellation_cluster_grid_assembly(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    def fake_cell(vertices, faces, labels, view, **kw):
        fig, ax = plt.subplots(figsize=(2, 2)); ax.axis("off")
        p = tempfile.mktemp(suffix=".png"); fig.savefig(p); plt.close(fig)
        return p

    monkeypatch.setattr(pn, "_render_cell_vedo", fake_cell)
    V = np.random.default_rng(0).normal(size=(50, 3))
    F = np.array([[0, 1, 2]])  # faces unused by the patched renderer
    lab = np.random.default_rng(0).integers(0, 3, size=50)
    labelings = {"Atlas": lab, "ddCRP": (lab + 1) % 3}
    out = tmp_path / "grid.png"
    fig, meta = pn.plot_parcellation_cluster_grid(
        V, F, labelings, views=["left_lateral", "anterior", "superior"],
        engine="vedo", save=out)
    assert meta["shape"] == (2, 3)
    assert out.exists()


def test_streamline_render_requires_fury():
    # FURY is an optional dep; the guard should raise a clear ImportError.
    sl = [np.cumsum(np.random.default_rng(0).normal(size=(20, 3)), axis=0)]
    try:
        import fury  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="FURY"):
            nt.render_streamlines(sl, out_path=tempfile.mktemp(suffix=".png"))
