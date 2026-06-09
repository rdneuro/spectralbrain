"""Smoke tests for the template-free six-view renderer (viz.hipp3d)."""

import numpy as np
import pytest

pytest.importorskip("vedo")
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import spectralbrain as sb
from spectralbrain.viz.hipp3d import SIXVIEWS, plot_hippocampus_sixview, plot_surface_sixview


def _blob_mesh():
    """A small closed surface (icosphere-ish) via marching cubes on a ball."""
    from spectralbrain.core.base import marching_cubes

    zz, yy, xx = np.mgrid[0:24, 0:24, 0:24]
    vol = (((xx - 12) ** 2 + (yy - 12) ** 2 + (zz - 12) ** 2) <= 49).astype(np.float32)
    v, f = marching_cubes(vol, np.eye(4), level=0.5)
    return v, f


def test_sixviews_constant():
    assert SIXVIEWS == (
        "anterior", "posterior", "inferior", "superior", "left_lateral", "right_lateral",
    )  # fmt: skip


def test_sixview_geometry_only_returns_figure():
    v, f = _blob_mesh()
    fig = plot_hippocampus_sixview((v, f), None, window=(180, 170))
    # 6 view axes present (a 2×3 grid).
    drawn = [ax for ax in fig.axes if ax.images]
    assert len(drawn) == 6
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_sixview_scalar_returns_figure_with_colorbar():
    v, f = _blob_mesh()
    scalars = v[:, 1]  # smooth A-P gradient
    fig = plot_hippocampus_sixview(
        (v, f), scalars, cmap="plasma", scalar_bar_title="HKS", window=(180, 170)
    )
    drawn = [ax for ax in fig.axes if ax.images]
    assert len(drawn) == 6
    # A colorbar axis exists in addition to the 6 render axes.
    assert len(fig.axes) >= 7
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_sixview_scalar_length_mismatch_raises():
    v, f = _blob_mesh()
    with pytest.raises(ValueError, match="scalars length"):
        plot_hippocampus_sixview((v, f), np.zeros(len(v) + 5), window=(120, 120))


def test_sixview_subset_of_views():
    v, f = _blob_mesh()
    fig = plot_hippocampus_sixview(
        (v, f), None, views=("superior", "left_lateral"), window=(160, 150)
    )
    drawn = [ax for ax in fig.axes if ax.images]
    assert len(drawn) == 2
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_sixview_accepts_brainmesh():
    v, f = _blob_mesh()
    mesh = sb.BrainMesh(v, f)
    fig = plot_surface_sixview(mesh, None, window=(150, 140))
    assert len([ax for ax in fig.axes if ax.images]) == 6
    import matplotlib.pyplot as plt

    plt.close(fig)
