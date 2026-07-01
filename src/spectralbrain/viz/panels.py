"""Parcellation-vs-clustering comparison panels in 3D.

The headline figure: a grid where **each column is an anatomical view** and **each
row is a distinct labeling** — a reference parcellation (e.g. Schaefer-200,
Brainnetome, an aseg structure) on the top row, and one or more data-driven
clusterings (ddCRP, Leiden, consensus, ...) on the rows below — every cell a 3D
surface render coloured by that labeling. Works uniformly for hippocampi, whole
brains, and white-matter bundle surfaces, because it operates on a generic
``(vertices, faces)`` mesh plus a dict of per-vertex labelings.

Rendering reuses SpectralBrain's existing offscreen vedo pipeline (with a PyVista
fallback); cells are composited into a single publication figure with row labels,
view headers, and panel letters.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger("spectralbrain.viz.panels")

PathLike = "str | os.PathLike[str]"


# ──────────────────────────────────────────────────────────────────────
# Single-cell renderers
# ──────────────────────────────────────────────────────────────────────
def _label_rgba(labels: np.ndarray, *, noise_color: str, categorical: bool):
    """Per-vertex RGBA (uint8) for a categorical labeling or continuous scalar."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from spectralbrain.viz.clusters import _cluster_cmap

    lab = np.asarray(labels)
    if categorical:
        unique = sorted(set(lab[lab >= 0].tolist()))
        cmap = _cluster_cmap(max(len(unique), 1))
        idx = {l: j for j, l in enumerate(unique)}
        rgba = np.array([mcolors.to_rgba(noise_color) if l < 0 else cmap(idx[l])
                         for l in lab], dtype=np.float64)
    else:
        v = np.asarray(lab, float)
        span = np.nanmax(v) - np.nanmin(v)
        norm = (v - np.nanmin(v)) / (span if span else 1.0)
        rgba = plt.get_cmap("viridis")(norm)
    return (rgba * 255).astype(np.uint8)


def _render_cell_vedo(vertices, faces, labels, view, *, noise_color, bg, size,
                      scale, categorical) -> str:
    """Render one (labeling, view) cell with vedo offscreen; return PNG path."""
    from spectralbrain.viz.clusters import (
        CAMERA_PRESETS, _build_vedo_mesh, _get_vedo,
    )
    vedo = _get_vedo()
    rgba_u8 = _label_rgba(labels, noise_color=noise_color, categorical=categorical)
    mesh = _build_vedo_mesh(vertices, faces, vedo)
    mesh.pointdata["RGBA"] = rgba_u8
    mesh.pointdata.select("RGBA")
    mesh.lighting("default")
    preset = CAMERA_PRESETS.get(view, {})
    plotter = vedo.Plotter(offscreen=True, size=size, bg=bg)
    plotter.show(mesh, viewup="z", zoom=1.1,
                 **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")})
    fd, png = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plotter.screenshot(png, scale=scale)
    plotter.close()
    return png


def _render_cell_pyvista(vertices, faces, labels, view, *, noise_color, bg, size,
                         scale, categorical) -> str:
    """PyVista fallback single-cell renderer; return PNG path."""
    from spectralbrain.viz.tracts3d import _require_pyvista, _set_pv_camera
    pv = _require_pyvista()
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    rgba = _label_rgba(labels, noise_color=noise_color, categorical=categorical)
    mesh = pv.PolyData(V, np.column_stack([np.full(len(F), 3), F]).ravel())
    mesh.point_data["RGBA"] = rgba
    plotter = pv.Plotter(off_screen=True, window_size=[size[0] * scale, size[1] * scale])
    plotter.set_background(bg)
    plotter.add_mesh(mesh, scalars="RGBA", rgb=True, smooth_shading=True)
    # map cluster view names onto tracts3d camera vocabulary where possible
    view_map = {"left_lateral": "left", "right_lateral": "right",
                "anterior": "anterior", "posterior": "posterior",
                "superior": "superior", "inferior": "inferior"}
    _set_pv_camera(plotter, V, view_map.get(view, "oblique"))
    fd, png = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plotter.screenshot(png)
    plotter.close()
    return png


# ──────────────────────────────────────────────────────────────────────
# The grid composer
# ──────────────────────────────────────────────────────────────────────
def plot_parcellation_cluster_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    labelings: Mapping[str, np.ndarray],
    *,
    views: Optional[Sequence[str]] = None,
    engine: str = "vedo",
    continuous: Optional[Sequence[str]] = None,
    noise_color: str = "lightgray",
    bg: str = "white",
    cell_size: tuple[int, int] = (600, 600),
    scale: int = 2,
    panel_letters: bool = True,
    title: Optional[str] = None,
    save: Optional[PathLike] = None,
    dpi: int = 300,
):
    """3D grid comparing parcellations and clusterings side by side.

    Rows are the entries of ``labelings`` (insertion order: put the reference
    parcellation first, then each clustering); columns are anatomical ``views``.
    Every cell is a 3D surface render coloured by that row's labeling, from that
    column's camera. Suitable for hippocampi, brains, and bundle surfaces.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    labelings : ordered mapping ``{name -> (V,) labels}``
        e.g. ``{"Schaefer-200": atlas, "ddCRP": r1.labels, "Leiden": r2.labels}``.
        Categorical by default; -1 is rendered as ``noise_color``.
    views : sequence of str, optional
        Camera presets (columns). Defaults to ``["left_lateral", "anterior",
        "superior"]``. Valid names: see
        :data:`spectralbrain.viz.clusters.CAMERA_PRESETS`.
    engine : {"vedo", "pyvista"}
        3D rendering backend.
    continuous : sequence of str, optional
        Names of labelings to render as continuous scalar maps (viridis) rather
        than categorical colours (e.g. a thickness/HKS overlay row).
    cell_size : (w, h)
        Per-cell render size in pixels (before ``scale``).
    save : path-like, optional
        Output figure path (PNG/PDF). A sibling ``.png`` is also written for
        non-PNG outputs.

    Returns
    -------
    (matplotlib.figure.Figure, dict)
        The composited figure and metadata (rendered cell paths, grid shape).
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    if views is None:
        from spectralbrain.viz.clusters import VIEWS_3POSE
        views = list(VIEWS_3POSE)
    views = list(views)
    row_names = list(labelings.keys())
    continuous = set(continuous or [])
    n_rows, n_cols = len(row_names), len(views)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Need at least one labeling and one view.")

    render = _render_cell_vedo if engine == "vedo" else _render_cell_pyvista

    cell_paths: dict[tuple[str, str], str] = {}
    for rname in row_names:
        labels = np.asarray(labelings[rname])
        if labels.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"labeling '{rname}' has {labels.shape[0]} entries but the mesh "
                f"has {vertices.shape[0]} vertices.")
        cat = rname not in continuous
        for view in views:
            png = render(vertices, faces, labels, view, noise_color=noise_color,
                         bg=bg, size=cell_size, scale=scale, categorical=cat)
            cell_paths[(rname, view)] = png

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    k = 0
    for i, rname in enumerate(row_names):
        for j, view in enumerate(views):
            ax = axes[i][j]
            ax.imshow(mpimg.imread(cell_paths[(rname, view)]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(view.replace("_", " ").title(), fontsize=11)
            if j == 0:
                n_k = int(np.unique(np.asarray(labelings[rname])[
                    np.asarray(labelings[rname]) >= 0]).size)
                ax.set_ylabel(f"{rname}\n(k={n_k})", fontsize=11, rotation=90,
                              labelpad=10)
            if panel_letters:
                ax.text(0.03, 0.97, letters[k % len(letters)],
                        transform=ax.transAxes, fontsize=12, fontweight="bold",
                        va="top", ha="left")
            k += 1

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    meta = {"cell_paths": cell_paths, "shape": (n_rows, n_cols),
            "rows": row_names, "views": views, "engine": engine}
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save), dpi=dpi, bbox_inches="tight")
        if save.suffix.lower() != ".png":
            fig.savefig(str(save.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
        logger.info("Saved parcellation/cluster grid → %s", save)
    return fig, meta


def plot_parcellation_vs_clusters(
    vertices: np.ndarray,
    faces: np.ndarray,
    parcellation: np.ndarray,
    clusterings: Mapping[str, np.ndarray],
    *,
    parcellation_name: str = "Parcellation",
    **kwargs,
):
    """Convenience wrapper: reference parcellation on top, clusterings below.

    ``clusterings`` may map names to label arrays or to
    :class:`~spectralbrain.statistics.clustering.ClusterResult` objects (their
    ``.labels`` are used).
    """
    labelings: dict[str, np.ndarray] = {parcellation_name: np.asarray(parcellation)}
    for name, val in clusterings.items():
        labelings[name] = np.asarray(getattr(val, "labels", val))
    return plot_parcellation_cluster_grid(vertices, faces, labelings, **kwargs)


__all__ = [
    "plot_parcellation_cluster_grid",
    "plot_parcellation_vs_clusters",
]
