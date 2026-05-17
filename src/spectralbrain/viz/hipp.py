"""Hippocampal surface visualisation — 3D multi-view + unfolded flatmaps.

Combines two rendering engines:
- **hippunfold_plot** (nilearn/matplotlib backend) — for 2D unfolded
  flatmap views and standard 3D views via ``plot_hipp_surf()``.
- **hippomaps** (BrainSpace/VTK backend) — for high-quality 3D
  folded renders via ``surfplot_sub_foldunfold()``.

HippUnfold version compatibility
---------------------------------
All functions accept both **v1** (``"0p5mm"``, ``"1mm"``, ``"2mm"``)
and **v2** (``"2k"``, ``"8k"``, ``"18k"``) density labels.  The
default changed from ``"0p5mm"`` to **``"8k"``** to match HippUnfold
v2's vertex-count-based naming convention.

- v1 labels still work but emit a ``DeprecationWarning``.
- ``"8k"`` ≈ v1 ``"0p5mm"`` (~8,000 combined hipp + dentate vertices).
- v2 merges ``label-hipp`` + ``label-dentate`` → ``label-hippdentate``;
  all label variants are accepted.

See :data:`DENSITIES` for the full mapping and :data:`HIPP_LABELS`
for recognised structure labels.

Each panel row shows 3D views of the hippocampus in the main columns
and an unfolded flatmap in the first or last column, enabling
simultaneous inspection of spatial localisation (3D) and subfield
identity (unfolded).

Figure types
------------
1. Single metric on one hippocampus (3D views + flatmap)
2. Bilateral panel (L + R side by side)
3. Group comparison (control vs patient vs difference)
4. Spectral descriptor gallery (HKS, WKS, BKS, … stacked)
5. Multi-subject normative deviation panel
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from spectralbrain.runtime import PathLike, ScalarMap, get_logger

logger = get_logger(__name__)

DPI: int = 600

# 3D views for hippocampus (hippunfold_plot view tokens).
HIPP_VIEWS_3D: List[str] = [
    "lateral", "medial", "dorsal", "ventral", "anterior",
]
"""Standard 3D views for hippocampal surface."""

HIPP_VIEWS_FULL: List[str] = HIPP_VIEWS_3D + ["flatmap"]
"""3D views + unfolded flatmap."""

# Density mapping for HippUnfold versions.
# v2 (vertex-count-based) is the new canonical naming.
# v1 (millimetre-based) names are kept as backward-compatible aliases.
DENSITIES: Dict[str, str] = {
    # HippUnfold v2 canonical names (preferred)
    "2k": "2k",         # ~2,000 combined vertices
    "8k": "8k",         # ~8,000 combined vertices (default)
    "18k": "18k",       # ~18,000 combined vertices
    # HippUnfold v1 backward-compatible aliases
    "0p5mm": "0p5mm",   # v1 default — 7,262 hipp + 1,788 dentate
    "1mm": "1mm",       # v1 — 2,004 hipp + 449 dentate
    "2mm": "2mm",       # v1 — 419 hipp + 64 dentate
}
"""Recognised HippUnfold density labels (v1 and v2)."""

# v1 → v2 approximate equivalence (for docs/migration only).
_DENSITY_V1_TO_V2: Dict[str, str] = {
    "0p5mm": "8k",
    "1mm": "2k",
    "2mm": "2k",
}

# HippUnfold v2 merges hipp + dentate into a single surface.
# v1 outputs: label-hipp, label-dentate  (separate surfaces)
# v2 outputs: label-hippdentate           (combined surface)
# SpectralBrain accepts all three.
HIPP_LABELS: List[str] = ["hipp", "dentate", "hippdentate"]
"""Recognised HippUnfold structure labels."""


def _resolve_density(density: str) -> str:
    """Validate and return the density string.

    Accepts both HippUnfold v1 (``"8k"``, ``"1mm"``, ``"2mm"``) and
    v2 (``"2k"``, ``"8k"``, ``"18k"``) density labels.  Emits a
    deprecation warning for v1 labels and returns the string unchanged
    so that downstream libraries (hippunfold_plot, hippomaps) can
    handle it in their own way.

    Parameters
    ----------
    density : str
        Density label to validate.

    Returns
    -------
    str
        The validated density string (unchanged).

    Raises
    ------
    ValueError
        If *density* is not a recognised label.
    """
    import warnings

    if density not in DENSITIES:
        raise ValueError(
            f"Unknown HippUnfold density: {density!r}. "
            f"Use one of {list(DENSITIES.keys())}."
        )
    if density in _DENSITY_V1_TO_V2:
        v2_equiv = _DENSITY_V1_TO_V2[density]
        warnings.warn(
            f"HippUnfold v1 density '{density}' is deprecated. "
            f"Consider switching to v2 equivalent '{v2_equiv}'. "
            f"See HippUnfold v2.0.0 release notes.",
            DeprecationWarning,
            stacklevel=3,
        )
    return density

# Default descriptor visual specs (hippocampal-specific).
HIPP_DESCRIPTOR_STYLES: Dict[str, Dict[str, Any]] = {
    "thickness":   {"cmap": "inferno", "vmin": 1.0, "vmax": 4.0, "label": "Thickness"},
    "curvature":   {"cmap": "RdBu_r", "vmin": -0.3, "vmax": 0.3, "label": "Curvature"},
    "gyrification": {"cmap": "magma",  "vmin": 0, "vmax": 2.0, "label": "Gyrification"},
    "subfields":   {"cmap": "tab10",   "vmin": None, "vmax": None, "label": "Subfields"},
    "hks":         {"cmap": "inferno", "vmin": None, "vmax": None, "label": "HKS"},
    "wks":         {"cmap": "cividis", "vmin": None, "vmax": None, "label": "WKS"},
    "bks":         {"cmap": "magma",   "vmin": None, "vmax": None, "label": "BKS"},
    "z_score":     {"cmap": "RdBu_r", "vmin": -3, "vmax": 3, "label": "Z-score"},
    "effect_d":    {"cmap": "RdBu_r", "vmin": -1.5, "vmax": 1.5, "label": "Cohen's d"},
    "shape_idx":   {"cmap": "RdBu_r", "vmin": -1, "vmax": 1, "label": "Shape Index"},
    "casorati":    {"cmap": "magma",   "vmin": None, "vmax": None, "label": "Casorati"},
}


# ======================================================================
# §0  LAZY IMPORTS
# ======================================================================

def _require_hippunfold_plot():
    """Lazy-import hippunfold_plot for hippocampal rendering."""
    try:
        from hippunfold_plot.plotting import plot_hipp_surf
        return plot_hipp_surf
    except ImportError as exc:
        raise ImportError(
            "hippunfold_plot is required for hippocampal flatmaps.\n"
            "  pip install hippunfold_plot"
        ) from exc


def _require_hippomaps():
    """Lazy-import hippomaps for normative hippocampal context."""
    try:
        import hippomaps
        return hippomaps
    except ImportError:
        logger.debug("hippomaps not installed — using hippunfold_plot only.")
        return None


def _apply_style():
    """Apply scienceplots if available."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
        plt.rcParams["mathtext.fontset"] = "cm"
    except ImportError:
        pass
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["figure.dpi"] = DPI


def _save_figure(fig, path, formats=None):
    """Save the figure to disk if a path is provided."""
    from spectralbrain.viz.graphics import savefig
    return savefig(fig, path, formats=formats, dpi=DPI)


# ======================================================================
# §1  CORE RENDERING
# ======================================================================

def _render_hipp_3d(
    surf_map: Any,
    *,
    view: str = "dorsal",
    hemi: str = "left",
    density: str = "8k",
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nan_color: Any = (0.85, 0.85, 0.85),
    bg_on_data: bool = True,
    alpha: float = 0.1,
    darkness: float = 2,
    dpi: int = DPI,
) -> Figure:
    """Render one 3D hippocampal view via hippunfold_plot.

    Parameters
    ----------
    surf_map : str or ndarray
        GIfTI path or vertex array.
    view : str
        ``"dorsal"``, ``"ventral"``, ``"lateral"``, ``"medial"``,
        ``"anterior"``, ``"posterior"``.
    hemi : str
        ``"left"`` or ``"right"``.
    density : str
        HippUnfold density.

    Returns
    -------
    matplotlib.Figure
    """
    plot_hipp_surf = _require_hippunfold_plot()
    density = _resolve_density(density)

    kwargs = dict(
        surf_map=surf_map,
        density=density,
        hemi=hemi,
        view=view,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        bg_on_data=bg_on_data,
        alpha=alpha,
        darkness=darkness,
        dpi=dpi,
        figsize=(4, 3),
        colorbar=False,
    )

    fig = plot_hipp_surf(**kwargs)
    return fig


def _render_hipp_flatmap(
    surf_map: Any,
    *,
    hemi: str = "left",
    density: str = "8k",
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = DPI,
) -> Figure:
    """Render an unfolded (flatmap) view via hippunfold_plot.

    Uses ``space='unfold'`` to get the 2D representation.
    """
    plot_hipp_surf = _require_hippunfold_plot()
    density = _resolve_density(density)

    fig = plot_hipp_surf(
        surf_map=surf_map,
        density=density,
        hemi=hemi,
        space="unfold",
        view="dorsal",  # top-down on unfold
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        bg_on_data=True,
        alpha=0.1,
        darkness=2,
        dpi=dpi,
        figsize=(3, 4),
        colorbar=False,
    )
    return fig


def _fig_to_image(fig: Figure) -> np.ndarray:
    """Convert matplotlib Figure to RGB array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    plt.close(fig)
    return img


# ======================================================================
# §2  SINGLE HIPPOCAMPUS — 3D + FLATMAP
# ======================================================================

def plot_hippocampus(
    surf_map: Any,
    *,
    hemi: str = "left",
    density: str = "8k",
    views: Optional[List[str]] = None,
    show_flatmap: bool = True,
    flatmap_position: Literal["first", "last"] = "last",
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nan_color: Any = (0.85, 0.85, 0.85),
    style: str = "default",
    display_type: str = "static",
    title: str = "",
    save: Optional[PathLike] = None,
    formats: Optional[Union[str, List[str]]] = None,
) -> Tuple[Figure, List[Axes]]:
    """Single hippocampus: 3D multi-view + unfolded flatmap.

    Parameters
    ----------
    surf_map : str or ndarray
        GIfTI path or per-vertex array.
    hemi : str
        ``"left"`` or ``"right"``.
    density : str
        HippUnfold density (``"8k"``, ``"2k"``, ``"8k"``, ``"18k"``).
    views : list of str, optional
        3D view names.  Default: lateral, medial, dorsal, ventral, anterior.
    show_flatmap : bool
        Include unfolded flatmap column.
    flatmap_position : str
        ``"first"`` or ``"last"`` column.
    cmap : str
    vmin, vmax : float
    nan_color, style, display_type : styling.
    title : str
    save : PathLike, optional

    Returns
    -------
    fig, axes

    Examples
    --------
    >>> plot_hippocampus(
    ...     "sub-01_hemi-L_thickness.shape.gii",
    ...     hemi="left", density="8k",
    ...     cmap="inferno", vmin=1.0, vmax=4.0,
    ...     save="hippo_thickness.png",
    ... )
    """
    if views is None:
        views = HIPP_VIEWS_3D

    _apply_style()

    # Render each 3D view.
    view_imgs = []
    for v in views:
        fig_v = _render_hipp_3d(
            surf_map, view=v, hemi=hemi, density=density,
            cmap=cmap, vmin=vmin, vmax=vmax, nan_color=nan_color,
        )
        view_imgs.append(_fig_to_image(fig_v))

    # Render flatmap.
    flatmap_img = None
    if show_flatmap:
        fig_flat = _render_hipp_flatmap(
            surf_map, hemi=hemi, density=density,
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        flatmap_img = _fig_to_image(fig_flat)

    # Compose into panel.
    n_cols = len(views) + (1 if show_flatmap else 0)
    fig, axes_row = plt.subplots(
        1, n_cols,
        figsize=(2.5 * n_cols, 3),
        dpi=DPI,
    )
    if n_cols == 1:
        axes_row = [axes_row]

    col_labels = list(views)
    col_images = list(view_imgs)
    if show_flatmap:
        if flatmap_position == "first":
            col_images.insert(0, flatmap_img)
            col_labels.insert(0, "unfolded")
        else:
            col_images.append(flatmap_img)
            col_labels.append("unfolded")

    import seaborn as sns
    for ax, img, label in zip(axes_row, col_images, col_labels):
        ax.imshow(img, aspect="auto", interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=7)
        sns.despine(ax=ax, left=True, bottom=True, top=True, right=True)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, list(axes_row)


# ======================================================================
# §3  BILATERAL PANEL
# ======================================================================

def plot_hippocampus_bilateral(
    surf_map_left: Any,
    surf_map_right: Any,
    *,
    density: str = "8k",
    views: Optional[List[str]] = None,
    show_flatmap: bool = True,
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nan_color: Any = (0.85, 0.85, 0.85),
    style: str = "default",
    display_type: str = "static",
    title: str = "Bilateral Hippocampus",
    save: Optional[PathLike] = None,
    formats: Optional[Union[str, List[str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Two-row bilateral panel: L (top) + R (bottom).

    Parameters
    ----------
    surf_map_left, surf_map_right : str or ndarray
    """
    if views is None:
        views = ["lateral", "medial", "dorsal", "anterior"]

    _apply_style()

    all_rows = []
    row_labels = ["Left", "Right"]

    for hemi, smap in [("left", surf_map_left), ("right", surf_map_right)]:
        row_imgs = []
        for v in views:
            fig_v = _render_hipp_3d(
                smap, view=v, hemi=hemi, density=density,
                cmap=cmap, vmin=vmin, vmax=vmax, nan_color=nan_color,
            )
            row_imgs.append(_fig_to_image(fig_v))

        if show_flatmap:
            fig_flat = _render_hipp_flatmap(
                smap, hemi=hemi, density=density,
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            row_imgs.append(_fig_to_image(fig_flat))

        all_rows.append(row_imgs)

    n_cols = len(views) + (1 if show_flatmap else 0)
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 3 * n_rows),
        dpi=DPI,
    )

    import seaborn as sns
    col_labels = list(views) + (["unfolded"] if show_flatmap else [])
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(all_rows[i][j], aspect="auto", interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True, top=True, right=True)
            if i == n_rows - 1:
                ax.set_xlabel(col_labels[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=8, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=10)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, axes


# ======================================================================
# §4  GROUP COMPARISON
# ======================================================================

def plot_hippocampus_comparison(
    group_a_map: Any,
    group_b_map: Any,
    diff_map: Optional[Any] = None,
    *,
    hemi: str = "left",
    density: str = "8k",
    views: Optional[List[str]] = None,
    show_flatmap: bool = True,
    cmap_groups: str = "inferno",
    cmap_diff: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmin_diff: float = -3.0,
    vmax_diff: float = 3.0,
    nan_color: Any = (0.85, 0.85, 0.85),
    style: str = "default",
    display_type: str = "static",
    row_labels: Optional[List[str]] = None,
    title: str = "Group Comparison",
    save: Optional[PathLike] = None,
    formats: Optional[Union[str, List[str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """2–3 row group comparison: A, B, [A−B].

    Parameters
    ----------
    group_a_map, group_b_map : str or ndarray
        Mean descriptor map per group.
    diff_map : str or ndarray, optional
        A − B difference (or t-map / z-map).
    """
    if views is None:
        views = ["lateral", "medial", "dorsal", "anterior"]
    if row_labels is None:
        row_labels = ["Control", "Patient"]
        if diff_map is not None:
            row_labels.append("Difference")

    _apply_style()

    specs = [
        (group_a_map, cmap_groups, vmin, vmax),
        (group_b_map, cmap_groups, vmin, vmax),
    ]
    if diff_map is not None:
        specs.append((diff_map, cmap_diff, vmin_diff, vmax_diff))

    all_rows = []
    for smap, cm, vmn, vmx in specs:
        row_imgs = []
        for v in views:
            fig_v = _render_hipp_3d(
                smap, view=v, hemi=hemi, density=density,
                cmap=cm, vmin=vmn, vmax=vmx, nan_color=nan_color,
            )
            row_imgs.append(_fig_to_image(fig_v))
        if show_flatmap:
            fig_flat = _render_hipp_flatmap(
                smap, hemi=hemi, density=density,
                cmap=cm, vmin=vmn, vmax=vmx,
            )
            row_imgs.append(_fig_to_image(fig_flat))
        all_rows.append(row_imgs)

    n_cols = len(views) + (1 if show_flatmap else 0)
    n_rows = len(specs)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
        dpi=DPI,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    import seaborn as sns
    col_labels = list(views) + (["unfolded"] if show_flatmap else [])
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(all_rows[i][j], aspect="auto", interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True, top=True, right=True)
            if i == n_rows - 1:
                ax.set_xlabel(col_labels[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(
                    row_labels[i], fontsize=8, fontweight="bold",
                    rotation=0, ha="right", va="center", labelpad=10,
                )

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, axes


# ======================================================================
# §5  DESCRIPTOR GALLERY
# ======================================================================

def plot_hippocampus_gallery(
    descriptors: Dict[str, Any],
    *,
    hemi: str = "left",
    density: str = "8k",
    views: Optional[List[str]] = None,
    show_flatmap: bool = True,
    nan_color: Any = (0.85, 0.85, 0.85),
    style: str = "default",
    display_type: str = "static",
    title: str = "Hippocampal Spectral Gallery",
    save: Optional[PathLike] = None,
    formats: Optional[Union[str, List[str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Multi-row descriptor gallery — one row per descriptor.

    Parameters
    ----------
    descriptors : dict of {name: surf_map}
        Keys should match HIPP_DESCRIPTOR_STYLES for auto-styling.
        Values are GIfTI paths or per-vertex arrays.
    hemi : str
    density : str
    views : list of str
    show_flatmap : bool
    title : str
    save : PathLike, optional

    Returns
    -------
    fig, axes

    Examples
    --------
    >>> plot_hippocampus_gallery(
    ...     {"thickness": thick_gii, "hks": hks_array,
    ...      "wks": wks_array, "bks": bks_array},
    ...     hemi="left", density="8k",
    ...     save="hippo_gallery.png",
    ... )
    """
    if views is None:
        views = ["lateral", "medial", "dorsal"]

    _apply_style()
    from spectralbrain.runtime import progress_simple

    all_rows = []
    row_labels = []

    desc_names = list(descriptors.keys())
    with progress_simple("Rendering hippocampal gallery", total=len(desc_names)) as tick:
        for name in desc_names:
            smap = descriptors[name]
            sty = HIPP_DESCRIPTOR_STYLES.get(name, {})
            cm = sty.get("cmap", "inferno")
            vmn = sty.get("vmin")
            vmx = sty.get("vmax")
            label = sty.get("label", name)
            row_labels.append(label)

            row_imgs = []
            for v in views:
                fig_v = _render_hipp_3d(
                    smap, view=v, hemi=hemi, density=density,
                    cmap=cm, vmin=vmn, vmax=vmx, nan_color=nan_color,
                )
                row_imgs.append(_fig_to_image(fig_v))

            if show_flatmap:
                fig_flat = _render_hipp_flatmap(
                    smap, hemi=hemi, density=density,
                    cmap=cm, vmin=vmn, vmax=vmx,
                )
                row_imgs.append(_fig_to_image(fig_flat))

            all_rows.append(row_imgs)
            tick(1)

    n_cols = len(views) + (1 if show_flatmap else 0)
    n_rows = len(desc_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.2 * n_rows),
        dpi=DPI,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    import seaborn as sns
    col_labels = list(views) + (["unfolded"] if show_flatmap else [])
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(all_rows[i][j], aspect="auto", interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_color("#aaaaaa")
            if i == n_rows - 1:
                ax.set_xlabel(col_labels[j], fontsize=6)
            if j == 0:
                ax.set_ylabel(
                    row_labels[i], fontsize=7, fontweight="bold",
                    rotation=0, ha="right", va="center", labelpad=10,
                )

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, axes


# ======================================================================
# §6  NORMATIVE DEVIATION PANEL
# ======================================================================

def plot_hippocampus_normative(
    z_map: Any,
    *,
    hemi: str = "left",
    density: str = "8k",
    views: Optional[List[str]] = None,
    show_flatmap: bool = True,
    threshold: float = 2.0,
    cmap: str = "RdBu_r",
    vmin: float = -3.0,
    vmax: float = 3.0,
    nan_color: Any = (0.85, 0.85, 0.85),
    style: str = "default",
    display_type: str = "static",
    title: str = "Hippocampal Normative Deviation",
    save: Optional[PathLike] = None,
    formats: Optional[Union[str, List[str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Normative z-score map with thresholded view.

    Two rows: full z-map (top), thresholded |Z| > threshold (bottom).

    Parameters
    ----------
    z_map : str or ndarray
        Per-vertex z-scores.
    threshold : float
        Threshold for the second row.
    """
    descriptors = {"Z-score": z_map}

    # Build thresholded version.
    if isinstance(z_map, np.ndarray):
        thr_map = z_map.copy()
        thr_map[np.abs(thr_map) <= threshold] = np.nan
        descriptors[f"|Z| > {threshold}"] = thr_map

    # Override styles for this specific plot.
    HIPP_DESCRIPTOR_STYLES["Z-score"] = {
        "cmap": cmap, "vmin": vmin, "vmax": vmax, "label": "Z-score",
    }
    HIPP_DESCRIPTOR_STYLES[f"|Z| > {threshold}"] = {
        "cmap": cmap, "vmin": vmin, "vmax": vmax,
        "label": f"|Z| > {threshold}",
    }

    return plot_hippocampus_gallery(
        descriptors,
        hemi=hemi, density=density, views=views,
        show_flatmap=show_flatmap, nan_color=nan_color,
        title=title, save=save, formats=formats,
    )


# ======================================================================
# §6  SPATIO-TEMPORAL DESCRIPTOR FIELD ON HIPPOCAMPAL UNFOLDED SHEET
# ======================================================================

def plot_hippocampus_spatiotemporal(
    H: np.ndarray,
    *,
    hemi: Literal["left", "right"] = "left",
    density: str = "8k",
    t_values: Optional[np.ndarray] = None,
    n_panels: int = 8,
    t_indices: Optional[List[int]] = None,
    cmap: str = "magma",
    log_norm: bool = True,
    show_subfields: bool = True,
    descriptor_name: str = "HKS",
    unfolded_surf_path: Optional[Union[str, Path]] = None,
    subfield_label_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[PathLike] = None,
    formats: Tuple[str, ...] = ("png", "pdf"),
) -> Tuple[Figure, np.ndarray]:
    """Spatio-temporal descriptor field on the HippUnfold unfolded sheet.

    Wraps :func:`spectralbrain.viz.clusters.plot_spatiotemporal_field`
    with HippUnfold-aware defaults: loads the canonical unfolded
    mid-thickness surface and subfield labels automatically when paths
    are provided, applies the AP × PD coordinate convention, and uses
    neuroimaging-standard axis labels.

    This is the hippocampal-specific version of the general
    spatio-temporal small-multiples visualization.  Each panel shows
    how HKS/WKS varies across the unfolded hippocampal sheet at a
    different diffusion time, revealing the local → global
    progression and how it relates to subfield boundaries.

    Parameters
    ----------
    H : (V, T) array
        Per-vertex spectral descriptor matrix.  V must match the
        number of vertices in the unfolded surface.
    hemi : str
        ``"left"`` or ``"right"``.
    density : str
        HippUnfold density label (``"8k"``, ``"2k"``, ``"18k"``; v1: ``"0p5mm"``, ``"1mm"``, ``"2mm"``).
    t_values : (T,) array or None
        Scale parameter values.
    n_panels : int
    t_indices : list of int or None
    cmap : str
    log_norm : bool
    show_subfields : bool
        Overlay subfield boundary lines.
    descriptor_name : str
    unfolded_surf_path : str or Path or None
        Path to the unfolded ``midthickness.surf.gii``.  If None,
        attempts to load from hippunfold_plot's bundled data.
    subfield_label_path : str or Path or None
        Path to ``subfields.label.gii``.  If None and show_subfields
        is True, attempts to load from hippunfold_plot's bundled data.
    figsize : tuple or None
    save : PathLike or None
    formats : tuple of str
        Export formats.

    Returns
    -------
    (Figure, ndarray of Axes)

    Notes
    -----
    The unfolded surface from HippUnfold uses Laplace-solved coordinates
    (AP × PD), not physical millimetres.  Axes are labelled accordingly
    as "AP coordinate" and "PD coordinate" to avoid misleading the
    reader.  Metric distortion between folded and unfolded spaces does
    not affect the plotted descriptor values (HKS/WKS are isometry-
    invariant), but it does affect area-based statistics — use folded-
    surface vertex areas for any integration.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    H = np.asarray(H, dtype=np.float64)
    density = _resolve_density(density)

    # --- load unfolded surface ---
    if unfolded_surf_path is not None:
        surf_gii = nib.load(str(unfolded_surf_path))
        coords = surf_gii.agg_data("pointset")       # (V, 3)
        faces = surf_gii.agg_data("triangle")         # (F, 3)
        # project 3D → 2D (drop the flat z ≈ 0)
        unfolded_2d = coords[:, :2]
    else:
        # attempt to get from hippunfold_plot bundled data
        try:
            from hippunfold_plot.utils import get_surf_coords
            coords, faces = get_surf_coords(
                density=density, hemi=hemi, space="unfold",
            )
            unfolded_2d = coords[:, :2]
        except (ImportError, Exception) as e:
            raise ValueError(
                f"Could not load unfolded surface. Provide "
                f"unfolded_surf_path explicitly. Error: {e}"
            )

    # --- load subfield labels ---
    subfield_labels = None
    if show_subfields:
        if subfield_label_path is not None:
            lab_gii = nib.load(str(subfield_label_path))
            subfield_labels = lab_gii.agg_data().astype(np.int64)
        else:
            try:
                from hippunfold_plot.utils import get_label_data
                subfield_labels = get_label_data(
                    density=density, hemi=hemi,
                ).astype(np.int64)
            except (ImportError, Exception):
                logger.warning(
                    "Could not load subfield labels.  "
                    "Set show_subfields=False or provide subfield_label_path."
                )

    # --- validate shapes ---
    if H.shape[0] != unfolded_2d.shape[0]:
        raise ValueError(
            f"H has {H.shape[0]} vertices but the unfolded surface "
            f"has {unfolded_2d.shape[0]}.  They must match."
        )

    # --- delegate to the generic spatio-temporal field plotter ---
    from spectralbrain.viz.clusters import plot_spatiotemporal_field

    fig, axes = plot_spatiotemporal_field(
        unfolded_coords=unfolded_2d,
        faces=faces,
        H=H,
        t_values=t_values,
        n_panels=n_panels,
        t_indices=t_indices,
        cmap=cmap,
        log_norm=log_norm,
        subfield_labels=subfield_labels,
        descriptor_name=descriptor_name,
        xlabel="AP coordinate (Laplace)",
        ylabel="PD coordinate (Laplace)",
        figsize=figsize,
        save=save,
    )

    return fig, axes


def plot_hippocampus_hovmoller(
    H: np.ndarray,
    *,
    hemi: Literal["left", "right"] = "left",
    density: str = "8k",
    t_values: Optional[np.ndarray] = None,
    axis: Literal["AP", "PD"] = "AP",
    unfolded_surf_path: Optional[Union[str, Path]] = None,
    cmap: str = "viridis",
    descriptor_name: str = "HKS",
    figsize: Tuple[float, float] = (8, 4),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Hovmöller diagram of descriptor along hippocampal axis × scale.

    Wraps :func:`spectralbrain.viz.clusters.plot_hovmoller` with
    HippUnfold-aware coordinate loading.

    Parameters
    ----------
    H : (V, T) array
    hemi, density : str
    t_values : (T,) or None
    axis : str
        ``"AP"`` or ``"PD"``.
    unfolded_surf_path : str or None
    cmap, descriptor_name, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    H = np.asarray(H, dtype=np.float64)
    density = _resolve_density(density)

    if unfolded_surf_path is not None:
        surf_gii = nib.load(str(unfolded_surf_path))
        coords = surf_gii.agg_data("pointset")
        unfolded_2d = coords[:, :2]
    else:
        try:
            from hippunfold_plot.utils import get_surf_coords
            coords, _ = get_surf_coords(
                density=density, hemi=hemi, space="unfold",
            )
            unfolded_2d = coords[:, :2]
        except (ImportError, Exception) as e:
            raise ValueError(
                f"Could not load unfolded coordinates: {e}"
            )

    from spectralbrain.viz.clusters import plot_hovmoller

    return plot_hovmoller(
        unfolded_coords=unfolded_2d,
        H=H,
        t_values=t_values,
        axis=axis,
        cmap=cmap,
        descriptor_name=descriptor_name,
        title=f"Hovmöller — {descriptor_name} along {axis} ({hemi})",
        figsize=figsize,
        save=save,
    )


# ======================================================================

__all__ = [
    # Constants
    "HIPP_VIEWS_3D", "HIPP_VIEWS_FULL", "DENSITIES",
    "HIPP_DESCRIPTOR_STYLES",
    "HIPP_LABELS",
    # Single hippocampus
    "plot_hippocampus",
    # Bilateral
    "plot_hippocampus_bilateral",
    # Group comparison
    "plot_hippocampus_comparison",
    # Descriptor gallery
    "plot_hippocampus_gallery",
    # Normative
    "plot_hippocampus_normative",
    # Spatio-temporal
    "plot_hippocampus_spatiotemporal",
    "plot_hippocampus_hovmoller",
]
