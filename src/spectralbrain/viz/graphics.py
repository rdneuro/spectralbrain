"""Publication-quality graphics and statistical visualisations.

Foundation module for all SpectralBrain visualisations: palettes,
colormaps, figure factory, multi-format export (always PNG @600 dpi),
a custom distplot, and figure functions directly linked to
:mod:`spectralbrain.statistics.analysis`.

Every ``plot_*`` returns ``(fig, ax)`` for customisation before saving.
Every function accepts an optional ``save`` parameter for auto-export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

# ======================================================================
# §1  DEFAULTS & STYLE
# ======================================================================

DPI: int = 600

PALETTE = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "dark": "#222222",
    "orange": "#EE8866",
    "teal": "#44AA99",
    "indigo": "#332288",
    "rose": "#CC6677",
}

PALETTE_LIST: list[str] = [
    PALETTE["blue"],
    PALETTE["red"],
    PALETTE["green"],
    PALETTE["purple"],
    PALETTE["orange"],
    PALETTE["teal"],
    PALETTE["cyan"],
    PALETTE["yellow"],
    PALETTE["indigo"],
    PALETTE["rose"],
]

COLOR_CONTROL = PALETTE["blue"]
COLOR_PATIENT = PALETTE["red"]
COLOR_SIGNIFICANT = PALETTE["red"]
COLOR_NS = PALETTE["grey"]


def set_style(
    context: Literal["paper", "poster", "talk"] = "paper",
    font_scale: float = 1.0,
) -> None:
    """Apply SpectralBrain matplotlib style globally."""
    sizes = {
        "paper": {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8},
        "talk": {"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12},
        "poster": {"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16},
    }.get(context, {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})

    rc = {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": sizes["font.size"] * font_scale,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=PALETTE_LIST),
    }
    for k, v in sizes.items():
        rc[k] = v * font_scale
    plt.rcParams.update(rc)


set_style("paper")

# ======================================================================
# §2  COLORMAPS
# ======================================================================


def _reg(name, colors, N=256):
    """Register a plot function in the graphics registry."""
    cmap = mcolors.LinearSegmentedColormap.from_list(name, colors, N=N)
    try:
        mpl.colormaps.register(cmap, name=name, force=True)
    except Exception:
        plt.register_cmap(name=name, cmap=cmap)
    return cmap


CMAP_DIVERGING = _reg("sb_diverging", [PALETTE["blue"], "#FFFFFF", PALETTE["red"]])
CMAP_SEQUENTIAL = _reg(
    "sb_sequential",
    [PALETTE["indigo"], PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["yellow"]],
)
CMAP_SPECTRAL = _reg(
    "sb_spectral",
    [
        PALETTE["indigo"],
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["green"],
        PALETTE["yellow"],
        PALETTE["orange"],
        PALETTE["red"],
        PALETTE["purple"],
    ],
)
CMAP_QUALITATIVE = mcolors.ListedColormap(PALETTE_LIST, name="sb_qualitative")
try:
    mpl.colormaps.register(CMAP_QUALITATIVE, name="sb_qualitative", force=True)
except Exception:
    plt.register_cmap(name="sb_qualitative", cmap=CMAP_QUALITATIVE)


# ======================================================================
# §3  FIGURE FACTORY
# ======================================================================

_JOURNAL_PRESETS = {
    "nature": {"width": 89, "height": 60},
    "nature_wide": {"width": 183, "height": 80},
    "neuroimage": {"width": 85, "height": 60},
    "neuroimage_wide": {"width": 170, "height": 80},
    "joss": {"width": 140, "height": 90},
    "a4_half": {"width": 170, "height": 120},
    "slide_16x9": {"width": 254, "height": 143},
}


def figure(
    preset: str | None = None,
    *,
    width: float | None = None,
    height: float | None = None,
    nrows: int = 1,
    ncols: int = 1,
    unit: str = "mm",
    **subplot_kw,
) -> tuple[Figure, Any]:
    """Create a publication-ready figure."""
    if preset and preset in _JOURNAL_PRESETS:
        p = _JOURNAL_PRESETS[preset]
        w = width or p["width"]
        h = height or p["height"]
    else:
        w = width or 140
        h = height or 90
    if unit == "mm":
        w, h = w / 25.4, h / 25.4
    elif unit == "cm":
        w, h = w / 2.54, h / 2.54
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **subplot_kw)
    fig.set_dpi(DPI)
    return fig, axes


# ======================================================================
# §4  MULTI-FORMAT SAVE (always PNG + extras)
# ======================================================================


def savefig(
    fig: Figure,
    path: PathLike,
    *,
    formats: str | list[str] | None = None,
    dpi: int = DPI,
    transparent: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
) -> list[Path]:
    """Save figure — **always** PNG, plus optional PDF/SVG/JPG."""
    base = Path(path)
    stem = base.parent / base.stem
    base.parent.mkdir(parents=True, exist_ok=True)

    if formats is None:
        fmt_list = []
    elif isinstance(formats, str):
        fmt_list = [formats]
    else:
        fmt_list = list(formats)

    all_fmts = ["png"] + [f for f in fmt_list if f != "png"]
    kw = dict(dpi=dpi, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
    saved = []
    for fmt in all_fmts:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(str(out), format=fmt, **kw)
        saved.append(out)
        logger.info("Saved %s (%d dpi)", out.name, dpi)
    return saved


# ======================================================================
# §5  CUSTOM DISTPLOT
# ======================================================================


def distplot(
    data: np.ndarray | list[np.ndarray],
    *,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    hist: bool = True,
    kde: bool = True,
    rug: bool = False,
    bins: int | str = "auto",
    alpha_hist: float = 0.35,
    alpha_kde: float = 0.9,
    fill_kde: bool = True,
    alpha_fill: float = 0.15,
    vertical_lines: dict[str, float] | None = None,
    xlabel: str = "",
    ylabel: str = "Density",
    title: str = "",
    ax: Axes | None = None,
    save: PathLike | None = None,
    **save_kw,
) -> tuple[Figure, Axes]:
    """Custom distribution plot — histogram + KDE + rug.

    Drop-in replacement for seaborn's deprecated ``distplot`` with
    SpectralBrain styling and multi-group support.
    """
    from scipy.stats import gaussian_kde

    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = [data]
    n_groups = len(data)
    if colors is None:
        colors = PALETTE_LIST[:n_groups]
    if labels is None:
        labels = [None] * n_groups
    if ax is None:
        fig, ax = figure(width=120, height=75)
    else:
        fig = ax.figure

    for i, (d, color, label) in enumerate(zip(data, colors, labels)):
        d = np.asarray(d).ravel()
        d = d[np.isfinite(d)]
        if hist:
            ax.hist(
                d,
                bins=bins,
                density=True,
                alpha=alpha_hist,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label if not kde else None,
            )
        if kde and len(d) > 2:
            kf = gaussian_kde(d)
            xr = np.linspace(d.min() - d.std(), d.max() + d.std(), 300)
            yk = kf(xr)
            ax.plot(xr, yk, color=color, alpha=alpha_kde, linewidth=1.8, label=label)
            if fill_kde:
                ax.fill_between(xr, yk, alpha=alpha_fill, color=color)
        if rug:
            ax.plot(d, np.zeros_like(d) - 0.01 * (i + 1), "|", color=color, markersize=4, alpha=0.5)

    if vertical_lines:
        for vl_label, vl_val in vertical_lines.items():
            ax.axvline(
                vl_val,
                color=PALETTE["dark"],
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=vl_label,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if any(l is not None for l in labels) or vertical_lines:
        ax.legend(fontsize=7)
    if save:
        savefig(fig, save, **save_kw)
    return fig, ax


# ======================================================================
# §6  STATISTICAL FIGURES (linked to analysis.py)
# ======================================================================


def plot_volcano(
    effect_sizes: np.ndarray,
    p_values: np.ndarray,
    *,
    alpha: float = 0.05,
    effect_threshold: float = 0.0,
    xlabel: str = "Effect size (Cohen's d)",
    title: str = "Volcano plot",
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """Volcano plot: effect size vs −log₁₀(p).

    Visualises :func:`~spectralbrain.statistics.analysis.vertexwise_ttest`.
    """
    if ax is None:
        fig, ax = figure(width=110, height=90)
    else:
        fig = ax.figure

    log_p = -np.log10(np.clip(p_values, 1e-300, 1))
    sig = (p_values < alpha) & (np.abs(effect_sizes) > effect_threshold)

    ax.scatter(effect_sizes[~sig], log_p[~sig], s=4, alpha=0.3, color=COLOR_NS, rasterized=True)
    ax.scatter(
        effect_sizes[sig], log_p[sig], s=6, alpha=0.7, color=COLOR_SIGNIFICANT, rasterized=True
    )
    ax.axhline(-np.log10(alpha), color=PALETTE["dark"], ls="--", lw=0.8, alpha=0.5)
    if effect_threshold > 0:
        for et in [effect_threshold, -effect_threshold]:
            ax.axvline(et, color=PALETTE["dark"], ls=":", lw=0.8, alpha=0.4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title)
    ax.annotate(
        f"{sig.sum()} significant",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7,
        color=COLOR_SIGNIFICANT,
    )

    if save:
        savefig(fig, save)
    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    label: str | None = None,
    color: str | None = None,
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """ROC curve with AUC — visualises :func:`~...analysis.classify`."""
    from sklearn.metrics import roc_auc_score, roc_curve

    if ax is None:
        fig, ax = figure(width=90, height=85)
    else:
        fig = ax.figure

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    lbl = f"{label} (AUC={auc:.3f})" if label else f"AUC = {auc:.3f}"
    ax.plot(fpr, tpr, color=color or PALETTE["blue"], lw=1.8, label=lbl)
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    if save:
        savefig(fig, save)
    return fig, ax


def plot_rdm(
    rdm_matrix: np.ndarray,
    *,
    labels: list[str] | None = None,
    cmap: str = "sb_sequential",
    title: str = "Representational Dissimilarity Matrix",
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """RDM heatmap — visualises :func:`~...analysis.rdm`."""
    if ax is None:
        fig, ax = figure(width=100, height=90)
    else:
        fig = ax.figure

    im = ax.imshow(rdm_matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Dissimilarity")
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    if save:
        savefig(fig, save)
    return fig, ax


def plot_connectome_matrix(
    matrix: np.ndarray,
    *,
    labels: list[str] | None = None,
    network_boundaries: list[int] | None = None,
    cmap: str = "sb_spectral",
    title: str = "Geometric Connectome",
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """Connectome heatmap — visualises :func:`~...distances.build_geometric_connectome`."""
    if ax is None:
        fig, ax = figure(width=110, height=95)
    else:
        fig = ax.figure

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spectral distance")
    if network_boundaries:
        for b in network_boundaries:
            ax.axhline(b - 0.5, color="white", lw=1.5)
            ax.axvline(b - 0.5, color="white", lw=1.5)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(title)
    if save:
        savefig(fig, save)
    return fig, ax


def plot_embedding(
    coords: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    group_names: dict[int, str] | None = None,
    colors: list[str] | None = None,
    method_name: str = "Embedding",
    alpha: float = 0.7,
    size: float = 15,
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """2D scatter — visualises PCA/MDS/UMAP from analysis.py."""
    if ax is None:
        fig, ax = figure(width=110, height=90)
    else:
        fig = ax.figure

    if labels is None:
        ax.scatter(
            coords[:, 0], coords[:, 1], s=size, alpha=alpha, color=PALETTE["blue"], rasterized=True
        )
    else:
        unique = np.unique(labels)
        cols = colors or PALETTE_LIST[: len(unique)]
        for i, u in enumerate(unique):
            mask = labels == u
            name = group_names.get(u, str(u)) if group_names else str(u)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=size,
                alpha=alpha,
                color=cols[i % len(cols)],
                label=name,
                rasterized=True,
            )
        ax.legend(markerscale=1.5, fontsize=7)
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(method_name)
    if save:
        savefig(fig, save)
    return fig, ax


def plot_effect_size_distribution(
    effect_sizes: np.ndarray,
    *,
    threshold: float = 0.5,
    title: str = "Vertex-wise effect sizes",
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """Cohen's d distribution — visualises :func:`~...analysis.cohens_d_map`."""
    fig, ax = distplot(
        effect_sizes,
        hist=True,
        kde=True,
        xlabel="Cohen's d",
        title=title,
        ax=ax,
        vertical_lines={f"|d|={threshold}": threshold, f"|d|={-threshold}": -threshold},
    )
    n_large = np.sum(np.abs(effect_sizes) > threshold)
    pct = 100 * n_large / len(effect_sizes)
    ax.annotate(
        f"{n_large} vertices ({pct:.1f}%) |d| > {threshold}",
        xy=(0.98, 0.92),
        xycoords="axes fraction",
        ha="right",
        fontsize=7,
        color=PALETTE["dark"],
    )
    if save:
        savefig(fig, save)
    return fig, ax


def plot_laterality(
    left: np.ndarray,
    right: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    group_names: dict[int, str] | None = None,
    title: str = "Lateralisation",
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """Paired violin L vs R — visualises :func:`~...analysis.asymmetry_test`."""
    if ax is None:
        fig, ax = figure(width=100, height=85)
    else:
        fig = ax.figure

    if labels is None:
        for data, pos, col in [(left, 0, PALETTE["blue"]), (right, 1, PALETTE["red"])]:
            parts = ax.violinplot(data, positions=[pos], showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(0.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Left", "Right"])
    else:
        unique = np.unique(labels)
        gn = group_names or {u: f"Group {u}" for u in unique}
        positions, tick_labels, pos = [], [], 0
        for u in unique:
            mask = labels == u
            for data, col, side in [
                (left[mask], PALETTE["blue"], "L"),
                (right[mask], PALETTE["red"], "R"),
            ]:
                parts = ax.violinplot(data, positions=[pos], showmedians=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(col)
                    pc.set_alpha(0.4)
                positions.append(pos)
                tick_labels.append(f"{gn[u]} {side}")
                pos += 1
            pos += 1  # gap between groups
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")

    ax.set_ylabel("Descriptor value")
    ax.set_title(title)
    if save:
        savefig(fig, save)
    return fig, ax


def plot_pvalue_histogram(
    p_values: np.ndarray,
    *,
    alpha: float = 0.05,
    title: str = "P-value distribution",
    ax: Axes | None = None,
    save: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """P-value diagnostic histogram — for :func:`~...analysis.vertexwise_ttest`."""
    if ax is None:
        fig, ax = figure(width=100, height=70)
    else:
        fig = ax.figure

    ax.hist(
        p_values,
        bins=50,
        density=True,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.axhline(1.0, color=PALETTE["grey"], ls="--", lw=1, label="Uniform null")
    ax.axvline(alpha, color=PALETTE["red"], ls="--", lw=1, alpha=0.7, label=f"α = {alpha}")
    pct = 100 * np.mean(p_values < alpha)
    ax.annotate(
        f"{pct:.1f}% < α",
        xy=(alpha + 0.02, ax.get_ylim()[1] * 0.9),
        fontsize=7,
        color=PALETTE["red"],
    )
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=7)
    if save:
        savefig(fig, save)
    return fig, ax


# ======================================================================

__all__ = [
    "CMAP_DIVERGING",
    "CMAP_QUALITATIVE",
    "CMAP_SEQUENTIAL",
    "CMAP_SPECTRAL",
    "COLOR_CONTROL",
    "COLOR_PATIENT",
    "DPI",
    "PALETTE",
    "PALETTE_LIST",
    "distplot",
    "figure",
    "plot_connectome_matrix",
    "plot_effect_size_distribution",
    "plot_embedding",
    "plot_laterality",
    "plot_pvalue_histogram",
    "plot_rdm",
    "plot_roc_curve",
    "plot_volcano",
    "savefig",
    "set_style",
]
