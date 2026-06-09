"""Publication-quality brain surface visualisations for spectral morphometry.

Architecture
------------
**yabplot** (PyVista/VTK) renders 3D brain surfaces to high-resolution
PNGs.  **matplotlib** composites these PNGs into multi-row panels
styled by **scienceplots** (``science`` + ``no-latex``).

Every public function:
- Accepts ``nan_color``, ``style``, and ``display_type`` arguments.
- Returns ``(fig, axes)`` for further customisation.
- Accepts ``save=`` for automatic multi-format export (always PNG @600 dpi).

Figure types
------------
1. Single metric brain plot (cortical or subcortical)
2. Group comparison panel (A vs B vs A−B)
3. Normative deviation map (z-scores)
4. Atlas-free clustering map
5. Morphometric descriptor gallery (4–10 descriptors stacked)
6. Multi-descriptor comparison panel
7. Bilateral (L vs R) comparison
8. Spectral progression (HKS across t / WKS across e)
9. Tract visualisation
10. Subcortical structure panel
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

DPI: int = 600

VIEWS_CORTEX: list[str] = [
    "right_lateral",
    "anterior",
    "left_lateral",
    "posterior",
    "superior",
    "inferior",
]
"""Standard 6-view cortical row."""

VIEWS_FULL: list[str] = [*VIEWS_CORTEX, "subcortex"]
"""7-view cortical + subcortical row."""

VIEWS_MEDIAL: list[str] = [
    "left_lateral",
    "left_medial",
    "right_medial",
    "right_lateral",
]
"""Classic 4-view medial/lateral row."""

# ── Default descriptor visual specs ───────────────────────────────────

DESCRIPTOR_STYLES: dict[str, dict[str, Any]] = {
    "hks": {"cmap": "inferno", "vminmax": [None, None], "label": "HKS"},
    "wks": {"cmap": "cividis", "vminmax": [None, None], "label": "WKS"},
    "si_hks": {"cmap": "viridis", "vminmax": [None, None], "label": "SI-HKS"},
    "bks": {"cmap": "magma", "vminmax": [None, None], "label": "BKS"},
    "ibks": {"cmap": "magma", "vminmax": [None, None], "label": "IBKS"},
    "gps": {"cmap": "coolwarm", "vminmax": [None, None], "label": "GPS"},
    "shapedna": {"cmap": "plasma", "vminmax": [None, None], "label": "ShapeDNA"},
    "bates_sp": {"cmap": "inferno", "vminmax": [None, None], "label": "Bates SP"},
    "gaussian_k": {"cmap": "RdBu_r", "vminmax": [None, None], "label": "Gaussian K"},
    "mean_k": {"cmap": "RdBu_r", "vminmax": [None, None], "label": "Mean H"},
    "shape_idx": {"cmap": "RdBu_r", "vminmax": [-1, 1], "label": "Shape Index"},
    "casorati": {"cmap": "magma", "vminmax": [None, None], "label": "Casorati"},
    "curvedness": {"cmap": "magma", "vminmax": [None, None], "label": "Curvedness"},
    "willmore": {"cmap": "inferno", "vminmax": [None, None], "label": "Willmore H²"},
    "z_score": {"cmap": "RdBu_r", "vminmax": [-3, 3], "label": "Z-score"},
    "effect_d": {"cmap": "RdBu_r", "vminmax": [-1.5, 1.5], "label": "Cohen's d"},
    "clusters": {"cmap": "tab10", "vminmax": [None, None], "label": "Clusters"},
    "normative": {"cmap": "coolwarm", "vminmax": [-3, 3], "label": "Normative Z"},
}


@dataclass
class BrainPlotSpec:
    """Visual specification for one brain plot row.

    Keeps colour range, cmap, and labels consistent across panels.

    Parameters
    ----------
    label : str
        Row label (left margin annotation).
    data : dict or ndarray or None
        Data for yabplot (parcellated dict or vertex-wise mesh).
    cmap : str
        Matplotlib colourmap name.
    vminmax : list
        [vmin, vmax]; [None, None] = auto from data.
    nan_color : tuple or str
        Colour for NaN / medial wall / missing regions.
    plot_kind : str
        ``"cortical"``, ``"subcortical"``, ``"tracts"``, ``"vertexwise"``.
    atlas : str or None
        Atlas name for parcellated data.
    extra_kwargs : dict
        Additional kwargs passed to the yabplot function.
    """

    label: str = ""
    data: Any = None
    cmap: str = "coolwarm"
    vminmax: list[float | None] = field(default_factory=lambda: [None, None])
    nan_color: Any = (1.0, 1.0, 1.0)
    plot_kind: str = "cortical"
    atlas: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_descriptor(
        cls,
        descriptor_name: str,
        data: Any = None,
        **overrides: Any,
    ) -> BrainPlotSpec:
        """Build a spec from a known descriptor name."""
        style = DESCRIPTOR_STYLES.get(descriptor_name, {})
        return cls(
            label=style.get("label", descriptor_name),
            data=data,
            cmap=overrides.get("cmap", style.get("cmap", "coolwarm")),
            vminmax=overrides.get("vminmax", style.get("vminmax", [None, None])),
            nan_color=overrides.get("nan_color", (1.0, 1.0, 1.0)),
            plot_kind=overrides.get("plot_kind", "cortical"),
            atlas=overrides.get("atlas"),
            extra_kwargs=overrides.get("extra_kwargs", {}),
        )


# ======================================================================
# §1  INTERNAL RENDERING ENGINE
# ======================================================================


def _require_yabplot():
    """Lazy-import yabplot for 3D brain visualisation."""
    try:
        import yabplot as yab

        return yab
    except ImportError as exc:
        raise ImportError(
            "yabplot is required for brain surface plots.\n  pip install yabplot"
        ) from exc


def _require_scienceplots():
    """Lazy-import scienceplots for publication styling."""
    try:
        import scienceplots  # noqa: F401

        return True
    except ImportError:
        logger.debug("scienceplots not installed — using SpectralBrain style.")
        return False


def _apply_style():
    """Apply scienceplots if available, else SpectralBrain defaults."""
    has_sp = _require_scienceplots()
    if has_sp:
        plt.style.use(["science", "no-latex"])
        # Near-LaTeX math rendering without TeX installation.
        plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["figure.dpi"] = DPI


def _get_plot_fn(kind: str):
    """Map plot_kind string to yabplot function."""
    yab = _require_yabplot()
    fns = {
        "cortical": yab.plot_cortical,
        "subcortical": yab.plot_subcortical,
        "tracts": yab.plot_tracts,
        "vertexwise": yab.plot_vertexwise,
    }
    if kind not in fns:
        raise ValueError(f"Unknown plot_kind: {kind!r}. Use: {list(fns)}")
    return fns[kind]


def _render_row(
    spec: BrainPlotSpec,
    out_png: Path,
    *,
    views: list[str],
    style: str = "matte",
    display_type: str = "none",
    figsize_px: tuple[int, int] = (3600, 600),
) -> Path:
    """Render one brain row to PNG via yabplot.

    Parameters
    ----------
    spec : BrainPlotSpec
    out_png : Path
        Output PNG path.
    views : list of str
    style : str
        yabplot lighting style.
    display_type : str
    figsize_px : (width, height) in pixels.

    Returns
    -------
    Path
    """
    fn = _get_plot_fn(spec.plot_kind)

    kwargs = {
        "views": views,
        "layout": (1, len(views)),
        "figsize": figsize_px,
        "cmap": spec.cmap,
        "vminmax": spec.vminmax,
        "nan_color": spec.nan_color,
        "style": style,
        "display_type": display_type,
        "export_path": str(out_png),
    }

    # Kind-specific args.
    if spec.plot_kind in ("cortical", "subcortical", "tracts"):
        if spec.data is not None:
            kwargs["data"] = spec.data
        if spec.atlas is not None:
            kwargs["atlas"] = spec.atlas
    elif spec.plot_kind == "vertexwise":
        # vertexwise expects (lh, rh) as positional args.
        if isinstance(spec.data, tuple) and len(spec.data) == 2:
            kwargs.pop("data", None)
            kwargs.pop("atlas", None)
            fn(spec.data[0], spec.data[1], **kwargs)
            return out_png
        else:
            raise ValueError("vertexwise plot_kind requires data=(lh_mesh, rh_mesh)")

    kwargs.update(spec.extra_kwargs)
    fn(**kwargs)
    return out_png


def _compose_panel(
    row_images: list[np.ndarray],
    row_labels: list[str],
    *,
    panel_width_in: float = 12.0,
    row_height_in: float = 1.6,
    title: str = "",
    dpi: int = DPI,
    label_fontsize: int = 8,
    title_fontsize: int = 10,
    border: bool = True,
    border_color: str = "#888888",
    border_width: float = 0.5,
) -> tuple[Figure, list[Axes]]:
    """Compose rendered PNG rows into a matplotlib figure.

    Parameters
    ----------
    row_images : list of ndarray
        Each is an RGBA/RGB image array from ``mpimg.imread``.
    row_labels : list of str
    panel_width_in, row_height_in : float
    title : str
    dpi : int
    border : bool
        Draw thin border around each row (scienceplots-style framing).

    Returns
    -------
    fig, axes
    """
    import seaborn as sns

    _apply_style()

    n_rows = len(row_images)
    fig_height = row_height_in * n_rows + (0.4 if title else 0.1)

    fig = plt.figure(figsize=(panel_width_in, fig_height), dpi=dpi)
    gs = GridSpec(
        n_rows,
        1,
        figure=fig,
        hspace=0.03,
        top=0.95 if title else 0.98,
        bottom=0.02,
        left=0.08,
        right=0.98,
    )

    axes = []
    for i, (img, label) in enumerate(zip(row_images, row_labels)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, aspect="auto", interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])

        if border:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(border_width)
                spine.set_color(border_color)
        else:
            sns.despine(ax=ax, left=True, bottom=True, top=True, right=True)

        if label:
            ax.set_ylabel(
                label,
                rotation=0,
                ha="right",
                va="center",
                fontsize=label_fontsize,
                labelpad=12,
                fontweight="bold",
            )

        axes.append(ax)

    if title:
        fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", y=0.98)

    return fig, axes


def _save_figure(
    fig: Figure,
    path: PathLike,
    *,
    formats: str | list[str] | None = None,
    dpi: int = DPI,
) -> list[Path]:
    """Save figure — always PNG + optional extras."""
    from spectralbrain.viz.graphics import savefig

    return savefig(fig, path, formats=formats, dpi=dpi)


# ======================================================================
# §2  PUBLIC API — individual & single-row plots
# ======================================================================


def plot_brain(
    data: Any = None,
    *,
    atlas: str | None = None,
    plot_kind: str = "cortical",
    cmap: str = "coolwarm",
    vminmax: list[float] | None = None,
    nan_color: Any = (1.0, 1.0, 1.0),
    style: str = "matte",
    display_type: str = "none",
    views: list[str] | None = None,
    title: str = "",
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Single-row brain surface plot.

    The basic building block — renders one metric across 6–7 views.

    Parameters
    ----------
    data : dict or (lh_mesh, rh_mesh) or None
        Parcellated dict for cortical/subcortical, or (lh, rh)
        pyvista.PolyData tuple for vertex-wise.
    atlas : str, optional
    plot_kind : str
        ``"cortical"``, ``"subcortical"``, ``"tracts"``, ``"vertexwise"``.
    cmap : str
    vminmax : [vmin, vmax] or None
    nan_color : tuple or str
    style : str
        yabplot lighting (``"matte"``, ``"glossy"``, ``"sculpted"``, ``"flat"``).
    display_type : str
        ``"none"`` for batch, ``"static"`` for notebooks.
    views : list of str, optional
        Default: 6-view cortical row.
    title : str
    save : PathLike, optional
    formats : str or list, optional

    Returns
    -------
    fig, ax

    Examples
    --------
    >>> plot_brain(z_scores, atlas='schaefer_200', cmap='RdBu_r',
    ...            vminmax=[-3, 3], save='cortical_z.png')
    """
    if views is None:
        views = VIEWS_CORTEX

    spec = BrainPlotSpec(
        label="",
        data=data,
        cmap=cmap,
        vminmax=vminmax or [None, None],
        nan_color=nan_color,
        plot_kind=plot_kind,
        atlas=atlas,
        extra_kwargs=kwargs,
    )

    tmp = Path(tempfile.mkdtemp())
    png = tmp / "brain_row.png"
    px_w = int(12.0 * DPI)
    px_h = int(1.8 * DPI)

    _render_row(
        spec, png, views=views, style=style, display_type=display_type, figsize_px=(px_w, px_h)
    )

    img = mpimg.imread(str(png))
    fig, axes = _compose_panel([img], [""], title=title, border=False)
    ax = axes[0]

    if save:
        _save_figure(fig, save, formats=formats)

    return fig, ax


def plot_brain_subcortical(
    data: Any = None,
    *,
    atlas: str = "aseg",
    cmap: str = "RdBu_r",
    vminmax: list[float] | None = None,
    nan_color: str = "#cccccc",
    nan_alpha: float = 0.3,
    style: str = "sculpted",
    bmesh_alpha: float = 0.08,
    views: list[str] | None = None,
    title: str = "",
    save: PathLike | None = None,
    display_type: str = "none",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Single-row subcortical structure plot.

    Parameters
    ----------
    data : dict of {structure_name: value}
    atlas : str
    nan_alpha : float
        Transparency for structures without data.
    bmesh_alpha : float
        Ghost cortex translucency.
    """
    if views is None:
        views = ["left_lateral", "superior", "right_lateral"]

    spec = BrainPlotSpec(
        data=data,
        cmap=cmap,
        vminmax=vminmax or [None, None],
        nan_color=nan_color,
        plot_kind="subcortical",
        atlas=atlas,
        extra_kwargs={
            "nan_alpha": nan_alpha,
            "bmesh_alpha": bmesh_alpha,
            **kwargs,
        },
    )

    tmp = Path(tempfile.mkdtemp())
    png = tmp / "subcort_row.png"
    _render_row(
        spec,
        png,
        views=views,
        style=style,
        display_type=display_type,
        figsize_px=(int(10 * DPI), int(2.0 * DPI)),
    )

    img = mpimg.imread(str(png))
    fig, axes = _compose_panel([img], [""], title=title, border=False)

    if save:
        _save_figure(fig, save)
    return fig, axes[0]


# ======================================================================
# §3  GROUP COMPARISON
# ======================================================================


def plot_group_comparison(
    group_a: BrainPlotSpec,
    group_b: BrainPlotSpec,
    difference: BrainPlotSpec | None = None,
    *,
    views: list[str] | None = None,
    style: str = "matte",
    display_type: str = "none",
    title: str = "Group Comparison",
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Two- or three-row group comparison panel.

    Parameters
    ----------
    group_a : BrainPlotSpec
        Control group (typically blue-ish cmap).
    group_b : BrainPlotSpec
        Patient group.
    difference : BrainPlotSpec, optional
        A − B difference map (diverging cmap, symmetric vminmax).
    views : list of str
    style, display_type : str
    title : str
    save : PathLike, optional

    Returns
    -------
    fig, axes
    """
    if views is None:
        views = VIEWS_CORTEX

    specs = [group_a, group_b]
    if difference is not None:
        specs.append(difference)

    tmp = Path(tempfile.mkdtemp())
    images = []
    labels = []
    px_w, px_h = int(12 * DPI), int(1.5 * DPI)

    for i, spec in enumerate(specs):
        png = tmp / f"row_{i}.png"
        _render_row(
            spec, png, views=views, style=style, display_type=display_type, figsize_px=(px_w, px_h)
        )
        images.append(mpimg.imread(str(png)))
        labels.append(spec.label)

    fig, axes = _compose_panel(images, labels, title=title)

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, axes


# ======================================================================
# §4  NORMATIVE MAP
# ======================================================================


def plot_normative_map(
    z_data: Any,
    *,
    atlas: str | None = None,
    plot_kind: str = "cortical",
    threshold: float = 2.0,
    cmap: str = "coolwarm",
    vminmax: list[float] | None = None,
    nan_color: Any = (0.85, 0.85, 0.85),
    style: str = "matte",
    display_type: str = "none",
    views: list[str] | None = None,
    title: str = "Normative Deviation",
    show_thresholded: bool = True,
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Normative z-score map with optional thresholded view.

    Parameters
    ----------
    z_data : dict or (lh, rh)
        Z-score values.
    threshold : float
        Threshold for the second row (if show_thresholded=True).
    show_thresholded : bool
        Show a second row with only extreme deviations.

    Returns
    -------
    fig, axes
    """
    if views is None:
        views = VIEWS_CORTEX

    vm = vminmax or [-3, 3]

    spec_full = BrainPlotSpec(
        label="Z-score",
        data=z_data,
        cmap=cmap,
        vminmax=vm,
        nan_color=nan_color,
        plot_kind=plot_kind,
        atlas=atlas,
    )

    specs = [spec_full]

    if show_thresholded:
        # Threshold: set values within [-thr, thr] to NaN.
        if isinstance(z_data, dict):
            thr_data = {k: (v if abs(v) > threshold else float("nan")) for k, v in z_data.items()}
        else:
            thr_data = z_data  # user handles thresholding for vertex-wise

        spec_thr = BrainPlotSpec(
            label=f"|Z| > {threshold}",
            data=thr_data,
            cmap=cmap,
            vminmax=vm,
            nan_color=nan_color,
            plot_kind=plot_kind,
            atlas=atlas,
        )
        specs.append(spec_thr)

    tmp = Path(tempfile.mkdtemp())
    images, labels = [], []
    px_w, px_h = int(12 * DPI), int(1.5 * DPI)

    for i, spec in enumerate(specs):
        png = tmp / f"norm_{i}.png"
        _render_row(
            spec, png, views=views, style=style, display_type=display_type, figsize_px=(px_w, px_h)
        )
        images.append(mpimg.imread(str(png)))
        labels.append(spec.label)

    fig, axes = _compose_panel(images, labels, title=title)

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, axes


# ======================================================================
# §5  CLUSTERING MAP
# ======================================================================


def plot_clustering_map(
    cluster_data: Any,
    *,
    atlas: str | None = None,
    plot_kind: str = "cortical",
    cmap: str = "tab10",
    nan_color: Any = (0.9, 0.9, 0.9),
    style: str = "matte",
    display_type: str = "none",
    views: list[str] | None = None,
    title: str = "Atlas-Free Clustering",
    save: PathLike | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Visualise atlas-free clustering on brain surface.

    Parameters
    ----------
    cluster_data : dict or (lh, rh)
        Cluster labels per region or per vertex.
    """
    if views is None:
        views = VIEWS_CORTEX

    return plot_brain(
        data=cluster_data,
        atlas=atlas,
        plot_kind=plot_kind,
        cmap=cmap,
        nan_color=nan_color,
        style=style,
        display_type=display_type,
        views=views,
        title=title,
        save=save,
        **kwargs,
    )


# ======================================================================
# §6  MORPHOMETRIC DESCRIPTOR GALLERY (4–10 rows)
# ======================================================================


def plot_morphometric_gallery(
    specs: list[BrainPlotSpec],
    *,
    views: list[str] | None = None,
    style: str = "matte",
    display_type: str = "none",
    title: str = "Spectral Morphometry Gallery",
    panel_width_in: float = 12.0,
    row_height_in: float = 1.4,
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Multi-row panel with one descriptor per row.

    The flagship figure for spectral morphometry papers — stack
    4–10 descriptors with consistent views for visual comparison.

    Parameters
    ----------
    specs : list of BrainPlotSpec
        One spec per row.  Use ``BrainPlotSpec.from_descriptor()``
        for standard styling.
    views : list of str
        Default: 6-view cortical.
    style : str
    title : str
    panel_width_in, row_height_in : float
    save : PathLike, optional

    Returns
    -------
    fig, axes

    Examples
    --------
    >>> specs = [
    ...     BrainPlotSpec.from_descriptor("hks",       data=hks_dict),
    ...     BrainPlotSpec.from_descriptor("wks",       data=wks_dict),
    ...     BrainPlotSpec.from_descriptor("bks",       data=bks_dict),
    ...     BrainPlotSpec.from_descriptor("shape_idx", data=si_dict),
    ...     BrainPlotSpec.from_descriptor("casorati",  data=cas_dict),
    ... ]
    >>> fig, axes = plot_morphometric_gallery(specs, save='gallery.png')
    """
    if views is None:
        views = VIEWS_CORTEX

    from spectralbrain.runtime import progress_simple

    tmp = Path(tempfile.mkdtemp())
    images, labels = [], []
    px_w = int(panel_width_in * DPI)
    px_h = int(row_height_in * DPI)

    with progress_simple("Rendering gallery", total=len(specs)) as tick:
        for i, spec in enumerate(specs):
            png = tmp / f"gallery_{i}.png"
            _render_row(
                spec,
                png,
                views=views,
                style=style,
                display_type=display_type,
                figsize_px=(px_w, px_h),
            )
            images.append(mpimg.imread(str(png)))
            labels.append(spec.label)
            tick(1)

    fig, axes = _compose_panel(
        images,
        labels,
        panel_width_in=panel_width_in,
        row_height_in=row_height_in,
        title=title,
    )

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, axes


def plot_top10_morphometrics(
    descriptor_data: dict[str, Any],
    *,
    atlas: str | None = None,
    plot_kind: str = "cortical",
    views: list[str] | None = None,
    style: str = "matte",
    display_type: str = "none",
    title: str = "Top 10 Spectral Morphometrics",
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Pre-configured 10-row gallery for the canonical descriptors.

    Parameters
    ----------
    descriptor_data : dict of {name: data}
        Keys from: ``"hks"``, ``"wks"``, ``"si_hks"``, ``"bks"``,
        ``"gps"``, ``"shapedna"``, ``"bates_sp"``, ``"gaussian_k"``,
        ``"mean_k"``, ``"shape_idx"``, ``"casorati"``, ``"curvedness"``.
    atlas : str
    plot_kind : str

    Returns
    -------
    fig, axes
    """
    order = [
        "hks",
        "wks",
        "si_hks",
        "bks",
        "gps",
        "gaussian_k",
        "mean_k",
        "shape_idx",
        "casorati",
        "curvedness",
    ]
    specs = []
    for name in order:
        if name in descriptor_data:
            specs.append(
                BrainPlotSpec.from_descriptor(
                    name,
                    data=descriptor_data[name],
                    plot_kind=plot_kind,
                    atlas=atlas,
                )
            )

    return plot_morphometric_gallery(
        specs,
        views=views,
        style=style,
        display_type=display_type,
        title=title,
        save=save,
        formats=formats,
        row_height_in=1.3,
    )


# ======================================================================
# §7  MULTI-DESCRIPTOR COMPARISON PANEL
# ======================================================================


def plot_multi_descriptor_panel(
    rows: list[BrainPlotSpec],
    *,
    views: list[str] | None = None,
    style: str = "matte",
    display_type: str = "none",
    title: str = "",
    panel_width_in: float = 12.0,
    row_height_in: float = 1.5,
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Generic multi-row panel — the workhorse compositor.

    Parameters
    ----------
    rows : list of BrainPlotSpec
        4–8 rows (or more).
    """
    return plot_morphometric_gallery(
        rows,
        views=views,
        style=style,
        display_type=display_type,
        title=title,
        panel_width_in=panel_width_in,
        row_height_in=row_height_in,
        save=save,
        formats=formats,
    )


# ======================================================================
# §8  BILATERAL COMPARISON
# ======================================================================


def plot_bilateral_comparison(
    left_spec: BrainPlotSpec,
    right_spec: BrainPlotSpec,
    *,
    style: str = "matte",
    display_type: str = "none",
    title: str = "L vs R Comparison",
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Side-by-side L vs R hemisphere comparison (2 rows).

    Parameters
    ----------
    left_spec, right_spec : BrainPlotSpec
        Specs for left and right hemisphere data.
    """
    views_l = ["left_lateral", "left_medial", "superior", "inferior"]
    views_r = ["right_lateral", "right_medial", "superior", "inferior"]

    tmp = Path(tempfile.mkdtemp())
    images, labels = [], []
    px_w, px_h = int(12 * DPI), int(1.5 * DPI)

    for spec, views, i in [(left_spec, views_l, 0), (right_spec, views_r, 1)]:
        png = tmp / f"bilat_{i}.png"
        _render_row(
            spec, png, views=views, style=style, display_type=display_type, figsize_px=(px_w, px_h)
        )
        images.append(mpimg.imread(str(png)))
        labels.append(spec.label)

    fig, axes = _compose_panel(images, labels, title=title)

    if save:
        _save_figure(fig, save, formats=formats)
    return fig, axes


# ======================================================================
# §9  SPECTRAL PROGRESSION (HKS across t / WKS across e)
# ======================================================================


def plot_spectral_progression(
    scale_specs: list[BrainPlotSpec],
    *,
    descriptor_name: str = "HKS",
    views: list[str] | None = None,
    style: str = "matte",
    display_type: str = "none",
    title: str | None = None,
    save: PathLike | None = None,
    formats: str | list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Multi-scale spectral descriptor progression.

    One row per scale parameter (time for HKS, energy for WKS).
    Visually demonstrates how the descriptor captures geometry at
    different spatial frequencies.

    Parameters
    ----------
    scale_specs : list of BrainPlotSpec
        One per scale.  Label should include the scale value
        (e.g. ``"t=10"``, ``"e=2.5"``).
    descriptor_name : str
        For the title.
    """
    if title is None:
        title = f"{descriptor_name} — multi-scale progression"

    return plot_morphometric_gallery(
        scale_specs,
        views=views,
        style=style,
        display_type=display_type,
        title=title,
        row_height_in=1.3,
        save=save,
        formats=formats,
    )


# ======================================================================
# §10  TRACT VISUALISATION
# ======================================================================


def plot_brain_tracts(
    data: Any = None,
    *,
    atlas: str = "xtract_tiny",
    cmap: str = "inferno",
    vminmax: list[float] | None = None,
    nan_color: str = "#BDBDBD",
    orientation_coloring: bool = False,
    style: str = "matte",
    display_type: str = "none",
    views: list[str] | None = None,
    title: str = "",
    save: PathLike | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """White matter tract visualisation.

    Parameters
    ----------
    data : dict of {tract_name: value} or None
    atlas : str
    orientation_coloring : bool
        RGB directional encoding (ignores data).
    """
    if views is None:
        views = ["left_lateral", "anterior", "superior"]

    spec = BrainPlotSpec(
        data=data,
        cmap=cmap,
        vminmax=vminmax or [None, None],
        nan_color=nan_color,
        plot_kind="tracts",
        atlas=atlas,
        extra_kwargs={
            "orientation_coloring": orientation_coloring,
            **kwargs,
        },
    )

    tmp = Path(tempfile.mkdtemp())
    png = tmp / "tracts.png"
    _render_row(
        spec,
        png,
        views=views,
        style=style,
        display_type=display_type,
        figsize_px=(int(10 * DPI), int(2.5 * DPI)),
    )

    img = mpimg.imread(str(png))
    fig, axes = _compose_panel([img], [""], title=title, border=False)

    if save:
        _save_figure(fig, save)
    return fig, axes[0]


# ======================================================================

__all__ = [
    "DESCRIPTOR_STYLES",
    # Constants
    "DPI",
    "VIEWS_CORTEX",
    "VIEWS_FULL",
    "VIEWS_MEDIAL",
    # Spec
    "BrainPlotSpec",
    "plot_bilateral_comparison",
    # Single-row
    "plot_brain",
    "plot_brain_subcortical",
    "plot_brain_tracts",
    "plot_clustering_map",
    # Multi-row panels
    "plot_group_comparison",
    "plot_morphometric_gallery",
    "plot_multi_descriptor_panel",
    "plot_normative_map",
    "plot_spectral_progression",
    "plot_top10_morphometrics",
]
