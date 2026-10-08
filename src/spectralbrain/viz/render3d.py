"""3D rendering engine -- cortical / subcortical surfaces, tracts and any mesh.

One module for every 3D render SpectralBrain produces; hippocampus-specific
wrappers live in :mod:`spectralbrain.viz.hipp`, cluster renders in
:mod:`spectralbrain.viz.clusters` and spectral-field panels in
:mod:`spectralbrain.viz.spectral`.

Sections
--------
SA  Brain surface panels (yabplot/PyVista renders composited with matplotlib,
    styled by scienceplots): single metric, group comparison, normative
    deviation, atlas-free clustering map, descriptor gallery, multi-descriptor
    and bilateral panels, spectral progression, tracts, subcortical panel.
    Every public function accepts ``nan_color``, ``style``, ``display_type``
    and ``save=``, and returns ``(fig, axes)``.
SB  White-matter tractography (``neuro-tracts`` conventions): streamlines
    with DEC or scalar colouring (FURY), bundle surfaces with scalar/HKS/WKS
    overlays (PyVista), glass-brain context, multi-POV montages and hybrid
    raster+vector publication panels. Colour policy: DEC RGB for
    orientation, perceptually uniform sequential maps (Crameri ``batlow`` if
    ``cmcrameri`` is present, else ``viridis``) for FA/MD/HKS/density,
    diverging maps centred on zero for signed fields, robust 2-98% limits.
SC  Generic mesh renders (vedo, PyVista fallback): surface with scalar
    overlay, wireframe, curvature maps, multi-view, mesh comparison and
    scalar-difference maps; return ``(Path, metadata)``.
SD  Template-free six-view engine (vedo): any ``(coords, faces)`` in the six
    canonical anatomical views, composited with a shared colour bar
    (:func:`plot_surface_sixview`).

All heavy dependencies (yabplot, FURY/DIPY, PyVista, vedo, nibabel,
scikit-image, trimesh, cmcrameri) are imported lazily; the module imports
cleanly without them and raises a clear message only when a path needs one.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from spectralbrain.runtime import PathLike, get_logger
from spectralbrain.viz import _camera as _cam

# ======================================================================
# SA  BRAIN SURFACE PANELS
# ======================================================================


logger = get_logger(__name__)

# -- Constants ---------------------------------------------------------

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

# -- Default descriptor visual specs -----------------------------------

DESCRIPTOR_STYLES: dict[str, dict[str, Any]] = {
    "hks": {"cmap": "inferno", "vminmax": [None, None], "label": "HKS"},
    "wks": {"cmap": "cividis", "vminmax": [None, None], "label": "WKS"},
    "si_hks": {"cmap": "viridis", "vminmax": [None, None], "label": "SI-HKS"},
    "bks": {"cmap": "magma", "vminmax": [None, None], "label": "BKS"},
    "ibks": {"cmap": "magma", "vminmax": [None, None], "label": "IBKS"},
    "gps": {"cmap": "coolwarm", "vminmax": [None, None], "label": "GPS", "signed": True},
    "shapedna": {"cmap": "plasma", "vminmax": [None, None], "label": "ShapeDNA"},
    "bates_sp": {"cmap": "inferno", "vminmax": [None, None], "label": "Bates SP"},
    "gaussian_k": {
        "cmap": "RdBu_r",
        "vminmax": [None, None],
        "label": "Gaussian K",
        "signed": True,
    },
    "mean_k": {"cmap": "RdBu_r", "vminmax": [None, None], "label": "Mean H", "signed": True},
    "shape_idx": {"cmap": "RdBu_r", "vminmax": [-1, 1], "label": "Shape Index", "signed": True},
    "casorati": {"cmap": "magma", "vminmax": [None, None], "label": "Casorati"},
    "curvedness": {"cmap": "magma", "vminmax": [None, None], "label": "Curvedness"},
    "willmore": {"cmap": "inferno", "vminmax": [None, None], "label": r"Willmore $H^2$"},
    "z_score": {"cmap": "RdBu_r", "vminmax": [-3, 3], "label": "Z-score", "signed": True},
    "effect_d": {"cmap": "RdBu_r", "vminmax": [-1.5, 1.5], "label": "Cohen's d", "signed": True},
    "clusters": {
        "cmap": "tab10",
        "vminmax": [None, None],
        "label": "Clusters",
        "categorical": True,
    },
    "normative": {"cmap": "coolwarm", "vminmax": [-3, 3], "label": "Normative Z", "signed": True},
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
    signed : bool or None
        Signed map (z, d, curvature, ...): when ``vminmax`` has ``None``
        entries the range is made symmetric about 0 (``+/-max|v|``) so the
        diverging colormap is centred on zero.  ``None`` -> inferred from a
        diverging ``cmap``.
    categorical : bool
        Integer labels (clusters/parcels): rendered with one discrete colour
        per label (labels < 0 -> ``nan_color``).
    """

    label: str = ""
    data: Any = None
    cmap: str = "coolwarm"
    vminmax: list[float | None] = field(default_factory=lambda: [None, None])
    nan_color: Any = (1.0, 1.0, 1.0)
    plot_kind: str = "cortical"
    atlas: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    signed: bool | None = None
    categorical: bool = False

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
            extra_kwargs=dict(overrides.get("extra_kwargs", {})),
            signed=overrides.get("signed", style.get("signed")),
            categorical=overrides.get("categorical", style.get("categorical", False)),
        )


# ======================================================================
# S1  INTERNAL RENDERING ENGINE
# ======================================================================

_DIVERGING_CMAPS = {
    "rdbu",
    "rdbu_r",
    "coolwarm",
    "bwr",
    "seismic",
    "rdylbu",
    "rdylbu_r",
    "piyg",
    "prgn",
    "brbg",
    "puor",
    "rdgy",
    "spectral",
    "spectral_r",
    "sb_diverging",
    "vik",
}


def _is_diverging(cmap: Any) -> bool:
    name = cmap if isinstance(cmap, str) else getattr(cmap, "name", "")
    return str(name).lower() in _DIVERGING_CMAPS


def _vertexwise_name(mesh: Any, scalars: str | None) -> str:
    """Name of the per-vertex array to use on a (pyvista-like) mesh."""
    if scalars is not None:
        return scalars
    name = getattr(mesh, "active_scalars_name", None)
    if name:
        return name
    keys = list(getattr(mesh, "point_data", {}).keys())
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"Cannot tell which per-vertex array to use; pass scalars=<name> (available: {keys})."
    )


def _data_values(data: Any, scalars: str | None = None) -> np.ndarray:
    """All finite numeric values carried by a spec's ``data`` (best effort)."""
    if data is None:
        return np.array([])
    if isinstance(data, dict):
        vals = np.asarray(
            [v for v in data.values() if isinstance(v, (int, float, np.number))], dtype=float
        )
    elif isinstance(data, (tuple, list)) and len(data) == 2 and not np.isscalar(data[0]):
        parts = []
        for m in data:
            if isinstance(m, np.ndarray):
                parts.append(np.asarray(m, float).ravel())
            elif hasattr(m, "point_data"):
                parts.append(np.asarray(m.point_data[_vertexwise_name(m, scalars)], float).ravel())
        vals = np.concatenate(parts) if parts else np.array([])
    else:
        try:
            vals = np.asarray(data, dtype=float).ravel()
        except (TypeError, ValueError):
            return np.array([])
    return vals[np.isfinite(vals)]


def _map_values(data: Any, fn, scalars: str | None = None) -> Any:
    """Apply ``fn`` (array -> array) to a copy of the data (dict / (lh, rh) / array)."""
    if isinstance(data, dict):
        keys = list(data)
        arr = fn(np.asarray([np.nan if data[k] is None else data[k] for k in keys], float))
        return {k: float(v) for k, v in zip(keys, arr)}
    if isinstance(data, (tuple, list)) and len(data) == 2:
        out = []
        for m in data:
            if isinstance(m, np.ndarray):
                out.append(fn(np.asarray(m, float).copy()))
            else:
                m2 = m.copy()
                name = _vertexwise_name(m2, scalars)
                m2.point_data[name] = fn(np.asarray(m2.point_data[name], float).copy())
                try:
                    m2.set_active_scalars(name)
                except Exception:
                    pass
                out.append(m2)
        return tuple(out)
    return fn(np.asarray(data, float).copy())


def _resolve_spec(spec: BrainPlotSpec) -> BrainPlotSpec:
    """Return a copy of ``spec`` with signed/categorical colour handling applied."""
    scal = spec.extra_kwargs.get("scalars") if spec.plot_kind == "vertexwise" else None
    vm = list(spec.vminmax) if spec.vminmax is not None else [None, None]
    if spec.categorical and spec.data is not None:
        from matplotlib.colors import ListedColormap

        vals = _data_values(spec.data, scal)
        uniq = np.unique(vals[vals >= 0])
        rank = {float(u): i for i, u in enumerate(uniq)}

        def _to_rank(a):
            out = np.full(a.shape, np.nan)
            for i, x in enumerate(a.ravel()):
                if np.isfinite(x) and x >= 0:
                    out.flat[i] = rank[float(x)]
            return out

        base = plt.get_cmap(spec.cmap if isinstance(spec.cmap, str) else "tab10")
        n_base = getattr(base, "N", 256)
        k = max(len(uniq), 1)
        colors = [base(i % n_base) if n_base < 256 else base(i / max(k - 1, 1)) for i in range(k)]
        return replace(
            spec,
            data=_map_values(spec.data, _to_rank, scal),
            cmap=ListedColormap(colors, name="sb_discrete"),
            vminmax=[-0.5, k - 0.5],
        )
    signed = spec.signed if spec.signed is not None else _is_diverging(spec.cmap)
    if signed and (vm[0] is None or vm[1] is None):
        vals = _data_values(spec.data, scal)
        m = float(np.max(np.abs(vals))) if vals.size else 1.0
        m = m or 1.0
        vm = [-m if vm[0] is None else vm[0], m if vm[1] is None else vm[1]]
    return replace(spec, vminmax=vm)


def _shared_vminmax(specs: list[BrainPlotSpec]) -> list[BrainPlotSpec]:
    """Give comparison rows one common colour range (only where not set)."""
    pooled = [
        _data_values(sp.data, sp.extra_kwargs.get("scalars")) for sp in specs if sp.data is not None
    ]
    vals = np.concatenate(pooled) if pooled else np.array([])
    if vals.size == 0:
        return specs
    signed = any((sp.signed if sp.signed is not None else _is_diverging(sp.cmap)) for sp in specs)
    if signed:
        m = float(np.max(np.abs(vals))) or 1.0
        lo, hi = -m, m
    else:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    out = []
    for sp in specs:
        vm = list(sp.vminmax) if sp.vminmax is not None else [None, None]
        out.append(
            replace(sp, vminmax=[lo if vm[0] is None else vm[0], hi if vm[1] is None else vm[1]])
        )
    return out


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
        logger.debug("scienceplots not installed -- using SpectralBrain style.")
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
    spec = _resolve_spec(spec)

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
            kwargs.update(spec.extra_kwargs)  # previously silently dropped
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
    """Save figure -- always PNG + optional extras."""
    from spectralbrain.viz.graphics import savefig

    return savefig(fig, path, formats=formats, dpi=dpi)


# ======================================================================
# S2  PUBLIC API -- individual & single-row plots
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
    signed: bool | None = None,
    categorical: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Single-row brain surface plot.

    The basic building block -- renders one metric across 6-7 views.

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
    signed : bool or None
        Centre the colour range on 0 when ``vminmax`` is not given (``None`` ->
        inferred from a diverging ``cmap``).
    categorical : bool
        Integer labels: one discrete colour per label, labels < 0 -> nan_color.

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
        signed=signed,
        categorical=categorical,
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
# S3  GROUP COMPARISON
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
    shared_scale: bool = True,
) -> tuple[Figure, list[Axes]]:
    """Two- or three-row group comparison panel.

    Parameters
    ----------
    group_a : BrainPlotSpec
        Control group (typically blue-ish cmap).
    group_b : BrainPlotSpec
        Patient group.
    difference : BrainPlotSpec, optional
        A - B difference map (diverging cmap, symmetric vminmax).
    views : list of str
    style, display_type : str
    title : str
    save : PathLike, optional
    shared_scale : bool
        If True (default), ``group_a`` and ``group_b`` share one colour range
        (wherever their ``vminmax`` entries are ``None``), so the two rows are
        directly comparable.  False -> each row auto-scales on its own.

    Returns
    -------
    fig, axes
    """
    if views is None:
        views = VIEWS_CORTEX

    specs = [group_a, group_b]
    if shared_scale:
        specs = _shared_vminmax(specs)
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
# S4  NORMATIVE MAP
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
    scalars: str | None = None,
) -> tuple[Figure, list[Axes]]:
    """Normative z-score map with optional thresholded view.

    Parameters
    ----------
    z_data : dict or (lh, rh)
        Z-score values.
    threshold : float
        Threshold for the second row (if show_thresholded=True).
    show_thresholded : bool
        Show a second row with only extreme deviations (|z| <= threshold set
        to NaN, for parcellated dicts *and* vertex-wise data).
    scalars : str, optional
        Vertex-wise only: name of the per-vertex array holding the z-scores
        (default: the meshes' active scalars).

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
        extra_kwargs={"scalars": scalars} if scalars and plot_kind == "vertexwise" else {},
    )

    specs = [spec_full]

    if show_thresholded:
        # Threshold: set values within [-thr, thr] to NaN.
        def _thr(a: np.ndarray) -> np.ndarray:
            with np.errstate(invalid="ignore"):
                a[~(np.abs(a) > threshold)] = np.nan
            return a

        # Works on a copy for dicts, (lh, rh) meshes/arrays and plain arrays.
        thr_data = _map_values(z_data, _thr, scalars)

        spec_thr = BrainPlotSpec(
            label=f"|Z| > {threshold}",
            data=thr_data,
            cmap=cmap,
            vminmax=vm,
            nan_color=nan_color,
            plot_kind=plot_kind,
            atlas=atlas,
            extra_kwargs=dict(spec_full.extra_kwargs),
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
# S5  CLUSTERING MAP
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
        Cluster labels per region or per vertex.  Rendered with one discrete
        colour per cluster (``cmap`` cycled); labels < 0 (noise) use
        ``nan_color``.
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
        categorical=kwargs.pop("categorical", True),
        **kwargs,
    )


# ======================================================================
# S6  MORPHOMETRIC DESCRIPTOR GALLERY (4-10 rows)
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

    The flagship figure for spectral morphometry papers -- stack
    4-10 descriptors with consistent views for visual comparison.

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
    # Canonical order first, then any other provided descriptor (e.g.
    # "shapedna", "bates_sp") -- nothing passed in is silently dropped.
    names = [n for n in order if n in descriptor_data]
    names += [n for n in descriptor_data if n not in order]
    unknown = [n for n in names if n not in DESCRIPTOR_STYLES]
    if unknown:
        warnings.warn(
            f"No DESCRIPTOR_STYLES entry for {unknown}; rendered with default styling.",
            stacklevel=2,
        )
    specs = [
        BrainPlotSpec.from_descriptor(
            name,
            data=descriptor_data[name],
            plot_kind=plot_kind,
            atlas=atlas,
        )
        for name in names
    ]
    if not specs:
        raise ValueError("descriptor_data is empty.")

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
# S7  MULTI-DESCRIPTOR COMPARISON PANEL
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
    """Generic multi-row panel -- the workhorse compositor.

    Parameters
    ----------
    rows : list of BrainPlotSpec
        4-8 rows (or more).
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
# S8  BILATERAL COMPARISON
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
    shared_scale: bool = True,
) -> tuple[Figure, list[Axes]]:
    """Side-by-side L vs R hemisphere comparison (2 rows).

    Parameters
    ----------
    left_spec, right_spec : BrainPlotSpec
        Specs for left and right hemisphere data.
    shared_scale : bool
        If True (default) both rows share one colour range (where their
        ``vminmax`` entries are ``None``) so asymmetries are visible.
    """
    if shared_scale:
        left_spec, right_spec = _shared_vminmax([left_spec, right_spec])
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
# S9  SPECTRAL PROGRESSION (HKS across t / WKS across e)
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
        title = f"{descriptor_name} -- multi-scale progression"

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
# S10  TRACT VISUALISATION
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
# SB  WHITE-MATTER TRACTOGRAPHY
# ======================================================================


# Canonical camera viewpoints (azimuth, elevation) in the RAS+ world frame.
#: Camera presets in the shared RAS convention of :mod:`spectralbrain.viz._camera`
#: (azimuth from +x towards +y, elevation towards +z): ``left`` camera at -x,
#: ``anterior`` at +y, ``superior`` at +z.  Every backend (FURY, PyVista)
#: builds an explicit position/focal-point/view-up camera from these angles.
TRACT_VIEWS: dict[str, dict[str, float]] = {
    "left": {"azimuth": 180.0, "elevation": 0.0},
    "right": {"azimuth": 0.0, "elevation": 0.0},
    "anterior": {"azimuth": 90.0, "elevation": 0.0},
    "posterior": {"azimuth": -90.0, "elevation": 0.0},
    "superior": {"azimuth": 0.0, "elevation": 90.0},
    "inferior": {"azimuth": 0.0, "elevation": -90.0},
    "oblique": {"azimuth": 135.0, "elevation": 25.0},
}

_TRACT_SIZE = (1600, 1600)
_TRACT_BG = "white"


# ----------------------------------------------------------------------
# Lazy dependency requires
# ----------------------------------------------------------------------
def _ensure_offscreen() -> None:
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


def _require_fury():
    _ensure_offscreen()
    try:
        from fury import actor, window

        return actor, window
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FURY is required for streamline rendering. Install with: pip install fury dipy"
        ) from exc


def _require_dipy_io():
    try:
        from dipy.io.streamline import load_tractogram
        from dipy.tracking.streamline import (
            select_random_set_of_streamlines,
            transform_streamlines,
        )

        return load_tractogram, select_random_set_of_streamlines, transform_streamlines
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DIPY is required to load streamlines. Install with: pip install dipy"
        ) from exc


def _require_pyvista():
    _ensure_offscreen()
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        return pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyVista is required for bundle-surface rendering. Install with: pip install pyvista"
        ) from exc


def _require_skimage_mc():
    try:
        from skimage.measure import marching_cubes

        return marching_cubes
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-image is required for mask->mesh. Install with: pip install scikit-image"
        ) from exc


# ----------------------------------------------------------------------
# Colour helpers (harmonised with spectralbrain.viz.graphics policy)
# ----------------------------------------------------------------------
def get_cmap(kind: str = "sequential"):
    """Perceptually-uniform colormap by role (``sequential``/``diverging``)."""
    try:
        from cmcrameri import cm as cmc

        if kind == "sequential":
            return cmc.batlow
        if kind == "diverging":
            return cmc.vik
    except Exception:
        pass
    import matplotlib.pyplot as plt

    return plt.get_cmap("viridis" if kind == "sequential" else "RdBu_r")


def robust_clim(
    values: np.ndarray, low: float = 2.0, high: float = 98.0, symmetric: bool = False
) -> tuple[float, float]:
    """Robust colour limits from percentiles (optionally symmetric on zero)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(v, [low, high])
    if symmetric:
        m = max(abs(lo), abs(hi))
        return (-m, m)
    return (float(lo), float(hi))


def resolve_scalar_clim(
    values: np.ndarray, clim: tuple | None = None, robust: bool = True, symmetric: bool = False
) -> tuple[float, float]:
    """Colour limits used by the scalar renderers (shared with the colorbar)."""
    if clim is not None:
        return float(clim[0]), float(clim[1])
    if robust:
        return robust_clim(values, symmetric=symmetric)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    if symmetric:
        m = float(np.max(np.abs(v)))
        return (-m, m)
    return (float(v.min()), float(v.max()))


def _scalar_rgb(
    values, cmap, lo: float, hi: float, nan_rgb: tuple = (0.74, 0.74, 0.74)
) -> np.ndarray:
    """Linear ``Normalize(lo, hi)`` colouring; NaN points get ``nan_rgb``."""
    v = np.asarray(values, float)
    span = (hi - lo) or 1.0
    rgb = np.asarray(cmap(np.clip((v - lo) / span, 0, 1)))[..., :3]
    rgb = np.array(rgb, dtype=float, copy=True)
    rgb[~np.isfinite(v)] = nan_rgb
    return rgb


def _dec_colors(streamlines) -> list[np.ndarray]:
    """Per-point direction-encoded RGB (x=L-R red, y=A-P green, z=I-S blue)."""
    colors = []
    for sl in streamlines:
        sl = np.asarray(sl, float)
        if len(sl) < 2:
            colors.append(np.tile([0.5, 0.5, 0.5], (len(sl), 1)))
            continue
        d = np.gradient(sl, axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n == 0] = 1.0
        colors.append(np.abs(d / n))
    return colors


# ----------------------------------------------------------------------
# 1. Streamlines
# ----------------------------------------------------------------------
def load_streamlines(path: PathLike, reference: PathLike | None = None, to_space: str = "world"):
    """Load a ``.trk``/``.tck`` tractogram into world (RAS+ mm) coordinates.

    ``.tck`` (MRtrix) stores no affine and needs ``reference`` (a NIfTI).
    Returns ``(streamlines, affine)``.
    """
    load_tractogram, _, _ = _require_dipy_io()
    from dipy.io.stateful_tractogram import Space

    ref = reference if reference is not None else "same"
    sft = load_tractogram(str(path), ref, to_space=Space.RASMM)
    return list(sft.streamlines), sft.affine


def render_streamlines(
    streamlines,
    *,
    color_by: str = "orientation",
    scalars: Sequence[np.ndarray] | None = None,
    tube: bool = True,
    linewidth: float = 0.4,
    cmap: Any = None,
    clim: tuple | None = None,
    robust: bool = True,
    view: str = "oblique",
    n_max: int = 20000,
    bg: str = _TRACT_BG,
    size: tuple[int, int] = _TRACT_SIZE,
    out_path: PathLike | None = None,
    random_state: int = 0,
) -> Path:
    """Render a streamline bundle offscreen (FURY) and snapshot to PNG.

    Parameters
    ----------
    streamlines : sequence of (N_i, 3) arrays
    color_by : {"orientation", "scalar"}
        ``"orientation"`` = DEC RGB (default); ``"scalar"`` colours each point by
        ``scalars`` with a perceptual colormap.
    scalars : sequence of (N_i,) arrays, optional
        Per-point scalar values (required for ``color_by="scalar"``).
    tube : bool
        Render as streamtubes (hero look) vs thin lines.
    view : str
        Camera preset (see :data:`TRACT_VIEWS`).
    n_max : int
        Downsample whole-brain tractograms above this many streamlines.

    Returns
    -------
    Path to the PNG.
    """
    actor, window = _require_fury()
    rng = np.random.default_rng(random_state)
    sl = list(streamlines)
    if len(sl) > n_max:
        idx = rng.choice(len(sl), size=n_max, replace=False)
        sl = [sl[i] for i in idx]
        if scalars is not None:
            scalars = [scalars[i] for i in idx]

    if color_by == "orientation":
        colors = _dec_colors(sl)
    elif color_by == "scalar":
        if scalars is None:
            raise ValueError("color_by='scalar' requires `scalars`.")
        allv = np.concatenate([np.asarray(s, float).ravel() for s in scalars])
        lo, hi = resolve_scalar_clim(allv, clim=clim, robust=robust)
        cm = cmap if cmap is not None else get_cmap("sequential")
        if isinstance(cm, str):
            import matplotlib.pyplot as plt

            cm = plt.get_cmap(cm)
        colors = [_scalar_rgb(s, cm, lo, hi) for s in scalars]
    else:
        raise ValueError("color_by must be 'orientation' or 'scalar'.")

    scene = window.Scene()
    scene.background({"white": (1, 1, 1), "black": (0, 0, 0)}.get(bg, (1, 1, 1)))
    if tube:
        stream_actor = actor.streamtube(sl, colors=colors, linewidth=linewidth)
    else:
        stream_actor = actor.line(sl, colors=colors, linewidth=max(linewidth, 1.0))
    scene.add(stream_actor)
    _set_fury_camera(scene, view, np.vstack([np.asarray(x, float) for x in sl]))

    # depth peeling + anti-aliasing for correct transparency / clean edges
    out = Path(out_path) if out_path else Path(tempfile.mkstemp(suffix=".png")[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        window.record(
            scene=scene, out_path=str(out), size=size, multi_samples=8, reset_camera=False
        )
    except TypeError:  # older FURY positional/keyword drift
        window.record(scene, out_path=str(out), size=size)
    logger.info("Saved streamline render -> %s", out)
    return out


def _set_fury_camera(scene, view: str, points: np.ndarray | None = None) -> None:
    """Place the FURY camera explicitly (RAS convention, see TRACT_VIEWS).

    Relative ``azimuth``/``elevation`` rotations act on VTK's default camera
    (which looks down from +z), so they produced mislabelled views; an
    explicit position/focal point/view-up is used instead.
    """
    if points is None or len(points) == 0:
        scene.reset_camera()
        points = np.array(scene.GetActiveCamera().GetFocalPoint(), float)[None, :]
    cam = _cam.camera_for_view(view, points, angles=TRACT_VIEWS)
    scene.set_camera(
        position=cam["position"], focal_point=cam["focal_point"], view_up=cam["viewup"]
    )
    scene.reset_clipping_range()


# ----------------------------------------------------------------------
# 2. Bundle surfaces with scalar / spectral overlays
# ----------------------------------------------------------------------
def mask_to_mesh(
    mask: np.ndarray,
    affine: np.ndarray | None = None,
    level: float = 0.5,
    smooth_sigma: float = 1.0,
    taubin_iter: int = 25,
    raw: bool = False,
):
    """Marching-cubes surface from a binary tract mask, mapped to world coords.

    With ``raw=False`` (default) the surface is produced by the shared,
    open-surface-safe improvement pipeline
    (:func:`spectralbrain.io.meshing.volume_to_mesh` with ``closed=False``):
    Gaussian field smoothing + Lewiner marching cubes + Taubin (shrink-free)
    smoothing, giving a clean surface suitable for spectral overlays. With
    ``raw=True`` a plain marching-cubes surface (no smoothing) is returned.

    Bundle masks are open / branching, so ``closed=False`` is used: the surface
    is *not* forced watertight and components are *not* pruned. Returns
    ``(vertices, faces)``.
    """
    aff = np.eye(4) if affine is None else np.asarray(affine, float)
    if raw:
        marching_cubes = _require_skimage_mc()
        verts, faces, _n, _v = marching_cubes(
            np.asarray(mask, float), level=level, method="lewiner", allow_degenerate=False
        )
        homog = np.c_[verts, np.ones(len(verts))]
        verts = (aff @ homog.T).T[:, :3]
        return np.asarray(verts, np.float64), np.asarray(faces, np.int64)

    from spectralbrain.io.meshing import volume_to_mesh

    verts, faces = volume_to_mesh(
        np.asarray(mask),
        aff,
        raw=False,
        closed=False,
        level=level,
        field_mode="gaussian",
        sigma_vox=float(smooth_sigma),
        taubin_iterations=int(taubin_iter),
    )
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def render_bundle_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scalars: np.ndarray | None = None,
    cmap: Any = None,
    clim: tuple | None = None,
    symmetric: bool = False,
    robust: bool = True,
    view: str = "oblique",
    bg: str = _TRACT_BG,
    size: tuple[int, int] = _TRACT_SIZE,
    metallic: float = 0.1,
    roughness: float = 0.6,
    nan_color: str = "#BDBDBD",
    out_path: PathLike | None = None,
) -> Path:
    """Render a bundle surface mesh with an optional per-vertex scalar overlay.

    PyVista PBR render with depth peeling + SSAA. For signed maps (t, d, r) pass
    ``symmetric=True`` to centre a diverging colormap on zero. Returns the PNG.
    """
    pv = _require_pyvista()
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    mesh = pv.PolyData(V, np.column_stack([np.full(len(F), 3), F]).ravel())

    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    plotter.set_background(bg)
    try:
        plotter.enable_depth_peeling(10)
    except Exception:
        pass
    plotter.enable_anti_aliasing("ssaa") if hasattr(plotter, "enable_anti_aliasing") else None

    if scalars is not None:
        s = np.asarray(scalars, float)
        if clim is None:
            clim = resolve_scalar_clim(s, robust=robust, symmetric=symmetric)
        cm = cmap if cmap is not None else get_cmap("diverging" if symmetric else "sequential")
        mesh["scalars"] = s
        plotter.add_mesh(
            mesh,
            scalars="scalars",
            cmap=cm,
            clim=clim,
            nan_color=nan_color,
            smooth_shading=True,
            pbr=True,
            metallic=metallic,
            roughness=roughness,
            show_scalar_bar=True,
        )
    else:
        plotter.add_mesh(
            mesh,
            color="#cccccc",
            smooth_shading=True,
            pbr=True,
            metallic=metallic,
            roughness=roughness,
        )

    _set_pv_camera(plotter, V, view)
    out = Path(out_path) if out_path else Path(tempfile.mkstemp(suffix=".png")[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    logger.info("Saved bundle-surface render -> %s", out)
    return out


def spectral_overlay(
    vertices: np.ndarray,
    faces: np.ndarray,
    kind: str = "hks",
    n_eigen: int = 100,
    t_index: int = 30,
    laplacian_method: str = "cotangent",
) -> np.ndarray:
    """Compute an HKS/WKS per-vertex field on a bundle surface for overlay.

    Uses SpectralBrain's own LBO + descriptor machinery so the bundle surface is
    described with the same spectral vocabulary as cortical/hippocampal surfaces.
    The marching-cubes bundle surface is a clean 2-manifold, so the default
    ``cotangent`` Laplacian is appropriate; pass ``laplacian_method="robust"`` for
    non-manifold inputs (requires ``robust_laplacian``). Returns a (V,) field.
    """
    from spectralbrain.core.meshes import BrainMesh

    mesh = BrainMesh(np.asarray(vertices, float), np.asarray(faces, np.int64))
    decomp = mesh.decompose(k=n_eigen, laplacian_method=laplacian_method)
    if kind == "hks":
        from spectralbrain.spectral.lbo.descriptors import compute_hks

        desc = np.asarray(compute_hks(decomp))
    elif kind == "wks":
        from spectralbrain.spectral.lbo.descriptors import compute_wks

        desc = np.asarray(compute_wks(decomp))
    else:
        raise ValueError("kind must be 'hks' or 'wks'.")
    col = int(np.clip(t_index, 0, desc.shape[1] - 1))
    return desc[:, col]


def _set_pv_camera(plotter, vertices: np.ndarray, view: str) -> None:
    """Explicit PyVista camera in the shared RAS convention.

    Accepts the TRACT_VIEWS names and the cluster/mesh preset names
    (``left_lateral``, ``right_medial``, ...); unknown names raise ValueError.
    """
    c = vertices.mean(axis=0)
    radius = float(np.linalg.norm(vertices - c, axis=1).max()) * 2.6
    d = _cam.view_direction(view, TRACT_VIEWS)
    pos = c + radius * d
    plotter.camera_position = [tuple(pos), tuple(c), _cam.view_up(d)]


# ----------------------------------------------------------------------
# 3. Multi-POV montage & publication panels
# ----------------------------------------------------------------------
def streamlines_multiview(
    streamlines,
    *,
    views: Sequence[str] = ("left", "anterior", "superior", "oblique"),
    color_by: str = "orientation",
    scalars: Sequence[np.ndarray] | None = None,
    titles: Sequence[str] | None = None,
    colorbar: dict | None = None,
    tube: bool = True,
    bg: str = _TRACT_BG,
    size: tuple[int, int] = (1200, 1200),
    out_path: PathLike | None = None,
    random_state: int = 0,
    cmap: Any = None,
    clim: tuple | None = None,
    robust: bool = True,
):
    """Render one bundle from several canonical POVs and composite into a panel.

    This is the multi-POV tract figure: each view is rendered offscreen, then all
    are assembled into a single hybrid raster+vector matplotlib figure with an
    optional shared colorbar. Returns ``(fig, png_paths)``.

    For ``color_by="scalar"`` the colour limits and colormap are resolved
    **once** (``clim`` / ``colorbar["clim"]`` if given, else robust 2-98%) and
    used for every view *and* for the shared colorbar, so the colorbar always
    matches the rendered colours.
    """
    for v in views:
        _cam.view_direction(v, TRACT_VIEWS)  # fail fast on unknown view names
    if color_by == "scalar" and scalars is not None:
        cb_in = dict(colorbar) if colorbar is not None else None
        if clim is None and cb_in is not None and "clim" in cb_in:
            clim = cb_in["clim"]
        allv = np.concatenate([np.asarray(s, float).ravel() for s in scalars])
        clim = resolve_scalar_clim(allv, clim=clim, robust=robust)
        if cmap is None:
            kind = cb_in.get("kind", "sequential") if cb_in is not None else "sequential"
            cmap = get_cmap(kind)
        if cb_in is not None:
            cb_in["clim"] = clim
            cb_in["cmap"] = cmap
            colorbar = cb_in
    tmp = Path(tempfile.mkdtemp())
    pngs = []
    for v in views:
        p = tmp / f"tract_{v}.png"
        render_streamlines(
            streamlines,
            color_by=color_by,
            scalars=scalars,
            tube=tube,
            view=v,
            bg=bg,
            size=size,
            out_path=p,
            random_state=random_state,
            cmap=cmap,
            clim=clim,
            robust=robust,
        )
        pngs.append(p)
    titles = list(titles) if titles is not None else [v.capitalize() for v in views]
    fig = compose_tract_panel(pngs, titles=titles, colorbar=colorbar, out_path=out_path)
    return fig, pngs


def compose_tract_panel(
    images: Sequence[PathLike],
    *,
    titles: Sequence[str] | None = None,
    colorbar: dict | None = None,
    ncols: int | None = None,
    panel_letters: bool = True,
    out_path: PathLike | None = None,
    dpi: int = 300,
):
    """Composite rendered PNGs into a publication panel (hybrid raster+vector).

    Parameters
    ----------
    images : sequence of PNG paths
    titles : sequence of str, optional
    colorbar : dict, optional
        ``{"kind": "sequential"|"diverging", "clim": (lo, hi), "label": str,
        "cmap": Colormap}`` to draw a shared vector colorbar.  The colorbar uses
        a linear ``Normalize(lo, hi)`` -- the same mapping the renderers use --
        and ``cmap`` (if given) overrides the ``kind`` default.
    ncols : int, optional
        Columns (defaults to len(images) up to 4).
    """
    import matplotlib.cm as mcm
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    imgs = [mpimg.imread(str(p)) for p in images]
    n = len(imgs)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            ax.imshow(imgs[i])
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=11)
            if panel_letters:
                ax.text(
                    0.02,
                    0.98,
                    letters[i],
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    va="top",
                    ha="left",
                )
        ax.axis("off")

    if colorbar is not None:
        kind = colorbar.get("kind", "sequential")
        lo, hi = colorbar.get("clim", (0.0, 1.0))
        cm = colorbar.get("cmap") or get_cmap(kind)
        if isinstance(cm, str):
            cm = plt.get_cmap(cm)
        # Renders map scalars linearly onto [lo, hi]; a TwoSlopeNorm here would
        # disagree with the rendered colours whenever |lo| != |hi|.
        norm = Normalize(lo, hi)
        sm = mcm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(colorbar.get("label", ""))
    else:
        # tight_layout is incompatible with manually-added colorbar axes; only
        # apply it when there is no shared colorbar (savefig crops either way).
        fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        png = out_path.with_suffix(".png")
        if out_path.suffix.lower() != ".png":
            fig.savefig(str(png), dpi=dpi, bbox_inches="tight")
        logger.info("Saved tract panel -> %s", out_path)
    return fig


def add_glass_brain(
    plotter,
    brain_mask: np.ndarray,
    affine: np.ndarray,
    *,
    opacity: float = 0.12,
    color: str = "#cccccc",
    level: float = 0.5,
):
    """Add a translucent glass-brain shell to a PyVista plotter for context."""
    verts, faces = mask_to_mesh(
        brain_mask, affine=affine, level=level, smooth_sigma=1.5, taubin_iter=15
    )
    pv = _require_pyvista()
    shell = pv.PolyData(verts, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    plotter.add_mesh(shell, color=color, opacity=opacity, smooth_shading=True)
    return plotter


# ======================================================================
# SC  GENERIC MESH RENDERS (vedo)
# ======================================================================


if TYPE_CHECKING:
    import vedo

# ---------------------------------------------------------------------------
#  Constants -- shared with points.py
# ---------------------------------------------------------------------------

_MESH_SIZE: tuple[int, int] = (1600, 1200)
_DEFAULT_SCALE: int = 2
_MESH_BG: str = "white"

# Curvature method codes used by VTK / vedo
CURVATURE_METHODS: dict[str, int] = {
    "gaussian": 0,
    "mean": 1,
    "maximum": 2,
    "minimum": 3,
}

# Standard multi-view camera presets (azimuth, elevation) in the RAS
# convention of :mod:`spectralbrain.viz._camera`: azimuth from +x towards +y,
# elevation towards +z -- left lateral camera at -x, anterior at +y, superior at
# +z.  Renders use explicit camera dicts built from these angles.
CAMERA_PRESETS: dict[str, dict[str, Any]] = _cam.presets(
    [
        "anterior",
        "posterior",
        "left_lateral",
        "right_lateral",
        "superior",
        "inferior",
        "left_medial",
        "right_medial",
        "oblique_left",
        "oblique_right",
    ]
)


# ======================================================================
# S0  Lazy imports & helpers
# ======================================================================


def _ensure_vedo_offscreen() -> None:
    """Set vedo to offscreen rendering mode."""
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")


def _get_vedo():
    """Lazy-import vedo, raising ImportError if unavailable."""
    _ensure_vedo_offscreen()
    try:
        import vedo

        try:
            vedo.start_xvfb()
        except Exception:
            pass
        return vedo
    except ImportError:
        raise ImportError(
            "vedo is required for mesh visualization.  Install with: pip install vedo"
        )


def _save_screenshot(plotter, save: PathLike | None, *, scale: int = _DEFAULT_SCALE) -> Path:
    """Capture a vedo Plotter to PNG and close it."""
    if save is None:
        fd, save = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(save), scale=scale)
    plotter.close()
    logger.info("Saved mesh render -> %s", save)
    return save


def _build_vedo_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vedo_module,
) -> vedo.Mesh:
    """Construct a vedo Mesh from numpy arrays.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array of int indices
    vedo_module : the vedo module (passed to avoid re-import)

    Returns
    -------
    vedo.Mesh
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=int)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    mesh = vedo_module.Mesh([vertices, faces])
    return mesh


def _resolve_cmap(scalar_name: str | None, cmap: str | None) -> str:
    """Pick colourmap: explicit > name-based > viridis."""
    if cmap is not None:
        return cmap
    LOOKUP = {
        "hks": "inferno",
        "wks": "cividis",
        "bks": "magma",
        "gps": "viridis",
        "shapedna": "plasma",
        "curvature": "RdBu_r",
        "mean": "RdBu_r",
        "gaussian": "RdBu_r",
        "thickness": "YlOrRd",
        "z_score": "RdBu_r",
        "difference": "RdBu_r",
    }
    if scalar_name is not None:
        full = scalar_name.lower().replace(" ", "_").replace("-", "_")
        if full in LOOKUP:  # exact keys first (e.g. "z_score")
            return LOOKUP[full]
        if full.startswith(("z_", "zscore", "t_stat", "tstat", "cohen", "effect", "diff")):
            return "RdBu_r"
        return LOOKUP.get(full.split("_")[0], "viridis")
    return "viridis"


# ======================================================================
# S1  Surface render -- smooth-shaded mesh with scalar overlay
# ======================================================================


def plot_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray | None = None,
    scalar_name: str = "HKS",
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    color: str = "gold",
    alpha: float = 1.0,
    show_edges: bool = False,
    edge_color: str = "gray",
    edge_width: float = 0.3,
    show_scalarbar: bool = True,
    lighting: str = "default",
    camera: dict[str, Any] | None = None,
    title: str | None = None,
    bg: str = _MESH_BG,
    size: tuple[int, int] = _MESH_SIZE,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Render a triangular mesh with optional scalar overlay.

    This is the primary mesh visualisation: a smooth Phong-shaded
    surface optionally coloured by a per-vertex spectral descriptor,
    morphometric measure, or statistical map.

    Parameters
    ----------
    vertices : (V, 3) array
        Mesh vertex coordinates.
    faces : (F, 3) array
        Triangle index array.
    scalars : (V,) array or None
        Per-vertex scalar values.  None -> uniform ``color``.
    scalar_name : str
        Label for colourbar and automatic cmap selection.
    cmap : str or None
        Colourmap.  None -> auto from scalar_name.
    vmin, vmax : float or None
        Colour range.  None -> 1st / 99th percentiles.
    color : str
        Uniform mesh colour when scalars is None.
    alpha : float
        Mesh opacity (0-1).
    show_edges : bool
        Overlay wireframe edges.
    edge_color, edge_width : str, float
        Edge appearance.
    show_scalarbar : bool
        Display colourbar.
    lighting : str
        VTK lighting style -- ``'default'``, ``'metallic'``,
        ``'plastic'``, ``'shiny'``, ``'glossy'``.
    camera : dict or None
        Camera configuration (``pos``, ``focal_point``, ``viewup``).
    title : str or None
        Figure title.
    bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'n_vertices'``, ``'n_faces'``,
        ``'scalar_range'``, ``'cmap'``.
    """
    vedo = _get_vedo()
    mesh = _build_vedo_mesh(vertices, faces, vedo)
    cmap_name = _resolve_cmap(scalar_name, cmap)

    meta: dict[str, Any] = {
        "n_vertices": vertices.shape[0],
        "n_faces": faces.shape[0],
        "cmap": cmap_name,
        "scalar_range": None,
    }

    if scalars is not None:
        scalars = np.asarray(scalars, dtype=np.float64)
        assert scalars.shape[0] == vertices.shape[0], (
            f"scalars ({scalars.shape[0]}) must match vertices ({vertices.shape[0]})"
        )
        if vmin is None:
            vmin = float(np.nanpercentile(scalars, 1))
        if vmax is None:
            vmax = float(np.nanpercentile(scalars, 99))

        mesh.pointdata[scalar_name] = scalars
        mesh.cmap(cmap_name, scalar_name, vmin=vmin, vmax=vmax)
        if show_scalarbar:
            mesh.add_scalarbar(title=scalar_name)
        meta["scalar_range"] = (vmin, vmax)
    else:
        mesh.color(color)

    mesh.alpha(alpha)
    mesh.lighting(lighting)

    if show_edges:
        mesh.linewidth(edge_width).linecolor(edge_color)

    plt = vedo.Plotter(offscreen=True, size=size, bg=bg, title=title or "")

    show_kw: dict[str, Any] = {"viewup": "z", "zoom": 1.2}
    if camera is not None:
        show_kw["camera"] = camera
    plt.show(mesh, **show_kw)

    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# S2  Wireframe render
# ======================================================================


def plot_wireframe(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    color: str = "steelblue",
    linewidth: float = 0.5,
    alpha: float = 1.0,
    camera: dict[str, Any] | None = None,
    title: str | None = None,
    bg: str = _MESH_BG,
    size: tuple[int, int] = _MESH_SIZE,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Wireframe render of a mesh for topology inspection.

    Useful for QC of reconstructed surfaces and for methods figures
    that need to show mesh structure clearly.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    color : str
        Wire colour.
    linewidth : float
        Wire thickness.
    alpha : float
        Opacity.
    camera, title, bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata.
    """
    vedo = _get_vedo()
    mesh = _build_vedo_mesh(vertices, faces, vedo)
    mesh.wireframe(True).color(color).linewidth(linewidth).alpha(alpha)

    plt = vedo.Plotter(offscreen=True, size=size, bg=bg, title=title or "")
    show_kw: dict[str, Any] = {"viewup": "z", "zoom": 1.2}
    if camera is not None:
        show_kw["camera"] = camera
    plt.show(mesh, **show_kw)

    meta = {"n_vertices": vertices.shape[0], "n_faces": faces.shape[0]}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# S3  Curvature map
# ======================================================================


def plot_curvature(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str = "mean",
    *,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = True,
    title: str | None = None,
    bg: str = _MESH_BG,
    size: tuple[int, int] = _MESH_SIZE,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Compute and render curvature on a mesh surface.

    Computes curvature using VTK's built-in estimator and immediately
    displays it with a diverging colourmap centred on zero.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    method : {'gaussian', 'mean', 'maximum', 'minimum'}
        Curvature type.
    cmap : str
        Colourmap (diverging recommended for curvature).
    vmin, vmax : float or None
        Colour range.  If *symmetric* is True and these are None,
        range is set to +/- 95th percentile.
    symmetric : bool
        Centre the colourmap on zero.
    title, bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'curvature_method'``,
        ``'curvature_stats'`` (mean, std, min, max).
    """
    vedo = _get_vedo()
    mesh = _build_vedo_mesh(vertices, faces, vedo)

    method_code = CURVATURE_METHODS.get(method.lower())
    if method_code is None:
        raise ValueError(
            f"Unknown curvature method '{method}'.  Choose from: {list(CURVATURE_METHODS.keys())}"
        )

    mesh.compute_curvature(method=method_code)

    # VTK names the array generically; retrieve it
    curv = mesh.pointdata["Curvature"]
    curv_clean = curv[np.isfinite(curv)]

    # Auto colour range
    if vmin is None or vmax is None:
        p95 = float(np.percentile(np.abs(curv_clean), 95))
        if symmetric:
            vmin = vmin if vmin is not None else -p95
            vmax = vmax if vmax is not None else p95
        else:
            vmin = vmin if vmin is not None else float(np.percentile(curv_clean, 1))
            vmax = vmax if vmax is not None else float(np.percentile(curv_clean, 99))

    label = f"{method.capitalize()} curvature"
    mesh.cmap(cmap, "Curvature", vmin=vmin, vmax=vmax)
    mesh.add_scalarbar(title=label)

    plt = vedo.Plotter(offscreen=True, size=size, bg=bg, title=title or label)
    plt.show(mesh, viewup="z", zoom=1.2)

    meta = {
        "curvature_method": method,
        "curvature_stats": {
            "mean": float(np.mean(curv_clean)),
            "std": float(np.std(curv_clean)),
            "min": float(np.min(curv_clean)),
            "max": float(np.max(curv_clean)),
        },
        "vmin": vmin,
        "vmax": vmax,
    }
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# S4  Multi-view panel -- same mesh from multiple camera angles
# ======================================================================


def plot_multi_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray | None = None,
    scalar_name: str = "HKS",
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    views: list[str] | None = None,
    *,
    color: str = "gold",
    lighting: str = "default",
    bg: str = _MESH_BG,
    size: tuple[int, int] | None = None,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Multi-view panel showing the same mesh from different angles.

    Renders the same mesh (optionally with scalar overlay) in a 1xN
    panel strip.  Standard views: anterior, posterior, lateral,
    medial, superior, inferior.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    scalars : (V,) array or None
    scalar_name : str
    cmap : str or None
    vmin, vmax : float or None
    views : list of str or None
        Camera preset names from ``CAMERA_PRESETS``.  None defaults
        to ``['left_lateral', 'anterior', 'superior', 'right_lateral']``.
    color : str
        Uniform colour when scalars is None.
    lighting : str
        VTK lighting preset.
    bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata.
    """
    vedo = _get_vedo()

    if views is None:
        views = ["left_lateral", "anterior", "superior", "right_lateral"]
    views = _cam.validate_views(views, CAMERA_PRESETS)
    n_views = len(views)

    if size is None:
        size = (600 * n_views, 600)

    cmap_name = _resolve_cmap(scalar_name, cmap)

    # Build the base mesh once, then clone per view
    base = _build_vedo_mesh(vertices, faces, vedo)
    if scalars is not None:
        scalars = np.asarray(scalars, dtype=np.float64)
        if vmin is None:
            vmin = float(np.nanpercentile(scalars, 1))
        if vmax is None:
            vmax = float(np.nanpercentile(scalars, 99))
        base.pointdata[scalar_name] = scalars
        base.cmap(cmap_name, scalar_name, vmin=vmin, vmax=vmax)
        base.add_scalarbar(title=scalar_name)
    else:
        base.color(color)
    base.lighting(lighting)

    plt = vedo.Plotter(
        shape=(1, n_views),
        sharecam=False,
        offscreen=True,
        size=size,
        bg=bg,
    )

    for i, view_name in enumerate(views):
        m = base.clone()
        cam = _cam.camera_for_view(view_name, vertices, angles=CAMERA_PRESETS)

        plt.at(i).show(
            m,
            title=view_name.replace("_", " ").title(),
            camera=cam,
            zoom=1.1,
        )

    meta = {
        "n_vertices": vertices.shape[0],
        "n_faces": faces.shape[0],
        "views": views,
        "scalar_range": (vmin, vmax) if scalars is not None else None,
    }
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# S5  Mesh comparison -- side-by-side panels
# ======================================================================


def plot_mesh_comparison(
    meshes: list[dict[str, Any]],
    *,
    shape: tuple[int, int] | None = None,
    bg: str = _MESH_BG,
    size: tuple[int, int] | None = None,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
    shared_scale: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Side-by-side comparison of multiple meshes.

    Each element in *meshes* is a dict with keys:

    - ``'vertices'`` : (V, 3) array (required)
    - ``'faces'`` : (F, 3) array (required)
    - ``'scalars'`` : (V,) array or None
    - ``'scalar_name'`` : str (default ``'value'``)
    - ``'cmap'`` : str or None
    - ``'vmin'``, ``'vmax'`` : float or None
    - ``'color'`` : str (default ``'gold'``)
    - ``'title'`` : str (default ``''``)

    Parameters
    ----------
    meshes : list of dict
        One dict per mesh panel.
    shape : (rows, cols) or None
        Grid layout.  None -> single row.
    bg, size, scale, save
        Standard render parameters.
    shared_scale : bool
        If True (default), panels whose ``vmin``/``vmax`` are not given share
        one colour range (1st/99th percentile over all panels' scalars), so
        the panels are directly comparable.  False -> per-panel ranges.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'n_panels'`` and ``'scalar_ranges'``.
    """
    vedo = _get_vedo()
    n = len(meshes)
    if shape is None:
        shape = (1, n)
    if size is None:
        size = (600 * shape[1], 600 * shape[0])

    plt = vedo.Plotter(shape=shape, offscreen=True, size=size, bg=bg)

    shared = None
    if shared_scale:
        pooled = [
            np.asarray(sp_["scalars"], dtype=np.float64).ravel()
            for sp_ in meshes
            if sp_.get("scalars") is not None
        ]
        if pooled:
            allv = np.concatenate(pooled)
            if np.isfinite(allv).any():
                shared = (float(np.nanpercentile(allv, 1)), float(np.nanpercentile(allv, 99)))
    ranges = []

    for i, spec in enumerate(meshes):
        m = _build_vedo_mesh(
            np.asarray(spec["vertices"]),
            np.asarray(spec["faces"]),
            vedo,
        )

        scalars = spec.get("scalars")
        scalar_name = spec.get("scalar_name", "value")
        cmap_name = _resolve_cmap(scalar_name, spec.get("cmap"))
        panel_title = spec.get("title", "")

        if scalars is not None:
            scalars = np.asarray(scalars, dtype=np.float64)
            v0, v1 = spec.get("vmin"), spec.get("vmax")
            if v0 is None:
                v0 = shared[0] if shared else float(np.nanpercentile(scalars, 1))
            if v1 is None:
                v1 = shared[1] if shared else float(np.nanpercentile(scalars, 99))
            ranges.append((v0, v1))
            m.pointdata[scalar_name] = scalars
            m.cmap(cmap_name, scalar_name, vmin=v0, vmax=v1)
            m.add_scalarbar(title=scalar_name)
        else:
            m.color(spec.get("color", "gold"))

        plt.at(i).show(m, title=panel_title, viewup="z", zoom=1.1)

    meta = {"n_panels": n, "shape": shape, "scalar_ranges": ranges}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# S6  Scalar difference map
# ======================================================================


def plot_scalar_difference(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars_a: np.ndarray,
    scalars_b: np.ndarray,
    *,
    label_a: str = "A",
    label_b: str = "B",
    diff_cmap: str = "RdBu_r",
    symmetric: bool = True,
    show_individual: bool = True,
    individual_cmap: str | None = None,
    shared_scale: bool = True,
    bg: str = _MESH_BG,
    size: tuple[int, int] | None = None,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Vertex-wise scalar difference map between two conditions.

    Computes ``scalars_a - scalars_b`` and displays the difference
    on the mesh surface with a diverging colourmap centred on zero.
    Optionally shows individual maps alongside.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    scalars_a, scalars_b : (V,) arrays
        Per-vertex values for conditions A and B.
    label_a, label_b : str
        Labels for panels.
    diff_cmap : str
        Colourmap for the difference (diverging recommended).
    symmetric : bool
        Centre the difference colourmap on zero.
    show_individual : bool
        Show A and B alongside the difference (3-panel layout).
    individual_cmap : str or None
        Colourmap for individual panels.  None -> 'viridis'.
    shared_scale : bool
        If True (default) panels A and B share one colour range (1st/99th
        percentile of both), so they are visually comparable.  False -> each
        individual panel is scaled to its own data.
    bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'diff_stats'``.
    """
    vedo = _get_vedo()
    scalars_a = np.asarray(scalars_a, dtype=np.float64)
    scalars_b = np.asarray(scalars_b, dtype=np.float64)
    diff = scalars_a - scalars_b

    n_panels = 3 if show_individual else 1
    if size is None:
        size = (600 * n_panels, 600)

    plt = vedo.Plotter(
        shape=(1, n_panels),
        offscreen=True,
        size=size,
        bg=bg,
    )

    panel_idx = 0
    ind_cmap = individual_cmap or "viridis"

    ind_range: tuple[float | None, float | None] = (None, None)
    if show_individual:
        if shared_scale:
            both = np.concatenate([scalars_a, scalars_b])
            if np.isfinite(both).any():
                ind_range = (
                    float(np.nanpercentile(both, 1)),
                    float(np.nanpercentile(both, 99)),
                )
        # Panel A
        m_a = _build_vedo_mesh(vertices, faces, vedo)
        m_a.pointdata[label_a] = scalars_a
        m_a.cmap(ind_cmap, label_a, vmin=ind_range[0], vmax=ind_range[1])
        m_a.add_scalarbar(title=label_a)
        plt.at(0).show(m_a, title=label_a, viewup="z", zoom=1.1)

        # Panel B
        m_b = _build_vedo_mesh(vertices, faces, vedo)
        m_b.pointdata[label_b] = scalars_b
        m_b.cmap(ind_cmap, label_b, vmin=ind_range[0], vmax=ind_range[1])
        m_b.add_scalarbar(title=label_b)
        plt.at(1).show(m_b, title=label_b, viewup="z", zoom=1.1)

        panel_idx = 2

    # Difference panel
    m_diff = _build_vedo_mesh(vertices, faces, vedo)
    m_diff.pointdata["Difference"] = diff

    diff_clean = diff[np.isfinite(diff)]
    if symmetric:
        p95 = float(np.percentile(np.abs(diff_clean), 95))
        d_vmin, d_vmax = -p95, p95
    else:
        d_vmin = float(np.percentile(diff_clean, 1))
        d_vmax = float(np.percentile(diff_clean, 99))

    m_diff.cmap(diff_cmap, "Difference", vmin=d_vmin, vmax=d_vmax)
    m_diff.add_scalarbar(title=f"{label_a} - {label_b}")
    plt.at(panel_idx).show(
        m_diff,
        title=f"Difference ({label_a} - {label_b})",
        viewup="z",
        zoom=1.1,
    )

    meta = {
        "diff_stats": {
            "mean": float(np.nanmean(diff)),
            "std": float(np.nanstd(diff)),
            "min": float(np.nanmin(diff)),
            "max": float(np.nanmax(diff)),
            "pct_positive": float(np.mean(diff[np.isfinite(diff)] > 0) * 100)
            if np.isfinite(diff).any()
            else float("nan"),
        },
        "individual_range": ind_range,
        "vmin": d_vmin,
        "vmax": d_vmax,
        "n_panels": n_panels,
    }
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# S7  PyVista fallback -- basic mesh render
# ======================================================================


def plot_mesh_pyvista(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray | None = None,
    cmap: str = "viridis",
    *,
    show_edges: bool = False,
    window_size: tuple[int, int] = (1600, 1200),
    save: PathLike | None = None,
) -> Path | None:
    """Minimal PyVista mesh render (fallback when vedo unavailable).

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    scalars : (V,) array or None
    cmap : str
    show_edges : bool
    window_size : (int, int)
    save : path or None

    Returns
    -------
    Path or None
        Output path if successful, None otherwise.
    """
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("PyVista not available -- cannot render mesh")
        return None

    pv.OFF_SCREEN = True

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=int)
    # PyVista expects faces as [3, i, j, k, 3, i, j, k, ...]
    pv_faces = np.column_stack([np.full(len(faces), 3, dtype=int), faces]).ravel()

    mesh = pv.PolyData(vertices, pv_faces)
    if scalars is not None:
        mesh.point_data["scalars"] = np.asarray(scalars, dtype=np.float64)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(
        mesh,
        scalars="scalars" if scalars is not None else None,
        cmap=cmap,
        show_edges=show_edges,
    )
    plotter.view_isometric()

    if save is None:
        fd, save = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    save = Path(save)
    plotter.screenshot(str(save))
    plotter.close()

    logger.info("Saved PyVista render -> %s", save)
    return save


# ======================================================================
# SD  TEMPLATE-FREE SIX-VIEW ENGINE
# ======================================================================


#: The six canonical views, in the order the user reads them (2x3 grid).
SIXVIEWS: tuple[str, ...] = (
    "anterior",
    "posterior",
    "inferior",
    "superior",
    "left_lateral",
    "right_lateral",
)


# ======================================================================
# S1  ENGINE SETUP
# ======================================================================


def _require_vedo() -> Any:
    """Lazy-import vedo in offscreen mode."""
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    try:
        import vedo
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vedo is required for 3D surface rendering.\n  pip install vedo") from exc
    vedo.settings.default_backend = "vtk"
    return vedo


# ======================================================================
# S2  SURFACE LOADING (geometry-agnostic)
# ======================================================================


def _gifti_surface(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Extract (coords, faces) from a GIFTI surface by dtype, not intent.

    HippUnfold GIFTIs store coords as float ``(N, 3)`` and faces as int
    ``(M, 3)``; ``agg_data()`` returns them in array order (often faces
    first), so we select by dtype+shape rather than positional order.
    """
    import nibabel as nib

    coords = faces = None
    for da in nib.load(str(path)).darrays:
        a = np.asarray(da.data)
        if a.ndim == 2 and a.shape[1] == 3:
            if a.dtype.kind == "f":
                coords = a
            elif a.dtype.kind in "iu":
                faces = a
    if coords is None or faces is None:
        raise ValueError(f"Could not read a triangulated surface from {path}")
    return coords.astype(np.float64), faces.astype(np.int64)


def _load_surface(surface: Any) -> tuple[np.ndarray, np.ndarray]:
    """Resolve many inputs to ``(coords, faces)``.

    Accepts: a ``BrainMesh`` (``.vertices``/``.faces``), a
    ``(coords, faces)`` tuple, a GIFTI ``.surf.gii`` path, or a
    FreeSurfer geometry path.
    """
    # BrainMesh-like.
    if hasattr(surface, "vertices") and hasattr(surface, "faces"):
        return np.asarray(surface.vertices, float), np.asarray(surface.faces, np.int64)
    # (coords, faces) tuple.
    if isinstance(surface, (tuple, list)) and len(surface) == 2:
        return np.asarray(surface[0], float), np.asarray(surface[1], np.int64)
    # Path.
    p = Path(surface)
    name = p.name.lower()
    if name.endswith((".gii", ".gii.gz")):
        return _gifti_surface(p)
    import nibabel as nib

    v, f = nib.freesurfer.read_geometry(str(p))
    return np.asarray(v, float), np.asarray(f, np.int64)


def _build_mesh(
    coords: np.ndarray,
    faces: np.ndarray,
    *,
    smooth_iter: int,
    vedo: Any,
) -> Any:
    """Build a clean vedo mesh from raw arrays (geometry only).

    Geometric Laplacian smoothing reorders vedo point data and therefore
    scrambles any attached per-vertex scalar -- so smoothing is applied
    here only to the bare geometry, *before* any scalar is attached, and
    the caller must skip it when rendering a scalar field. Visual
    smoothness for scalar renders comes from Phong shading instead, which
    interpolates normals without moving vertices or touching point data.
    """
    coords = np.asarray(coords, float)
    faces = np.asarray(faces, np.int64)
    nv = coords.shape[0]
    # Drop padding / out-of-range faces (merged hippdentate carries -1).
    faces = faces[np.all((faces >= 0) & (faces < nv), axis=1)]
    mesh = vedo.Mesh([coords, faces])
    if smooth_iter and smooth_iter > 0:
        mesh = mesh.smooth(niter=smooth_iter)
    mesh.compute_normals()
    return mesh


# ======================================================================
# S3  CAMERAS
# ======================================================================


def _sixview_cameras(center: np.ndarray, radius: float) -> dict[str, dict]:
    """Camera dicts for the six canonical views in RAS space.

    RAS axes: x = right(+)/left(-), y = anterior(+)/posterior(-),
    z = superior(+)/inferior(-).
    """
    cx, cy, cz = center
    r = radius
    return {
        "anterior": dict(position=[cx, cy + r, cz], focal_point=[cx, cy, cz], viewup=[0, 0, 1]),
        "posterior": dict(position=[cx, cy - r, cz], focal_point=[cx, cy, cz], viewup=[0, 0, 1]),
        "superior": dict(position=[cx, cy, cz + r], focal_point=[cx, cy, cz], viewup=[0, 1, 0]),
        "inferior": dict(position=[cx, cy, cz - r], focal_point=[cx, cy, cz], viewup=[0, 1, 0]),
        "left_lateral": dict(position=[cx - r, cy, cz], focal_point=[cx, cy, cz], viewup=[0, 0, 1]),
        "right_lateral": dict(
            position=[cx + r, cy, cz], focal_point=[cx, cy, cz], viewup=[0, 0, 1]
        ),
    }


_VIEW_LABELS: dict[str, str] = {
    "anterior": "Anterior",
    "posterior": "Posterior",
    "inferior": "Inferior",
    "superior": "Superior",
    "left_lateral": "Left lateral",
    "right_lateral": "Right lateral",
}


# ======================================================================
# S4  RENDERING
# ======================================================================


def _render_one(
    mesh: Any,
    cam: dict,
    *,
    window: tuple[int, int],
    parallel_scale: float,
    vedo: Any,
) -> np.ndarray:
    """Render a single view offscreen -> RGB array (orthographic).

    Orthographic projection (no perspective foreshortening -- right for
    anatomical figures) with an explicit ``parallel_scale`` (half the
    viewport height in world units) computed per view to frame the mesh
    tightly with margin: no cropping, no wasted whitespace, and the
    explicit camera dict keeps each view's orientation distinct.
    """
    plt = vedo.Plotter(offscreen=True, size=window, bg="white", axes=0)
    plt.show(mesh, camera=cam, interactive=False)
    camobj = plt.camera
    camobj.ParallelProjectionOn()
    camobj.SetParallelScale(parallel_scale)
    plt.render()
    img = plt.screenshot(asarray=True)
    plt.close()
    return np.asarray(img)


#: For each axis-aligned view, the (horizontal, vertical) world axes that
#: map to the image plane -- used to frame each view exactly.
_VIEW_PLANE: dict[str, tuple[int, int]] = {
    "anterior": (0, 2),  # see x-z plane
    "posterior": (0, 2),
    "superior": (0, 1),  # see x-y plane
    "inferior": (0, 1),
    "left_lateral": (1, 2),  # see y-z plane
    "right_lateral": (1, 2),
}


def _view_scale(view: str, extent: np.ndarray, window: tuple[int, int], pad: float) -> float:
    """Exact parallel scale (half-height) to frame ``view`` with margin."""
    w, h = window
    hi, vi = _VIEW_PLANE.get(view, (0, 1))
    half_v = 0.5 * float(extent[vi])
    half_h = 0.5 * float(extent[hi])
    # Fit both dimensions: scale >= half-height and >= half-width*(h/w).
    return max(half_v, half_h * (h / w)) * (1.0 + pad)


def _sixview_figure(
    coords: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray | None,
    *,
    cmap: str | None,
    signed: bool,
    clim: tuple[float, float] | None,
    scalar_bar_title: str,
    title: str | None,
    views: tuple[str, ...],
    smooth_iter: int,
    surface_color: str,
    window: tuple[int, int],
    pad: float,
    save: PathLike | None,
    formats: list[str] | None,
    nan_color: str = "lightgray",
):
    """Core: render the requested views with vedo, compose in matplotlib."""
    import matplotlib.pyplot as plt

    views = tuple(views)
    unknown = [v for v in views if v not in _VIEW_PLANE]
    if unknown:
        raise ValueError(f"Unknown view(s) {unknown}. Valid views: {list(_VIEW_PLANE)}")
    vedo = _require_vedo()

    # -- colour mapping ------------------------------------------------
    have_scalars = scalars is not None
    mappable = None
    if have_scalars:
        # Scalar fidelity: build WITHOUT geometric smoothing (which would
        # scramble the scalar<->vertex correspondence); Phong shading gives
        # visual smoothness without moving vertices or reordering data.
        mesh = _build_mesh(coords, faces, smooth_iter=0, vedo=vedo)
        s = np.asarray(scalars, float)
        if len(s) != mesh.npoints:
            raise ValueError(f"scalars length ({len(s)}) != mesh vertices ({mesh.npoints}).")
        finite = s[np.isfinite(s)]
        if cmap is None:
            cmap = "RdBu_r" if signed else "plasma"
        if clim is None:
            lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
            if signed:
                m = max(abs(lo), abs(hi)) or 1.0
                clim = (-m, m)
            else:
                clim = (float(lo), float(hi))
        from matplotlib import cm, colors

        norm = colors.Normalize(vmin=clim[0], vmax=clim[1])
        # Colour per vertex in matplotlib so NaN (medial wall / missing) gets
        # an explicit ``nan_color`` -- never the colour of clim[0], which for a
        # signed map would read as the strongest negative effect.
        lut = plt.get_cmap(cmap).copy()
        lut.set_bad(nan_color)
        rgba = lut(norm(np.ma.masked_invalid(s)))
        mesh.pointdata["RGBA"] = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        mesh.pointdata.select("RGBA")
        mesh.phong()  # smooth shading without geometric change
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        # Geometry-only: smoothing is safe (no scalar to scramble).
        mesh = _build_mesh(coords, faces, smooth_iter=smooth_iter, vedo=vedo)
        mesh.color(surface_color).phong()

    # -- per-view renders ----------------------------------------------
    center = np.array(mesh.center_of_mass())
    extent = np.ptp(mesh.points, axis=0)
    diag = float(np.linalg.norm(extent))
    cam_dist = diag * 3.0  # camera distance (irrelevant under ortho; clear of mesh)
    cams = _sixview_cameras(center, cam_dist)
    imgs = {}
    for v in views:
        cam = cams[v]
        scale = _view_scale(v, extent, window, pad)
        imgs[v] = _render_one(mesh.clone(), cam, window=window, parallel_scale=scale, vedo=vedo)

    # -- matplotlib composition ----------------------------------------
    n = len(views)
    ncol = 3 if n >= 3 else n
    nrow = int(np.ceil(n / ncol))
    fig_w = ncol * 2.4 + (1.0 if have_scalars else 0.0)
    fig_h = nrow * 2.3 + (0.5 if title else 0.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    for i, v in enumerate(views):
        ax = axes[i]
        ax.imshow(imgs[v])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(_VIEW_LABELS.get(v, v), fontsize=8, style="italic", color="0.3")
    for j in range(len(views), len(axes)):
        axes[j].axis("off")

    if mappable is not None:
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=axes.tolist(), fraction=0.025, pad=0.02, aspect=30)
        cbar.set_label(scalar_bar_title, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=11, y=0.99)

    if save is not None:
        from spectralbrain.viz.graphics import savefig

        savefig(fig, save, formats=formats, dpi=DPI)
    return fig


def plot_surface_sixview(
    surface: Any,
    scalars: np.ndarray | None = None,
    *,
    hemi: str = "L",
    cmap: str | None = None,
    signed: bool = False,
    clim: tuple[float, float] | None = None,
    scalar_bar_title: str = "value",
    title: str | None = None,
    views: tuple[str, ...] = SIXVIEWS,
    smooth_iter: int = 0,
    surface_color: str = "lightsteelblue",
    window: tuple[int, int] = (560, 520),
    pad: float = 0.05,
    save: PathLike | None = None,
    formats: list[str] | None = None,
    nan_color: str = "lightgray",
):
    """Render any surface in six canonical anatomical views.

    Template-free six-view engine for any ``(coords, faces)`` -- a cortical
    hemisphere, a subcortical ROI, a bundle surface or a hippocampus.
    Vertex-to-scalar correspondence is guaranteed because the scalar is
    rendered on the very mesh it was computed on. Defaults are tuned for
    larger meshes (no extra smoothing); for hippocampal defaults use
    :func:`spectralbrain.viz.hipp.plot_hippocampus_sixview`.

    Parameters
    ----------
    surface : BrainMesh, (coords, faces), or path
        The surface (GIFTI ``.surf.gii`` or FreeSurfer
        geometry paths are accepted).
    scalars : ndarray, shape (N,), optional
        Per-vertex field (HKS, thickness, Cohen's d, ...). If ``None`` the
        bare geometry is rendered in ``surface_color``.
    hemi : {"L", "R"}
        Hemisphere label (annotation only; cameras are anatomical).
    cmap : str, optional
        Defaults to ``"plasma"`` (unsigned) or ``"RdBu_r"`` (``signed``).
    signed : bool
        Symmetric colour limits about zero (for contrasts / t-stats).
    clim : (lo, hi), optional
        Manual colour limits; else 2nd-98th percentile.
    scalar_bar_title : str
        Colorbar label (include units, e.g. ``"Thickness (mm)"``).
    title : str, optional
        Figure suptitle.
    views : tuple of str
        Subset / ordering of :data:`SIXVIEWS`.
    smooth_iter : int
        Laplacian smoothing iterations (0 = render the mesh as given).
    surface_color : str
        Mesh colour when ``scalars`` is ``None`` (named/hex; not a gray
        string).
    window : (w, h)
        Per-view render size in pixels (scaled up by anti-aliasing).
    pad : float
        Margin fraction around the fitted mesh (0 = tightest framing).
        The camera auto-fits the bounds per view, so no view is cropped.
    save : path, optional
        If given, write the figure (``formats`` controls extensions).
    nan_color : str
        Colour for vertices whose scalar is NaN (e.g. medial wall).

    Returns
    -------
    matplotlib.figure.Figure
    """
    coords, faces = _load_surface(surface)
    return _sixview_figure(
        coords,
        faces,
        scalars,
        cmap=cmap,
        signed=signed,
        clim=clim,
        scalar_bar_title=scalar_bar_title,
        title=title,
        views=views,
        smooth_iter=smooth_iter,
        surface_color=surface_color,
        window=window,
        pad=pad,
        save=save,
        formats=formats,
        nan_color=nan_color,
    )


__all__: list[str] = [
    "CAMERA_PRESETS",
    "CURVATURE_METHODS",
    "DESCRIPTOR_STYLES",
    "DPI",
    "SIXVIEWS",
    "TRACT_VIEWS",
    "VIEWS_CORTEX",
    "VIEWS_FULL",
    "VIEWS_MEDIAL",
    "BrainPlotSpec",
    "add_glass_brain",
    "compose_tract_panel",
    "get_cmap",
    "load_streamlines",
    "mask_to_mesh",
    "plot_bilateral_comparison",
    "plot_brain",
    "plot_brain_subcortical",
    "plot_brain_tracts",
    "plot_clustering_map",
    "plot_curvature",
    "plot_group_comparison",
    "plot_mesh",
    "plot_mesh_comparison",
    "plot_mesh_pyvista",
    "plot_morphometric_gallery",
    "plot_multi_descriptor_panel",
    "plot_multi_view",
    "plot_normative_map",
    "plot_scalar_difference",
    "plot_spectral_progression",
    "plot_surface_sixview",
    "plot_top10_morphometrics",
    "plot_wireframe",
    "render_bundle_surface",
    "render_streamlines",
    "resolve_scalar_clim",
    "robust_clim",
    "spectral_overlay",
    "streamlines_multiview",
]
