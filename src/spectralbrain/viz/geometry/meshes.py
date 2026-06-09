"""3D mesh rendering with vedo and optional PyVista fallback.

This module handles publication-quality 3D renders of triangular meshes
for SpectralBrain's mesh-based analysis pathway.  It complements
``points.py`` (which handles atlas-free point clouds) by covering
scenarios where mesh connectivity *is* available — either from
FreeSurfer surface files, HippUnfold outputs, or reconstructed from
point clouds via Poisson / Delaunay.

The six figure types cover the critical mesh visual outputs:

1. **Surface render** — smooth-shaded mesh with optional scalar
   overlay (HKS, WKS, thickness, curvature).
2. **Wireframe render** — mesh topology visualisation for QC and
   methods figures.
3. **Curvature map** — Gaussian, mean, principal curvatures computed
   and displayed directly on the mesh surface.
4. **Multi-view panel** — same mesh from multiple camera angles
   (anterior, posterior, lateral, medial, superior, inferior).
5. **Mesh comparison** — side-by-side panels comparing two or more
   meshes (e.g., left vs right hemisphere, patient vs control).
6. **Scalar difference map** — vertex-wise difference between two
   meshes overlaid as a diverging colourmap.

All functions follow the SpectralBrain convention: return
``(Path, metadata_dict)`` for vedo-based renders.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spectralbrain.runtime import PathLike, get_logger

if TYPE_CHECKING:
    import vedo

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  Constants — shared with points.py
# ---------------------------------------------------------------------------

_DEFAULT_SIZE: tuple[int, int] = (1600, 1200)
_DEFAULT_SCALE: int = 2
_DEFAULT_BG: str = "white"

# Curvature method codes used by VTK / vedo
CURVATURE_METHODS: dict[str, int] = {
    "gaussian": 0,
    "mean": 1,
    "maximum": 2,
    "minimum": 3,
}

# Standard multi-view camera presets (azimuth, elevation)
CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "anterior": {"azimuth": 0, "elevation": 0},
    "posterior": {"azimuth": 180, "elevation": 0},
    "left_lateral": {"azimuth": -90, "elevation": 0},
    "right_lateral": {"azimuth": 90, "elevation": 0},
    "superior": {"azimuth": 0, "elevation": 90},
    "inferior": {"azimuth": 0, "elevation": -90},
    "left_medial": {"azimuth": 90, "elevation": 0},
    "right_medial": {"azimuth": -90, "elevation": 0},
}


# ======================================================================
# §0  Lazy imports & helpers
# ======================================================================


def _ensure_offscreen() -> None:
    """Set vedo to offscreen rendering mode."""
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")


def _get_vedo():
    """Lazy-import vedo, raising ImportError if unavailable."""
    _ensure_offscreen()
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
    logger.info("Saved mesh render → %s", save)
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
        key = scalar_name.lower().replace(" ", "_").split("_")[0]
        return LOOKUP.get(key, "viridis")
    return "viridis"


# ======================================================================
# §1  Surface render — smooth-shaded mesh with scalar overlay
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
    bg: str = _DEFAULT_BG,
    size: tuple[int, int] = _DEFAULT_SIZE,
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
        Per-vertex scalar values.  None → uniform ``color``.
    scalar_name : str
        Label for colourbar and automatic cmap selection.
    cmap : str or None
        Colourmap.  None → auto from scalar_name.
    vmin, vmax : float or None
        Colour range.  None → 1st / 99th percentiles.
    color : str
        Uniform mesh colour when scalars is None.
    alpha : float
        Mesh opacity (0–1).
    show_edges : bool
        Overlay wireframe edges.
    edge_color, edge_width : str, float
        Edge appearance.
    show_scalarbar : bool
        Display colourbar.
    lighting : str
        VTK lighting style — ``'default'``, ``'metallic'``,
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
# §2  Wireframe render
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
    bg: str = _DEFAULT_BG,
    size: tuple[int, int] = _DEFAULT_SIZE,
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
# §3  Curvature map
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
    bg: str = _DEFAULT_BG,
    size: tuple[int, int] = _DEFAULT_SIZE,
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
        range is set to ± 95th percentile.
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
# §4  Multi-view panel — same mesh from multiple camera angles
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
    bg: str = _DEFAULT_BG,
    size: tuple[int, int] | None = None,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Multi-view panel showing the same mesh from different angles.

    Renders the same mesh (optionally with scalar overlay) in a 1×N
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
        offscreen=True,
        size=size,
        bg=bg,
    )

    for i, view_name in enumerate(views):
        m = base.clone()
        preset = CAMERA_PRESETS.get(view_name, {})

        plt.at(i).show(
            m,
            title=view_name.replace("_", " ").title(),
            viewup="z",
            zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
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
# §5  Mesh comparison — side-by-side panels
# ======================================================================


def plot_mesh_comparison(
    meshes: list[dict[str, Any]],
    *,
    shape: tuple[int, int] | None = None,
    bg: str = _DEFAULT_BG,
    size: tuple[int, int] | None = None,
    scale: int = _DEFAULT_SCALE,
    save: PathLike | None = None,
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
        Grid layout.  None → single row.
    bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'n_panels'``.
    """
    vedo = _get_vedo()
    n = len(meshes)
    if shape is None:
        shape = (1, n)
    if size is None:
        size = (600 * shape[1], 600 * shape[0])

    plt = vedo.Plotter(shape=shape, offscreen=True, size=size, bg=bg)

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
            v0 = spec.get("vmin") or float(np.nanpercentile(scalars, 1))
            v1 = spec.get("vmax") or float(np.nanpercentile(scalars, 99))
            m.pointdata[scalar_name] = scalars
            m.cmap(cmap_name, scalar_name, vmin=v0, vmax=v1)
            m.add_scalarbar(title=scalar_name)
        else:
            m.color(spec.get("color", "gold"))

        plt.at(i).show(m, title=panel_title, viewup="z", zoom=1.1)

    meta = {"n_panels": n, "shape": shape}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §6  Scalar difference map
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
    bg: str = _DEFAULT_BG,
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
        Colourmap for individual panels.  None → 'viridis'.
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

    if show_individual:
        # Panel A
        m_a = _build_vedo_mesh(vertices, faces, vedo)
        m_a.pointdata[label_a] = scalars_a
        m_a.cmap(ind_cmap, label_a)
        m_a.add_scalarbar(title=label_a)
        plt.at(0).show(m_a, title=label_a, viewup="z", zoom=1.1)

        # Panel B
        m_b = _build_vedo_mesh(vertices, faces, vedo)
        m_b.pointdata[label_b] = scalars_b
        m_b.cmap(ind_cmap, label_b)
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
    m_diff.add_scalarbar(title=f"{label_a} − {label_b}")
    plt.at(panel_idx).show(
        m_diff,
        title=f"Difference ({label_a} − {label_b})",
        viewup="z",
        zoom=1.1,
    )

    meta = {
        "diff_stats": {
            "mean": float(np.nanmean(diff)),
            "std": float(np.nanstd(diff)),
            "min": float(np.nanmin(diff)),
            "max": float(np.nanmax(diff)),
            "pct_positive": float(np.mean(diff > 0) * 100),
        },
        "vmin": d_vmin,
        "vmax": d_vmax,
        "n_panels": n_panels,
    }
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §7  PyVista fallback — basic mesh render
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
        logger.warning("PyVista not available — cannot render mesh")
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

    logger.info("Saved PyVista render → %s", save)
    return save


# ======================================================================
# __all__
# ======================================================================

__all__ = [
    "CAMERA_PRESETS",
    # Constants
    "CURVATURE_METHODS",
    "plot_curvature",
    # Core renders
    "plot_mesh",
    "plot_mesh_comparison",
    # PyVista fallback
    "plot_mesh_pyvista",
    "plot_multi_view",
    "plot_scalar_difference",
    "plot_wireframe",
]
