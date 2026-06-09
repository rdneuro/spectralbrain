"""Template-free 3D surface rendering in six canonical anatomical views.

The hippocampal renderers in :mod:`spectralbrain.viz.hipp` delegate to
``hippunfold_plot`` / ``hippomaps``, which assume the *bundled* HippUnfold
templates and therefore silently mismatch when handed a raw surface whose
vertex count does not match a template — exactly the case for HippUnfold
**v2 ``den-8k``** meshes or separate ``hipp``/``dentate`` surfaces.

This module renders *whatever mesh you hand it* — vertex↔scalar
correspondence is guaranteed by construction — in the six standard
neuroanatomical views (anterior, posterior, inferior, superior,
left-lateral, right-lateral). Rendering is done with **vedo** (offscreen
VTK) and the six RGB renders are composited into a single matplotlib
figure with a shared colorbar and view labels, following the HipPlots
annotation conventions (view labels below each render, colorbar on the
right, no text over the surface).

It is geometry-agnostic: the same engine renders a HippUnfold hippocampus,
a marching-cubes mesh of an ``aseg`` ROI, or a whole cortical hemisphere.

Examples
--------
>>> import spectralbrain as sb
>>> mesh = sb.BrainMesh(*sb.io.load_freesurfer_surface("lh.white"))
>>> fig = sb.viz.plot_hippocampus_sixview(mesh, scalars=thickness,
...                                       scalar_bar_title="Thickness (mm)")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

DPI: int = 600

#: The six canonical views, in the order the user reads them (2×3 grid).
SIXVIEWS: tuple[str, ...] = (
    "anterior",
    "posterior",
    "inferior",
    "superior",
    "left_lateral",
    "right_lateral",
)


# ======================================================================
# §1  ENGINE SETUP
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
# §2  SURFACE LOADING (geometry-agnostic)
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
    scrambles any attached per-vertex scalar — so smoothing is applied
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
# §3  CAMERAS
# ======================================================================


def _sixview_cameras(center: np.ndarray, radius: float) -> dict[str, dict]:
    """Camera dicts for the six canonical views in RAS space.

    RAS axes: x = right(+)/left(−), y = anterior(+)/posterior(−),
    z = superior(+)/inferior(−).
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
# §4  RENDERING
# ======================================================================


def _render_one(
    mesh: Any,
    cam: dict,
    *,
    window: tuple[int, int],
    parallel_scale: float,
    vedo: Any,
) -> np.ndarray:
    """Render a single view offscreen → RGB array (orthographic).

    Orthographic projection (no perspective foreshortening — right for
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
#: map to the image plane — used to frame each view exactly.
_VIEW_PLANE: dict[str, tuple[int, int]] = {
    "anterior": (0, 2),  # see x–z plane
    "posterior": (0, 2),
    "superior": (0, 1),  # see x–y plane
    "inferior": (0, 1),
    "left_lateral": (1, 2),  # see y–z plane
    "right_lateral": (1, 2),
}


def _view_scale(view: str, extent: np.ndarray, window: tuple[int, int], pad: float) -> float:
    """Exact parallel scale (half-height) to frame ``view`` with margin."""
    w, h = window
    hi, vi = _VIEW_PLANE.get(view, (0, 1))
    half_v = 0.5 * float(extent[vi])
    half_h = 0.5 * float(extent[hi])
    # Fit both dimensions: scale ≥ half-height and ≥ half-width·(h/w).
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
):
    """Core: render the requested views with vedo, compose in matplotlib."""
    import matplotlib.pyplot as plt

    vedo = _require_vedo()

    # ── colour mapping ────────────────────────────────────────────────
    have_scalars = scalars is not None
    mappable = None
    if have_scalars:
        # Scalar fidelity: build WITHOUT geometric smoothing (which would
        # scramble the scalar↔vertex correspondence); Phong shading gives
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
        s_filled = np.where(np.isfinite(s), s, clim[0])
        mesh.cmap(cmap, s_filled, vmin=clim[0], vmax=clim[1])
        mesh.phong()  # smooth shading without geometric change
        from matplotlib import cm, colors

        norm = colors.Normalize(vmin=clim[0], vmax=clim[1])
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        # Geometry-only: smoothing is safe (no scalar to scramble).
        mesh = _build_mesh(coords, faces, smooth_iter=smooth_iter, vedo=vedo)
        mesh.color(surface_color).phong()

    # ── per-view renders ──────────────────────────────────────────────
    center = np.array(mesh.center_of_mass())
    extent = np.ptp(mesh.points, axis=0)
    diag = float(np.linalg.norm(extent))
    cam_dist = diag * 3.0  # camera distance (irrelevant under ortho; clear of mesh)
    cams = _sixview_cameras(center, cam_dist)
    imgs = {}
    for v in views:
        cam = cams.get(v, cams["superior"])
        scale = _view_scale(v, extent, window, pad)
        imgs[v] = _render_one(mesh.clone(), cam, window=window, parallel_scale=scale, vedo=vedo)

    # ── matplotlib composition ────────────────────────────────────────
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


# ======================================================================
# §5  PUBLIC API
# ======================================================================


def plot_hippocampus_sixview(
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
    smooth_iter: int = 10,
    surface_color: str = "wheat",
    window: tuple[int, int] = (560, 520),
    pad: float = 0.05,
    save: PathLike | None = None,
    formats: list[str] | None = None,
):
    """Render a hippocampal surface in six canonical anatomical views.

    Template-free: works directly on HippUnfold v2 ``den-8k`` meshes,
    separate ``hipp``/``dentate`` surfaces, or any ``(coords, faces)`` —
    vertex↔scalar correspondence is guaranteed because the scalar is
    rendered on the very mesh it was computed on.

    Parameters
    ----------
    surface : BrainMesh, (coords, faces), or path
        The hippocampal surface (GIFTI ``.surf.gii`` or FreeSurfer
        geometry paths are accepted).
    scalars : ndarray, shape (N,), optional
        Per-vertex field (HKS, thickness, Cohen's d, …). If ``None`` the
        bare geometry is rendered in ``surface_color``.
    hemi : {"L", "R"}
        Hemisphere label (annotation only; cameras are anatomical).
    cmap : str, optional
        Defaults to ``"plasma"`` (unsigned) or ``"RdBu_r"`` (``signed``).
    signed : bool
        Symmetric colour limits about zero (for contrasts / t-stats).
    clim : (lo, hi), optional
        Manual colour limits; else 2nd–98th percentile.
    scalar_bar_title : str
        Colorbar label (include units, e.g. ``"Thickness (mm)"``).
    title : str, optional
        Figure suptitle.
    views : tuple of str
        Subset / ordering of :data:`SIXVIEWS`.
    smooth_iter : int
        Laplacian smoothing iterations (10–15 keeps anatomical detail).
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
    )


def plot_surface_sixview(
    surface: Any,
    scalars: np.ndarray | None = None,
    *,
    smooth_iter: int = 0,
    surface_color: str = "lightsteelblue",
    **kwargs: Any,
):
    """Six-view render for any surface (cortical hemisphere, subcortical ROI).

    Thin wrapper over :func:`plot_hippocampus_sixview` with defaults tuned
    for larger meshes (no subdivision; no extra smoothing). All keyword
    arguments are forwarded.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return plot_hippocampus_sixview(
        surface,
        scalars,
        smooth_iter=smooth_iter,
        surface_color=surface_color,
        **kwargs,
    )


__all__ = [
    "SIXVIEWS",
    "plot_hippocampus_sixview",
    "plot_surface_sixview",
]
