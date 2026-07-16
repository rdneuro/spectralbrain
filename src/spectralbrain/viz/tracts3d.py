"""Advanced 3D white-matter tractography visualization.

Library-native implementation of the ``neuro-tracts`` conventions: streamline
rendering with direction-encoded (DEC) or scalar colouring, bundle-surface meshes
with embedded scalar/spectral (HKS/WKS) overlays, glass-brain context, and
multi-POV montages — all rendered **offscreen** with depth peeling and
anti-aliasing, then composited into hybrid raster+vector publication panels.

Engine choice (mirrors neuro-tracts):

=============================  =========  ==========================
Rendering                      Engine     Entry point
=============================  =========  ==========================
Streamlines / tractogram       FURY       :func:`render_streamlines`
Bundle surface + scalar/HKS    PyVista    :func:`render_bundle_surface`
Multi-POV montage of a bundle  composite  :func:`streamlines_multiview`
Publication panel              matplotlib :func:`compose_tract_panel`
=============================  =========  ==========================

Colour policy: DEC RGB for orientation (x=L-R red, y=A-P green, z=I-S blue);
perceptually-uniform sequential (Crameri ``batlow`` if ``cmcrameri`` present, else
``viridis``) for FA/MD/HKS/density; diverging centred on zero for signed maps;
robust 2-98% colour limits. Never jet/rainbow.

All heavy deps (FURY/DIPY, PyVista, nibabel, scikit-image, trimesh, cmcrameri)
are imported lazily; the module imports cleanly without them and raises a clear
message only when a path needs one.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger("spectralbrain.viz.tracts3d")

PathLike = "str | os.PathLike[str]"

# Canonical camera viewpoints (azimuth, elevation) in the RAS+ world frame.
TRACT_VIEWS: dict[str, dict[str, float]] = {
    "left": {"azimuth": 180.0, "elevation": 0.0},
    "right": {"azimuth": 0.0, "elevation": 0.0},
    "anterior": {"azimuth": 90.0, "elevation": 0.0},
    "posterior": {"azimuth": -90.0, "elevation": 0.0},
    "superior": {"azimuth": 0.0, "elevation": 90.0},
    "inferior": {"azimuth": 0.0, "elevation": -90.0},
    "oblique": {"azimuth": 135.0, "elevation": 25.0},
}

_DEFAULT_SIZE = (1600, 1600)
_DEFAULT_BG = "white"


# ──────────────────────────────────────────────────────────────────────
# Lazy dependency requires
# ──────────────────────────────────────────────────────────────────────
def _ensure_offscreen() -> None:
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


def _require_fury():
    _ensure_offscreen()
    try:
        from fury import actor, window  # noqa: F401
        return actor, window
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FURY is required for streamline rendering. "
                          "Install with: pip install fury dipy") from exc


def _require_dipy_io():
    try:
        from dipy.io.streamline import load_tractogram  # noqa: F401
        from dipy.tracking.streamline import (  # noqa: F401
            select_random_set_of_streamlines, transform_streamlines)
        return load_tractogram, select_random_set_of_streamlines, transform_streamlines
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DIPY is required to load streamlines. "
                          "Install with: pip install dipy") from exc


def _require_pyvista():
    _ensure_offscreen()
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        return pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyVista is required for bundle-surface rendering. "
                          "Install with: pip install pyvista") from exc


def _require_skimage_mc():
    try:
        from skimage.measure import marching_cubes
        return marching_cubes
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-image is required for mask->mesh. "
                          "Install with: pip install scikit-image") from exc


# ──────────────────────────────────────────────────────────────────────
# Colour helpers (harmonised with spectralbrain.viz.graphics policy)
# ──────────────────────────────────────────────────────────────────────
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


def robust_clim(values: np.ndarray, low: float = 2.0, high: float = 98.0,
                symmetric: bool = False) -> tuple[float, float]:
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


# ──────────────────────────────────────────────────────────────────────
# 1. Streamlines
# ──────────────────────────────────────────────────────────────────────
def load_streamlines(path: PathLike, reference: Optional[PathLike] = None,
                     to_space: str = "world"):
    """Load a ``.trk``/``.tck`` tractogram into world (RAS+ mm) coordinates.

    ``.tck`` (MRtrix) stores no affine and needs ``reference`` (a NIfTI).
    Returns ``(streamlines, affine)``.
    """
    load_tractogram, _, _ = _require_dipy_io()
    from dipy.io.stateful_tractogram import Space
    ref = reference if reference is not None else "same"
    sft = load_tractogram(str(path), ref, to_space=Space.RASMM)
    return list(sft.streamlines), sft.affine


def render_streamlines(streamlines, *, color_by: str = "orientation",
                       scalars: Optional[Sequence[np.ndarray]] = None,
                       tube: bool = True, linewidth: float = 0.4,
                       cmap: Any = None, clim: Optional[tuple] = None,
                       robust: bool = True, view: str = "oblique",
                       n_max: int = 20000, bg: str = _DEFAULT_BG,
                       size: tuple[int, int] = _DEFAULT_SIZE,
                       out_path: Optional[PathLike] = None,
                       random_state: int = 0) -> Path:
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
        lo, hi = clim or (robust_clim(allv) if robust else (allv.min(), allv.max()))
        cm = cmap if cmap is not None else get_cmap("sequential")
        rng_span = (hi - lo) or 1.0
        colors = [cm(np.clip((np.asarray(s, float) - lo) / rng_span, 0, 1))[:, :3]
                  for s in scalars]
    else:
        raise ValueError("color_by must be 'orientation' or 'scalar'.")

    scene = window.Scene()
    scene.background({"white": (1, 1, 1), "black": (0, 0, 0)}.get(bg, (1, 1, 1)))
    if tube:
        stream_actor = actor.streamtube(sl, colors=colors, linewidth=linewidth)
    else:
        stream_actor = actor.line(sl, colors=colors, linewidth=max(linewidth, 1.0))
    scene.add(stream_actor)
    _set_fury_camera(scene, view)

    # depth peeling + anti-aliasing for correct transparency / clean edges
    out = Path(out_path) if out_path else Path(tempfile.mkstemp(suffix=".png")[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        window.record(scene=scene, out_path=str(out), size=size,
                      multi_samples=8, reset_camera=False)
    except TypeError:  # older FURY positional/keyword drift
        window.record(scene, out_path=str(out), size=size)
    logger.info("Saved streamline render → %s", out)
    return out


def _set_fury_camera(scene, view: str) -> None:
    scene.reset_camera()
    preset = TRACT_VIEWS.get(view, TRACT_VIEWS["oblique"])
    try:
        scene.azimuth(preset["azimuth"])
        scene.elevation(preset["elevation"])
    except Exception:  # pragma: no cover - camera API drift
        pass


# ──────────────────────────────────────────────────────────────────────
# 2. Bundle surfaces with scalar / spectral overlays
# ──────────────────────────────────────────────────────────────────────
def mask_to_mesh(mask: np.ndarray, affine: Optional[np.ndarray] = None,
                 level: float = 0.5, smooth_sigma: float = 1.0,
                 taubin_iter: int = 25, raw: bool = False):
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
            np.asarray(mask, float), level=level, method="lewiner",
            allow_degenerate=False)
        homog = np.c_[verts, np.ones(len(verts))]
        verts = (aff @ homog.T).T[:, :3]
        return np.asarray(verts, np.float64), np.asarray(faces, np.int64)

    from spectralbrain.io.meshing import volume_to_mesh
    verts, faces = volume_to_mesh(
        np.asarray(mask), aff, raw=False, closed=False, level=level,
        field_mode="gaussian", sigma_vox=float(smooth_sigma),
        taubin_iterations=int(taubin_iter),
    )
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def render_bundle_surface(vertices: np.ndarray, faces: np.ndarray, *,
                          scalars: Optional[np.ndarray] = None,
                          cmap: Any = None, clim: Optional[tuple] = None,
                          symmetric: bool = False, robust: bool = True,
                          view: str = "oblique", bg: str = _DEFAULT_BG,
                          size: tuple[int, int] = _DEFAULT_SIZE,
                          metallic: float = 0.1, roughness: float = 0.6,
                          nan_color: str = "#BDBDBD",
                          out_path: Optional[PathLike] = None) -> Path:
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
            clim = robust_clim(s, symmetric=symmetric) if robust else (np.nanmin(s), np.nanmax(s))
        cm = cmap if cmap is not None else get_cmap("diverging" if symmetric else "sequential")
        mesh["scalars"] = s
        plotter.add_mesh(mesh, scalars="scalars", cmap=cm, clim=clim,
                         nan_color=nan_color, smooth_shading=True,
                         pbr=True, metallic=metallic, roughness=roughness,
                         show_scalar_bar=True)
    else:
        plotter.add_mesh(mesh, color="#cccccc", smooth_shading=True, pbr=True,
                         metallic=metallic, roughness=roughness)

    _set_pv_camera(plotter, V, view)
    out = Path(out_path) if out_path else Path(tempfile.mkstemp(suffix=".png")[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()
    logger.info("Saved bundle-surface render → %s", out)
    return out


def spectral_overlay(vertices: np.ndarray, faces: np.ndarray,
                     kind: str = "hks", n_eigen: int = 100,
                     t_index: int = 30, laplacian_method: str = "cotangent",
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
        from spectralbrain.spectral.descriptors import compute_hks
        desc = np.asarray(compute_hks(decomp))
    elif kind == "wks":
        from spectralbrain.spectral.descriptors import compute_wks
        desc = np.asarray(compute_wks(decomp))
    else:
        raise ValueError("kind must be 'hks' or 'wks'.")
    col = int(np.clip(t_index, 0, desc.shape[1] - 1))
    return desc[:, col]


def _set_pv_camera(plotter, vertices: np.ndarray, view: str) -> None:
    c = vertices.mean(axis=0)
    radius = float(np.linalg.norm(vertices - c, axis=1).max()) * 2.6
    preset = TRACT_VIEWS.get(view, TRACT_VIEWS["oblique"])
    az = np.deg2rad(preset["azimuth"])
    el = np.deg2rad(preset["elevation"])
    pos = c + radius * np.array([np.cos(el) * np.cos(az),
                                 np.cos(el) * np.sin(az),
                                 np.sin(el)])
    up = (0, 0, 1) if abs(preset["elevation"]) < 80 else (0, 1, 0)
    plotter.camera_position = [tuple(pos), tuple(c), up]


# ──────────────────────────────────────────────────────────────────────
# 3. Multi-POV montage & publication panels
# ──────────────────────────────────────────────────────────────────────
def streamlines_multiview(streamlines, *,
                          views: Sequence[str] = ("left", "anterior",
                                                  "superior", "oblique"),
                          color_by: str = "orientation",
                          scalars: Optional[Sequence[np.ndarray]] = None,
                          titles: Optional[Sequence[str]] = None,
                          colorbar: Optional[dict] = None,
                          tube: bool = True, bg: str = _DEFAULT_BG,
                          size: tuple[int, int] = (1200, 1200),
                          out_path: Optional[PathLike] = None,
                          random_state: int = 0):
    """Render one bundle from several canonical POVs and composite into a panel.

    This is the multi-POV tract figure: each view is rendered offscreen, then all
    are assembled into a single hybrid raster+vector matplotlib figure with an
    optional shared colorbar. Returns ``(fig, png_paths)``.
    """
    tmp = Path(tempfile.mkdtemp())
    pngs = []
    for v in views:
        p = tmp / f"tract_{v}.png"
        render_streamlines(streamlines, color_by=color_by, scalars=scalars,
                           tube=tube, view=v, bg=bg, size=size, out_path=p,
                           random_state=random_state)
        pngs.append(p)
    titles = list(titles) if titles is not None else [v.capitalize() for v in views]
    fig = compose_tract_panel(pngs, titles=titles, colorbar=colorbar,
                              out_path=out_path)
    return fig, pngs


def compose_tract_panel(images: Sequence[PathLike], *,
                        titles: Optional[Sequence[str]] = None,
                        colorbar: Optional[dict] = None,
                        ncols: Optional[int] = None,
                        panel_letters: bool = True,
                        out_path: Optional[PathLike] = None,
                        dpi: int = 300):
    """Composite rendered PNGs into a publication panel (hybrid raster+vector).

    Parameters
    ----------
    images : sequence of PNG paths
    titles : sequence of str, optional
    colorbar : dict, optional
        ``{"kind": "sequential"|"diverging", "clim": (lo, hi), "label": str}`` to
        draw a shared vector colorbar.
    ncols : int, optional
        Columns (defaults to len(images) up to 4).
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize, TwoSlopeNorm

    imgs = [mpimg.imread(str(p)) for p in images]
    n = len(imgs)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            ax.imshow(imgs[i])
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=11)
            if panel_letters:
                ax.text(0.02, 0.98, letters[i], transform=ax.transAxes,
                        fontsize=13, fontweight="bold", va="top", ha="left")
        ax.axis("off")

    if colorbar is not None:
        kind = colorbar.get("kind", "sequential")
        lo, hi = colorbar.get("clim", (0.0, 1.0))
        cm = get_cmap(kind)
        norm = (TwoSlopeNorm(0.0, lo, hi) if kind == "diverging" and lo < 0 < hi
                else Normalize(lo, hi))
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
        logger.info("Saved tract panel → %s", out_path)
    return fig


def add_glass_brain(plotter, brain_mask: np.ndarray, affine: np.ndarray, *,
                    opacity: float = 0.12, color: str = "#cccccc",
                    level: float = 0.5):
    """Add a translucent glass-brain shell to a PyVista plotter for context."""
    verts, faces = mask_to_mesh(brain_mask, affine=affine, level=level,
                                smooth_sigma=1.5, taubin_iter=15)
    pv = _require_pyvista()
    shell = pv.PolyData(verts, np.column_stack(
        [np.full(len(faces), 3), faces]).ravel())
    plotter.add_mesh(shell, color=color, opacity=opacity, smooth_shading=True)
    return plotter


__all__ = [
    "TRACT_VIEWS",
    "add_glass_brain",
    "compose_tract_panel",
    "get_cmap",
    "load_streamlines",
    "mask_to_mesh",
    "render_bundle_surface",
    "render_streamlines",
    "robust_clim",
    "spectral_overlay",
    "streamlines_multiview",
]
