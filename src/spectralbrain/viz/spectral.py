"""Spectral-field rendering -- geometric eigenmodes and spectral deformation.

Rendering counterpart of :mod:`spectralbrain.spectral.lbo` (the computation
lives there; this module only draws).

Eigenmode panels
----------------
Panel of geometric eigenmodes in the style of *Mode-Based Morphometry* (Cao et
al., 2024, HBM; Fig. 1b), but operator-agnostic: it renders the basis of ANY
SpectralBrain operator (LBO, Helmholtz, anisotropic, Hodge, polyharmonic...),
not only the MBM Helmholtz basis.

Each eigenmode ``psi_k`` is a per-vertex scalar field. Each one is rendered on
the fsLR-32k midthickness surface (Conte69) with a blue-white-red diverging map
(negative-zero-positive) and symmetric colour limits, ordered by increasing
spatial frequency (increasing lambda <=> decreasing wavelength). The final panel
is assembled at publication quality (PDF/PNG).

Two ways to use it:

1. RECOMMENDED (faithful to Cao, guaranteed mesh match): compute the modes on
   the render mesh itself with
   :func:`spectralbrain.spectral.lbo.modes.compute_geometric_eigenmodes`
   (LaPy, as in the MBM pipeline). :func:`plot_eigenmode_panel` does this when
   no eigenvectors are passed.
2. ADVANCED: pass eigenvectors already computed by SpectralBrain. The mesh of
   those eigenvectors MUST be the render mesh (Conte69 fsLR-32k); resample
   native-mesh modes first.

The SIGN of an eigenvector is arbitrary: what carries meaning are its nodal
lines (``psi_k = 0``). Signs are standardised only for visual consistency and
the colour bar is labelled in arbitrary units.

Spectral deformation
--------------------
:func:`render_scale_on_mesh` paints a per-vertex field (typically the
``log_scale`` of :func:`spectralbrain.spectral.lbo.collections.spectral_deformation`)
on any mesh, offscreen with vedo.

Heavy dependencies (yabplot, vedo) are imported lazily inside the functions;
the module imports cleanly without them.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from spectralbrain.runtime import PathLike
from spectralbrain.spectral.lbo.modes import (
    compute_geometric_eigenmodes,
    eigenmode_wavelength,
)

# Offscreen rendering on headless workstations. Must be set BEFORE any VTK use.
os.environ.setdefault("VTK_USE_OFFSCREEN", "1")

# Default views (same convention as the neuro-brainplots skill / Cao Fig. 1b).
_FOUR_VIEWS = ["left_lateral", "left_medial", "right_medial", "right_lateral"]
_ONE_VIEW = ["left_lateral"]


def _load_yabplot():
    """Lazy import of yabplot + helpers (clear error if missing)."""
    try:
        import yabplot as yab
        from yabplot.mesh import make_cortical_mesh
        from yabplot.utils import load_gii
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "yabplot is required to render eigenmodes. Install it, or use only "
            "spectralbrain.spectral.lbo.modes.compute_geometric_eigenmodes "
            "(which needs only lapy)."
        ) from exc
    return yab, make_cortical_mesh, load_gii


# ======================================================================
# S1  LOW-LEVEL UTILITIES
# ======================================================================


def standardize_sign(field: np.ndarray, method: str = "max_abs") -> np.ndarray:
    """Standardise the (arbitrary) sign of an eigenvector by visual convention."""
    f = np.asarray(field, dtype=float)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return f
    if method == "skew":
        m, s = finite.mean(), finite.std()
        sign = 1.0
        if s > 0:
            skew = np.mean(((finite - m) / s) ** 3)
            sign = -1.0 if skew < 0 else 1.0
        return sign * f
    idx = np.nanargmax(np.abs(f))
    return f if f[idx] >= 0 else -f


def _medial_mask_bool(mask: np.ndarray, n: int) -> np.ndarray:
    """Interpret the medial-wall mask as boolean (V,) or as a list of indices.

    * ``bool`` of length V -> used directly;
    * numeric of length V containing only 0/1 (e.g. ``.label.gii`` / float
      0.0/1.0) -> binary (1 = medial wall);
    * any other integer array -> indices of the medial vertices.
    """
    mask = np.asarray(mask)
    if mask.dtype == bool:
        if mask.shape[0] != n:
            raise ValueError(f"boolean mask has {mask.shape[0]} entries, the mesh has {n}.")
        return mask
    if mask.ndim == 1 and mask.shape[0] == n:
        vals = np.unique(mask[np.isfinite(mask)]) if mask.dtype.kind == "f" else np.unique(mask)
        if np.isin(vals, (0, 1)).all():
            return mask.astype(float) == 1
    idx = np.asarray(mask, dtype=float)
    if not np.all(np.isfinite(idx)) or np.any(idx != np.round(idx)):
        raise ValueError("a non-binary medial mask must contain integer indices.")
    out = np.zeros(n, dtype=bool)
    out[idx.astype(int)] = True
    return out


def _apply_medial_mask(field: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return field
    out = field.astype(float).copy()
    out[_medial_mask_bool(mask, out.shape[0])] = np.nan
    return out


def _symmetric_clim(values: np.ndarray, percentile: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    vmax = np.percentile(np.abs(v), percentile)
    return float(vmax) if vmax > 0 else float(np.max(np.abs(v)) or 1.0)


def select_mode_indices(
    evals: np.ndarray,
    n_show: int = 12,
    skip_constant: bool = True,
) -> np.ndarray:
    """Choose which eigenvector columns to render (skips the constant mode)."""
    evals = np.asarray(evals, dtype=float)
    start = 0
    if skip_constant and evals.size and evals[0] <= 1e-10:
        start = 1
    return np.arange(start, min(start + n_show, evals.size))


# ======================================================================
# S2  EIGENMODE RENDER: ONE PNG (4 OR 1 VIEWS) PER MODE
# ======================================================================


def render_eigenmodes(
    evecs_lh: np.ndarray,
    evecs_rh: np.ndarray,
    lh_surf_path: PathLike,
    rh_surf_path: PathLike,
    mode_indices: Sequence[int],
    out_dir: PathLike = "eigenmode_pngs",
    views: Sequence[str] | None = None,
    cmap: str = "RdBu_r",
    style: str = "matte",
    percentile_clim: float = 99.0,
    sign_method: str = "max_abs",
    medial_mask_lh: np.ndarray | None = None,
    medial_mask_rh: np.ndarray | None = None,
    figsize: tuple[int, int] = (1600, 400),
    zoom: float = 1.25,
) -> list:
    """Render one PNG per mode (each PNG already holding the requested views).

    Symmetric colour limits per mode (eigenmode values are arbitrary), by
    percentile.
    """
    yab, make_cortical_mesh, load_gii = _load_yabplot()
    views = list(views) if views is not None else list(_FOUR_VIEWS)
    os.makedirs(str(out_dir), exist_ok=True)

    lh_v, lh_f = load_gii(str(lh_surf_path))
    rh_v, rh_f = load_gii(str(rh_surf_path))

    png_paths = []
    for k in mode_indices:
        # Mask BEFORE sign and clim: the medial wall must define neither the
        # sign convention nor the colour scale.
        psi_lh = _apply_medial_mask(np.asarray(evecs_lh[:, k], float), medial_mask_lh)
        psi_rh = _apply_medial_mask(np.asarray(evecs_rh[:, k], float), medial_mask_rh)
        psi_lh = standardize_sign(psi_lh, method=sign_method)
        psi_rh = standardize_sign(psi_rh, method=sign_method)
        vmax = _symmetric_clim(np.concatenate([psi_lh, psi_rh]), percentile_clim)

        lh_mesh = make_cortical_mesh(lh_v, lh_f, psi_lh, scalar_name="mode")
        rh_mesh = make_cortical_mesh(rh_v, rh_f, psi_rh, scalar_name="mode")

        out_png = os.path.join(str(out_dir), f"mode_{k:03d}.png")
        yab.plot_vertexwise(
            lh_mesh,
            rh_mesh,
            scalars="mode",
            cmap=cmap,
            vminmax=[-vmax, vmax],
            views=views,
            style=style,
            figsize=figsize,
            zoom=zoom,
            export_path=out_png,
        )
        png_paths.append(out_png)
        print(f"  mode {k:>3d}  |  vmax={vmax:.3e}  ->  {out_png}")
    return png_paths


# ======================================================================
# S3  PANEL ASSEMBLY (matplotlib)
# ======================================================================


def assemble_eigenmode_panel(
    png_paths: Sequence[PathLike],
    mode_indices: Sequence[int],
    evals: np.ndarray,
    out_path: PathLike = "eigenmode_panel.pdf",
    ncols: int = 3,
    cmap: str = "RdBu_r",
    annotate: str = "wavelength",
    dpi: int = 300,
    panel_title: str | None = None,
    label_fontsize: float = 7.0,
) -> str:
    """Assemble the PNGs into a publication grid with one colour bar (a.u., -/0/+)."""
    n = len(png_paths)
    if n == 0:
        raise ValueError("No PNGs to assemble into the panel.")
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(ncols * 3.0, nrows * 1.25 + 0.6))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        wspace=0.04,
        hspace=0.18,
        left=0.01,
        right=0.90,
        top=0.94 if panel_title else 0.98,
        bottom=0.02,
    )

    for i, (png, k) in enumerate(zip(png_paths, mode_indices)):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(plt.imread(str(png)))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        lam = float(evals[k]) if k < len(evals) else np.nan
        if annotate == "wavelength":
            wl = eigenmode_wavelength(lam)
            label = (
                rf"$\psi_{{{k}}}$  $\cdot$  {wl:.0f} mm" if np.isfinite(wl) else rf"$\psi_{{{k}}}$"
            )
        elif annotate == "lambda":
            label = rf"$\psi_{{{k}}}$  $\cdot$  $\lambda$={lam:.3g}"
        elif annotate == "index":
            label = rf"$\psi_{{{k}}}$"
        else:
            label = ""
        if label:
            ax.set_title(label, fontsize=label_fontsize, pad=2)

    cax = fig.add_axes([0.915, 0.12, 0.012, 0.76])
    sm = ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["-", "0", "+"])
    cb.ax.tick_params(labelsize=label_fontsize + 1, length=0)
    cb.outline.set_visible(False)
    cb.set_label("Amplitude (a.u.)", fontsize=label_fontsize)

    if panel_title:
        fig.suptitle(panel_title, fontsize=label_fontsize + 3, y=0.99)

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"panel saved to: {out_path}")
    return str(out_path)


# ======================================================================
# S4  HIGH-LEVEL ORCHESTRATOR
# ======================================================================


def plot_eigenmode_panel(
    lh_surf_path: PathLike,
    rh_surf_path: PathLike,
    n_modes: int = 50,
    n_show: int = 12,
    ncols: int = 3,
    out_path: PathLike = "eigenmode_panel.pdf",
    out_dir: PathLike = "eigenmode_pngs",
    views_per_mode: Sequence[str] | None = None,
    evecs_lh: np.ndarray | None = None,
    evecs_rh: np.ndarray | None = None,
    evals: np.ndarray | None = None,
    cmap: str = "RdBu_r",
    annotate: str = "wavelength",
    skip_constant: bool = True,
    medial_mask_lh: np.ndarray | None = None,
    medial_mask_rh: np.ndarray | None = None,
    panel_title: str | None = None,
) -> str:
    """Full pipeline: (compute ->) render -> assemble a Cao-style panel."""
    if evecs_lh is None or evecs_rh is None or evals is None:
        evals_lh, evecs_lh, _, _ = compute_geometric_eigenmodes(lh_surf_path, n_modes)
        evals_rh, evecs_rh, _, _ = compute_geometric_eigenmodes(rh_surf_path, n_modes)
        evals = 0.5 * (evals_lh + evals_rh)
    else:
        evals = np.asarray(evals, dtype=float)

    idx = select_mode_indices(evals, n_show=n_show, skip_constant=skip_constant)
    views = list(views_per_mode) if views_per_mode is not None else list(_ONE_VIEW)

    png_paths = render_eigenmodes(
        evecs_lh,
        evecs_rh,
        lh_surf_path,
        rh_surf_path,
        mode_indices=idx,
        out_dir=out_dir,
        views=views,
        cmap=cmap,
        medial_mask_lh=medial_mask_lh,
        medial_mask_rh=medial_mask_rh,
    )
    return assemble_eigenmode_panel(
        png_paths,
        idx,
        evals,
        out_path=out_path,
        ncols=ncols,
        cmap=cmap,
        annotate=annotate,
        panel_title=panel_title,
    )


# ======================================================================
# S5  SPECTRAL DEFORMATION RENDER (vedo, offscreen)
# ======================================================================


def render_scale_on_mesh(
    V: np.ndarray,
    F: np.ndarray,
    scalar: np.ndarray,
    out_path: PathLike = "spectral_deformation.png",
    cmap: str = "RdBu_r",
    percentile_clim: float = 98.0,
    symmetric: bool = True,
    title: str | None = None,
    size: tuple[int, int] = (1200, 900),
    zoom: float = 1.3,
) -> str:
    """Paint a per-vertex scalar field (typically log omega) on the mesh.

    Blue-white-red diverging map, symmetric percentile colour limits (robust
    to outliers), offscreen vedo render. Works for any structure
    (hippocampus, subcortical, ...) -- no fsLR template required.
    """
    try:
        import vedo
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vedo is required: pip install vedo") from exc

    s = np.asarray(scalar, dtype=float)
    finite = s[np.isfinite(s)]
    vmax = np.percentile(np.abs(finite), percentile_clim) if finite.size else 1.0
    vmax = float(vmax) if vmax > 0 else 1.0
    vmin = -vmax if symmetric else float(np.percentile(finite, 100 - percentile_clim))

    mesh = vedo.Mesh([np.asarray(V, float), np.asarray(F, int)])
    mesh.cmap(cmap, s, vmin=vmin, vmax=vmax).add_scalarbar(title=title or "log omega")

    plt_ = vedo.Plotter(offscreen=True, size=size)
    plt_.show(mesh, zoom=zoom, axes=0)
    plt_.screenshot(str(out_path))
    plt_.close()
    print(f"render saved to: {out_path}")
    return str(out_path)


__all__: list[str] = [
    "assemble_eigenmode_panel",
    "plot_eigenmode_panel",
    "render_eigenmodes",
    "render_scale_on_mesh",
    "select_mode_indices",
    "standardize_sign",
]
