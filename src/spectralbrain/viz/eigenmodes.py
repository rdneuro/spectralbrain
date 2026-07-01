# -*- coding: utf-8 -*-
"""
viz/eigenmodes.py  —  SpectralBrain
===================================

Painel de eigenmodes geométricos no estilo *Mode-Based Morphometry*
(Cao et al., 2024, HBM; Figura 1b), porém agnóstico ao operador: você pode
renderizar a base de QUALQUER operador da SB (LBO, Helmholtz, anisotrópico,
Hodge, polyharmonic...), não só o Helmholtz do MBM.

Cada eigenmode psi_k é um campo escalar por-vértice. Renderizamos cada um na
superfície midthickness fsLR-32k (Conte69) com um divergente azul-branco-vermelho
(negativo-zero-positivo) e clim simétrico, ordenados por frequência espacial
crescente (lambda crescente  <=>  wavelength decrescente). O painel final é
montado em qualidade de publicação (PDF/PNG 300 dpi).

Dois caminhos de uso
--------------------
1. RECOMENDADO (fiel ao Cao, mesh-match garantido): computar os modos na própria
   malha de render via LaPy — a mesma biblioteca usada no MBM e no laboratório do
   Reuter — com `compute_geometric_eigenmodes`.
2. AVANÇADO: passar autovetores já calculados pela SB. Nesse caso a malha dos
   autovetores TEM de ser a mesma do render (Conte69 fsLR-32k). Se vierem de malha
   nativa, faça o resample antes.

Filosofia "topology over magnitude": o SINAL de um autovetor é arbitrário. O que
carrega significado são as *nodal lines* (onde psi_k = 0), não se um lobo é azul
ou vermelho. Por isso padronizamos o sinal apenas por consistência visual e a
barra de cor é rotulada em unidades arbitrárias.

Dependências pesadas (yabplot, lapy) são importadas de forma PREGUIÇOSA dentro das
funções — o módulo importa limpo mesmo sem elas instaladas.

Autor: Rodrigo Debona (Velho Mago) — com Claudinho.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Render offscreen na workstation (headless). Defina ANTES de qualquer VTK.
os.environ.setdefault("VTK_USE_OFFSCREEN", "1")

# Vistas padrão (mesma convenção do skill neuro-brainplots / Cao Fig. 1b).
_FOUR_VIEWS = ["left_lateral", "left_medial", "right_medial", "right_lateral"]
_ONE_VIEW = ["left_lateral"]

PathLike = Union[str, os.PathLike]

__all__ = [
    "eigenmode_wavelength",
    "standardize_sign",
    "compute_geometric_eigenmodes",
    "render_eigenmodes",
    "select_mode_indices",
    "assemble_eigenmode_panel",
    "plot_eigenmode_panel",
]


def _load_yabplot():
    """Import preguiçoso do yabplot + helpers (erro claro se ausente)."""
    try:
        import yabplot as yab
        from yabplot.mesh import make_cortical_mesh
        from yabplot.utils import load_gii
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "yabplot é necessário para render de eigenmodes. Instale-o ou use "
            "apenas compute_geometric_eigenmodes (que precisa só de lapy)."
        ) from exc
    return yab, make_cortical_mesh, load_gii


# ─────────────────────────── utilidades de baixo nível ────────────────────────
def eigenmode_wavelength(eigenvalue: float) -> float:
    """Wavelength espacial de um modo: wavelength = 2*pi / sqrt(lambda)."""
    lam = float(eigenvalue)
    if lam <= 1e-12:
        return np.inf
    return 2.0 * np.pi / np.sqrt(lam)


def standardize_sign(field: np.ndarray, method: str = "max_abs") -> np.ndarray:
    """Padroniza o sinal (arbitrário) de um autovetor por convenção visual."""
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


def _apply_medial_mask(field: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return field
    out = field.astype(float).copy()
    mask = np.asarray(mask)
    if mask.dtype == bool:
        out[mask] = np.nan
    else:
        out[mask.astype(int)] = np.nan
    return out


def _symmetric_clim(values: np.ndarray, percentile: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    vmax = np.percentile(np.abs(v), percentile)
    return float(vmax) if vmax > 0 else float(np.max(np.abs(v)) or 1.0)


# ───────────────────── caminho 1: computar modos via LaPy ──────────────────────
def compute_geometric_eigenmodes(
    surf_path: PathLike,
    n_modes: int = 50,
    use_lumped_mass: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eigenmodes geométricos (Laplace-Beltrami) na MALHA DE RENDER, via LaPy.

    Resolve Delta psi = -lambda psi (forma fraca FEM) na superfície. Garante
    mesh-match: os modos são resolvidos na mesma malha em que serão plotados.
    """
    try:
        from lapy import TriaMesh, Solver
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "LaPy não encontrado. `pip install lapy` ou use o caminho 2 "
            "(passe autovetores prontos para render_eigenmodes)."
        ) from exc
    _, _, load_gii = _load_yabplot()

    verts, faces = load_gii(str(surf_path))
    tria = TriaMesh(verts, faces)
    solver = Solver(tria, lump=use_lumped_mass)
    evals, evecs = solver.eigs(k=int(n_modes))
    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs, dtype=float)
    order = np.argsort(evals)
    return evals[order], evecs[:, order], verts, faces


# ───────────────────── render: 1 PNG (4 ou 1 vistas) por modo ──────────────────
def render_eigenmodes(
    evecs_lh: np.ndarray,
    evecs_rh: np.ndarray,
    lh_surf_path: PathLike,
    rh_surf_path: PathLike,
    mode_indices: Sequence[int],
    out_dir: PathLike = "eigenmode_pngs",
    views: Optional[Sequence[str]] = None,
    cmap: str = "RdBu_r",
    style: str = "matte",
    percentile_clim: float = 99.0,
    sign_method: str = "max_abs",
    medial_mask_lh: Optional[np.ndarray] = None,
    medial_mask_rh: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (1600, 400),
    zoom: float = 1.25,
) -> list:
    """Renderiza um PNG por modo (cada PNG já com as vistas pedidas).

    Clim simétrico por modo (valores de eigenmode são arbitrários), por percentil.
    """
    yab, make_cortical_mesh, load_gii = _load_yabplot()
    views = list(views) if views is not None else list(_FOUR_VIEWS)
    os.makedirs(str(out_dir), exist_ok=True)

    lh_v, lh_f = load_gii(str(lh_surf_path))
    rh_v, rh_f = load_gii(str(rh_surf_path))

    png_paths = []
    for k in mode_indices:
        psi_lh = standardize_sign(evecs_lh[:, k], method=sign_method)
        psi_rh = standardize_sign(evecs_rh[:, k], method=sign_method)
        vmax = _symmetric_clim(np.concatenate([psi_lh, psi_rh]), percentile_clim)
        psi_lh = _apply_medial_mask(psi_lh, medial_mask_lh)
        psi_rh = _apply_medial_mask(psi_rh, medial_mask_rh)

        lh_mesh = make_cortical_mesh(lh_v, lh_f, psi_lh, scalar_name="mode")
        rh_mesh = make_cortical_mesh(rh_v, rh_f, psi_rh, scalar_name="mode")

        out_png = os.path.join(str(out_dir), f"mode_{k:03d}.png")
        yab.plot_vertexwise(
            lh_mesh, rh_mesh, scalars="mode",
            cmap=cmap, vminmax=[-vmax, vmax],
            views=views, style=style,
            figsize=figsize, zoom=zoom, export_path=out_png,
        )
        png_paths.append(out_png)
        print(f"  modo {k:>3d}  |  vmax={vmax:.3e}  ->  {out_png}")
    return png_paths


def select_mode_indices(
    evals: np.ndarray, n_show: int = 12, skip_constant: bool = True,
) -> np.ndarray:
    """Escolhe quais colunas de autovetores renderizar (pula o modo constante)."""
    evals = np.asarray(evals, dtype=float)
    start = 0
    if skip_constant and evals.size and evals[0] <= 1e-10:
        start = 1
    return np.arange(start, min(start + n_show, evals.size))


# ───────────────────── montagem do painel (matplotlib) ─────────────────────────
def assemble_eigenmode_panel(
    png_paths: Sequence[PathLike],
    mode_indices: Sequence[int],
    evals: np.ndarray,
    out_path: PathLike = "eigenmode_panel.pdf",
    ncols: int = 3,
    cmap: str = "RdBu_r",
    annotate: str = "wavelength",
    dpi: int = 300,
    panel_title: Optional[str] = None,
    label_fontsize: float = 7.0,
) -> str:
    """Monta os PNGs numa grade de publicação, com colorbar única (a.u., −/0/+)."""
    n = len(png_paths)
    if n == 0:
        raise ValueError("Nenhum PNG para montar o painel.")
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(ncols * 3.0, nrows * 1.25 + 0.6))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, wspace=0.04, hspace=0.18,
        left=0.01, right=0.90, top=0.94 if panel_title else 0.98, bottom=0.02,
    )

    for i, (png, k) in enumerate(zip(png_paths, mode_indices)):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(plt.imread(str(png)))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        lam = float(evals[k]) if k < len(evals) else np.nan
        if annotate == "wavelength":
            wl = eigenmode_wavelength(lam)
            label = rf"$\psi_{{{k}}}$  ·  {wl:.0f} mm" if np.isfinite(wl) else rf"$\psi_{{{k}}}$"
        elif annotate == "lambda":
            label = rf"$\psi_{{{k}}}$  ·  $\lambda$={lam:.3g}"
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
    cb.ax.set_yticklabels(["−", "0", "+"])
    cb.ax.tick_params(labelsize=label_fontsize + 1, length=0)
    cb.outline.set_visible(False)
    cb.set_label("amplitude (a.u.)", fontsize=label_fontsize)

    if panel_title:
        fig.suptitle(panel_title, fontsize=label_fontsize + 3, y=0.99)

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"painel salvo em: {out_path}")
    return str(out_path)


# ───────────────────────────── orquestrador high-level ─────────────────────────
def plot_eigenmode_panel(
    lh_surf_path: PathLike,
    rh_surf_path: PathLike,
    n_modes: int = 50,
    n_show: int = 12,
    ncols: int = 3,
    out_path: PathLike = "eigenmode_panel.pdf",
    out_dir: PathLike = "eigenmode_pngs",
    views_per_mode: Optional[Sequence[str]] = None,
    evecs_lh: Optional[np.ndarray] = None,
    evecs_rh: Optional[np.ndarray] = None,
    evals: Optional[np.ndarray] = None,
    cmap: str = "RdBu_r",
    annotate: str = "wavelength",
    skip_constant: bool = True,
    medial_mask_lh: Optional[np.ndarray] = None,
    medial_mask_rh: Optional[np.ndarray] = None,
    panel_title: Optional[str] = None,
) -> str:
    """Pipeline completo: (computa ->) renderiza -> monta painel estilo Cao."""
    if evecs_lh is None or evecs_rh is None or evals is None:
        evals_lh, evecs_lh, _, _ = compute_geometric_eigenmodes(lh_surf_path, n_modes)
        evals_rh, evecs_rh, _, _ = compute_geometric_eigenmodes(rh_surf_path, n_modes)
        evals = 0.5 * (evals_lh + evals_rh)
    else:
        evals = np.asarray(evals, dtype=float)

    idx = select_mode_indices(evals, n_show=n_show, skip_constant=skip_constant)
    views = list(views_per_mode) if views_per_mode is not None else list(_ONE_VIEW)

    png_paths = render_eigenmodes(
        evecs_lh, evecs_rh, lh_surf_path, rh_surf_path,
        mode_indices=idx, out_dir=out_dir, views=views, cmap=cmap,
        medial_mask_lh=medial_mask_lh, medial_mask_rh=medial_mask_rh,
    )
    return assemble_eigenmode_panel(
        png_paths, idx, evals, out_path=out_path, ncols=ncols,
        cmap=cmap, annotate=annotate, panel_title=panel_title,
    )


# ─────────────────────────────── exemplo de uso ───────────────────────────────
if __name__ == "__main__":
    import yabplot as yab
    try:
        LH, RH = yab.data.get_surface_paths("midthickness", "bmesh")
    except Exception:
        LH = "conte69.L.midthickness.surf.gii"
        RH = "conte69.R.midthickness.surf.gii"

    plot_eigenmode_panel(
        lh_surf_path=LH, rh_surf_path=RH,
        n_modes=50, n_show=12, ncols=3,
        views_per_mode=["left_lateral"],
        out_path="eigenmode_panel_LBO.pdf",
        annotate="wavelength",
        panel_title="Geometric eigenmodes (LBO) — fsLR-32k",
    )
