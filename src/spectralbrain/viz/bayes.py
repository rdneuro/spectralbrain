"""Bayesian posterior visualisation for spectral morphometry.

Directly linked to :mod:`spectralbrain.statistics.bayesian` — every
model class has a matching figure function.  Also includes general
Bayesian visualisation tools (ridgeline plots, ROPE decision diagrams,
forest plots, prior–posterior overlays).

Figure types
------------
1. Posterior distribution with HDI + ROPE (general)
2. Forest plot (coefficients + credible intervals)
3. Prior vs posterior overlay
4. ROPE decision diagram (stacked bar)
5. Ridgeline plot (multi-group, multi-panel)
6. Horseshoe shrinkage path (HorseshoeRegression)
7. BEST effect size posterior (BayesianGroupComparison)
8. Site effects caterpillar (HierarchicalLinearModel)
9. GP normative trajectory (GaussianProcessNormative)
10. Connectome difference matrix (BayesianConnectome)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

DPI: int = 600

# ── SpectralBrain palette subset for Bayesian plots ───────────────────

_BLUE = "#4477AA"
_RED = "#EE6677"
_GREEN = "#228833"
_PURPLE = "#AA3377"
_ORANGE = "#EE8866"
_TEAL = "#44AA99"
_GREY = "#BBBBBB"
_DARK = "#222222"
_CYAN = "#66CCEE"
_INDIGO = "#332288"

_PALETTE = [_BLUE, _RED, _GREEN, _PURPLE, _ORANGE,
            _TEAL, _CYAN, _INDIGO]

_ROPE_BELOW = "#4477AA"
_ROPE_INSIDE = "#BBBBBB"
_ROPE_ABOVE = "#EE6677"


def _apply_style():
    """Apply the publication-quality matplotlib style preset."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
        plt.rcParams["mathtext.fontset"] = "cm"
    except ImportError:
        pass
    plt.rcParams.update({"savefig.dpi": DPI, "figure.dpi": DPI})


def _save(fig, path, formats=None):
    """Save the current figure if a path is provided."""
    from spectralbrain.viz.graphics import savefig
    return savefig(fig, path, formats=formats, dpi=DPI)


# ======================================================================
# §1  POSTERIOR DISTRIBUTION + HDI + ROPE
# ======================================================================

def plot_posterior(
    samples: np.ndarray,
    *,
    hdi_prob: float = 0.94,
    rope: Optional[Tuple[float, float]] = None,
    ref_val: Optional[float] = None,
    color: str = _BLUE,
    xlabel: str = "Parameter",
    title: str = "",
    ax: Optional[Axes] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Posterior distribution with HDI band and optional ROPE.

    Parameters
    ----------
    samples : ndarray, shape (n,)
    hdi_prob : float
        Highest Density Interval probability mass.
    rope : (lo, hi), optional
        Region Of Practical Equivalence — shaded in grey.
    ref_val : float, optional
        Reference value (vertical dashed line, e.g. 0).
    """
    _apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=DPI)
    else:
        fig = ax.figure

    s = samples.ravel()
    kde = gaussian_kde(s)
    x = np.linspace(s.min() - s.std() * 0.5, s.max() + s.std() * 0.5, 500)
    y = kde(x)

    # Fill posterior.
    ax.fill_between(x, y, alpha=0.25, color=color)
    ax.plot(x, y, color=color, lw=1.8)

    # HDI.
    lo, hi = _hdi(s, hdi_prob)
    mask = (x >= lo) & (x <= hi)
    ax.fill_between(x[mask], y[mask], alpha=0.4, color=color,
                    label=f"{hdi_prob*100:.0f}% HDI [{lo:.3f}, {hi:.3f}]")
    for v in [lo, hi]:
        ax.axvline(v, color=color, ls=":", lw=0.8, alpha=0.6)

    # ROPE.
    if rope is not None:
        ax.axvspan(rope[0], rope[1], alpha=0.12, color=_GREY, zorder=0,
                   label=f"ROPE [{rope[0]:.2f}, {rope[1]:.2f}]")

    # Reference value.
    if ref_val is not None:
        ax.axvline(ref_val, color=_DARK, ls="--", lw=1, alpha=0.7,
                   label=f"ref = {ref_val}")

    # Posterior mean.
    mean_val = s.mean()
    ax.axvline(mean_val, color=color, ls="-", lw=1.2, alpha=0.5)
    ax.annotate(f"mean = {mean_val:.3f}", xy=(mean_val, y.max() * 0.95),
                fontsize=7, ha="center", color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")

    if save:
        _save(fig, save)
    return fig, ax


def _hdi(samples: np.ndarray, prob: float) -> Tuple[float, float]:
    """Highest Density Interval (narrowest interval containing prob mass)."""
    s = np.sort(samples)
    n = len(s)
    interval_width = int(np.ceil(prob * n))
    widths = s[interval_width:] - s[:n - interval_width]
    best = widths.argmin()
    return float(s[best]), float(s[best + interval_width])


# ======================================================================
# §2  FOREST PLOT
# ======================================================================

def plot_forest(
    var_names: List[str],
    posteriors: List[np.ndarray],
    *,
    hdi_prob: float = 0.94,
    ref_val: float = 0.0,
    colors: Optional[List[str]] = None,
    title: str = "Forest Plot",
    xlabel: str = "Coefficient",
    ax: Optional[Axes] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Forest plot — coefficients + credible intervals.

    Parameters
    ----------
    var_names : list of str
        Names of the variables / parameters.
    posteriors : list of ndarray
        One posterior sample array per parameter.
    hdi_prob : float
    ref_val : float
        Reference line (typically 0).
    """
    _apply_style()
    n = len(var_names)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 0.35 * n + 1), dpi=DPI)
    else:
        fig = ax.figure

    cols = colors or [_BLUE] * n
    positions = np.arange(n)

    for i, (name, post) in enumerate(zip(var_names, posteriors)):
        s = post.ravel()
        lo, hi = _hdi(s, hdi_prob)
        lo50, hi50 = _hdi(s, 0.50)
        mean = s.mean()

        c = cols[i % len(cols)]
        # Thin line: full HDI.
        ax.plot([lo, hi], [i, i], color=c, lw=1.2, solid_capstyle="round")
        # Thick line: 50% HDI.
        ax.plot([lo50, hi50], [i, i], color=c, lw=3.5, solid_capstyle="round",
                alpha=0.7)
        # Dot: posterior mean.
        ax.plot(mean, i, "o", color=c, markersize=5, zorder=5)

    ax.axvline(ref_val, color=_DARK, ls="--", lw=0.8, alpha=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(var_names, fontsize=7)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()

    if save:
        _save(fig, save)
    return fig, ax


# ======================================================================
# §3  PRIOR vs POSTERIOR
# ======================================================================

def plot_prior_posterior(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    *,
    xlabel: str = "Parameter",
    title: str = "Prior → Posterior",
    ax: Optional[Axes] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Overlay prior and posterior distributions."""
    _apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=DPI)
    else:
        fig = ax.figure

    for samples, label, color, alpha in [
        (prior_samples, "Prior", _GREY, 0.5),
        (posterior_samples, "Posterior", _BLUE, 0.8),
    ]:
        s = samples.ravel()
        kde = gaussian_kde(s)
        x = np.linspace(s.min() - s.std(), s.max() + s.std(), 300)
        ax.fill_between(x, kde(x), alpha=alpha * 0.3, color=color)
        ax.plot(x, kde(x), color=color, lw=1.5, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title, fontweight="bold")

    if save:
        _save(fig, save)
    return fig, ax


# ======================================================================
# §4  ROPE DECISION DIAGRAM
# ======================================================================

def plot_rope_decision(
    posteriors: Dict[str, np.ndarray],
    rope: Tuple[float, float] = (-0.1, 0.1),
    *,
    title: str = "ROPE Decision",
    xlabel: str = "Probability",
    ax: Optional[Axes] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Stacked horizontal bar — P(below) | P(ROPE) | P(above).

    Parameters
    ----------
    posteriors : dict of {label: samples}
    rope : (lo, hi)
    """
    _apply_style()
    names = list(posteriors.keys())
    n = len(names)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 0.4 * n + 0.8), dpi=DPI)
    else:
        fig = ax.figure

    lo, hi = rope
    y_pos = np.arange(n)

    for i, (name, samples) in enumerate(posteriors.items()):
        s = samples.ravel()
        p_below = (s < lo).mean()
        p_rope = ((s >= lo) & (s <= hi)).mean()
        p_above = (s > hi).mean()

        ax.barh(i, p_below, height=0.6, color=_ROPE_BELOW, edgecolor="white", lw=0.5)
        ax.barh(i, p_rope, left=p_below, height=0.6, color=_ROPE_INSIDE, edgecolor="white", lw=0.5)
        ax.barh(i, p_above, left=p_below + p_rope, height=0.6, color=_ROPE_ABOVE, edgecolor="white", lw=0.5)

        # Annotate probabilities.
        for p, x_start, col in [
            (p_below, p_below / 2, _ROPE_BELOW),
            (p_rope, p_below + p_rope / 2, _DARK),
            (p_above, p_below + p_rope + p_above / 2, _ROPE_ABOVE),
        ]:
            if p > 0.05:
                ax.text(x_start, i, f"{p:.0%}", ha="center", va="center",
                        fontsize=6, fontweight="bold",
                        color="white" if col != _DARK else _DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color=_ROPE_BELOW, label=f"P(< {lo:.2f})"),
        mpatches.Patch(color=_ROPE_INSIDE, label=f"P(ROPE)"),
        mpatches.Patch(color=_ROPE_ABOVE, label=f"P(> {hi:.2f})"),
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc="lower right",
              ncol=3, frameon=False)

    if title:
        ax.set_title(title, fontweight="bold")

    if save:
        _save(fig, save)
    return fig, ax


# ======================================================================
# §5  RIDGELINE PLOT (the showpiece)
# ======================================================================

def plot_ridgeline(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    overlap: float = 0.6,
    colors: Optional[List[str]] = None,
    xlabel: str = "Value",
    title: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[PathLike] = None,
    formats: Optional[Union[str, List[str]]] = None,
) -> Tuple[Figure, List[Axes]]:
    """Multi-panel ridgeline plot.

    Each panel corresponds to one feature.  Within each panel,
    overlapping KDE distributions are stacked vertically by group.

    Parameters
    ----------
    data : dict of {feature_name: {group_name: samples}}
        Outer dict = panels (features).
        Inner dict = ridges within each panel (groups).
    overlap : float
        Vertical overlap between ridges (0 = no overlap, 1 = full).
    colors : list of str, optional
        Colours per group.
    xlabel : str
    title : str
    figsize : (w, h), optional

    Returns
    -------
    fig, axes

    Examples
    --------
    >>> plot_ridgeline({
    ...     "HKS": {"Control": ctrl_hks, "MTLE-L": mtle_l_hks, "MTLE-R": mtle_r_hks},
    ...     "WKS": {"Control": ctrl_wks, "MTLE-L": mtle_l_wks, "MTLE-R": mtle_r_wks},
    ... }, overlap=0.7, save="ridgeline.png")
    """
    _apply_style()
    features = list(data.keys())
    n_panels = len(features)

    # Discover group names (consistent across panels).
    all_groups = []
    for feat_data in data.values():
        for g in feat_data:
            if g not in all_groups:
                all_groups.append(g)
    n_groups = len(all_groups)
    cols = colors or _PALETTE[:n_groups]

    if figsize is None:
        figsize = (4 * n_panels, 0.6 * n_groups + 1.5)

    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=figsize, dpi=DPI, sharey=True,
    )
    if n_panels == 1:
        axes_arr = [axes_arr]

    import seaborn as sns

    for panel_idx, (feat_name, feat_data) in enumerate(data.items()):
        ax = axes_arr[panel_idx]

        # Global x range for this panel.
        all_vals = np.concatenate([
            feat_data[g].ravel() for g in all_groups if g in feat_data
        ])
        x_lo = np.percentile(all_vals, 1) - np.std(all_vals) * 0.3
        x_hi = np.percentile(all_vals, 99) + np.std(all_vals) * 0.3
        x = np.linspace(x_lo, x_hi, 300)

        # Compute max density across all groups for normalisation.
        max_density = 0
        kdes = {}
        for g in all_groups:
            if g in feat_data:
                s = feat_data[g].ravel()
                kdes[g] = gaussian_kde(s)(x)
                max_density = max(max_density, kdes[g].max())
            else:
                kdes[g] = np.zeros_like(x)

        # Plot ridges bottom-to-top (last group on top).
        for i, g in enumerate(reversed(all_groups)):
            y_base = i * (1 - overlap)
            y_kde = kdes[g] / (max_density + 1e-30) * 0.9
            color = cols[(n_groups - 1 - i) % len(cols)]

            # Gradient fill.
            ax.fill_between(x, y_base, y_base + y_kde,
                            alpha=0.55, color=color, zorder=n_groups - i)
            ax.plot(x, y_base + y_kde,
                    color=color, lw=1.2, zorder=n_groups - i + 1)

            # Baseline.
            ax.axhline(y_base, color=color, lw=0.3, alpha=0.3,
                       zorder=n_groups - i - 1)

            # Group label on left.
            if panel_idx == 0:
                ax.text(x_lo - (x_hi - x_lo) * 0.02,
                        y_base + 0.15, g,
                        ha="right", va="bottom", fontsize=7,
                        fontweight="bold", color=color)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.1, n_groups * (1 - overlap) + 0.5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(feat_name, fontsize=9, fontweight="bold")
        ax.set_yticks([])
        sns.despine(ax=ax, left=True)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        _save(fig, save, formats=formats)
    return fig, axes_arr


# ======================================================================
# §6  HORSESHOE SHRINKAGE (HorseshoeRegression)
# ======================================================================

def plot_horseshoe_coefficients(
    trace,
    *,
    var_names: Optional[List[str]] = None,
    hdi_prob: float = 0.94,
    title: str = "Horseshoe Regression — Feature Selection",
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Horseshoe coefficient plot: forest + shrinkage heatmap.

    Left panel: forest plot of β posteriors.
    Right panel: local shrinkage (κ = 1/(1+λ²)) — darker = more shrunk.

    Parameters
    ----------
    trace : arviz.InferenceData
        From ``HorseshoeRegression.trace_``.
    var_names : list of str, optional
        Names for each feature / coefficient.
    """
    _apply_style()

    beta = trace.posterior["beta"].values
    lam = trace.posterior["lambda"].values

    # Flatten chains.
    beta = beta.reshape(-1, beta.shape[-1])
    lam = lam.reshape(-1, lam.shape[-1])
    d = beta.shape[1]

    if var_names is None:
        var_names = [f"β_{i}" for i in range(d)]

    # Shrinkage factor: κ = 1/(1 + λ²)
    kappa = 1.0 / (1.0 + lam ** 2)
    kappa_mean = kappa.mean(axis=0)

    fig, (ax_forest, ax_shrink) = plt.subplots(
        1, 2, figsize=(8, 0.35 * d + 1.2), dpi=DPI,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
    )

    # Forest plot.
    posteriors = [beta[:, i] for i in range(d)]
    # Color by shrinkage: more shrunk = grey, less = blue.
    colors = []
    for k in kappa_mean:
        if k > 0.7:
            colors.append(_GREY)
        elif k > 0.3:
            colors.append(_ORANGE)
        else:
            colors.append(_BLUE)

    plot_forest(
        var_names, posteriors,
        hdi_prob=hdi_prob, ref_val=0.0,
        colors=colors, title="", xlabel="β",
        ax=ax_forest,
    )
    ax_forest.set_title("Coefficient posteriors", fontsize=9)

    # Shrinkage heatmap.
    kappa_img = kappa_mean[:, None]
    ax_shrink.imshow(
        kappa_img, cmap="Greys", aspect="auto",
        vmin=0, vmax=1, interpolation="nearest",
    )
    ax_shrink.set_xticks([0])
    ax_shrink.set_xticklabels(["κ"], fontsize=7)
    ax_shrink.set_yticks(range(d))
    ax_shrink.set_yticklabels(["" for _ in range(d)])
    ax_shrink.set_title("Shrinkage", fontsize=8)

    # Annotate κ values.
    for i, k in enumerate(kappa_mean):
        ax_shrink.text(0, i, f"{k:.2f}", ha="center", va="center",
                       fontsize=6, color="white" if k > 0.5 else _DARK)

    fig.suptitle(title, fontsize=10, fontweight="bold")

    if save:
        _save(fig, save)
    return fig, (ax_forest, ax_shrink)


# ======================================================================
# §7  BEST EFFECT SIZE (BayesianGroupComparison)
# ======================================================================

def plot_best_posterior(
    trace,
    *,
    rope: Tuple[float, float] = (-0.1, 0.1),
    title: str = "BEST — Bayesian Group Comparison",
    save: Optional[PathLike] = None,
) -> Tuple[Figure, List[Axes]]:
    """Three-panel BEST visualisation: Δμ, Δσ, effect size.

    Parameters
    ----------
    trace : arviz.InferenceData
        From ``BayesianGroupComparison.trace_``.
    rope : (lo, hi)
    """
    _apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=DPI)

    panels = [
        ("diff_means", "Δμ (A − B)", _BLUE),
        ("diff_stds", "Δσ (A − B)", _TEAL),
        ("effect_size", "Effect size (Cohen's d)", _PURPLE),
    ]

    for ax, (var, label, color) in zip(axes, panels):
        samples = trace.posterior[var].values.ravel()
        plot_posterior(
            samples, hdi_prob=0.94,
            rope=rope if var == "effect_size" else None,
            ref_val=0.0, color=color,
            xlabel=label, ax=ax,
        )

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()

    if save:
        _save(fig, save)
    return fig, list(axes)


# ======================================================================
# §8  SITE EFFECTS (HierarchicalLinearModel)
# ======================================================================

def plot_site_effects(
    trace,
    *,
    site_names: Optional[List[str]] = None,
    title: str = "Hierarchical Model — Site Random Effects",
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Caterpillar plot of random intercepts per site.

    Parameters
    ----------
    trace : arviz.InferenceData
        From ``HierarchicalLinearModel.trace_``.
    site_names : list of str, optional
    """
    u = trace.posterior["u_site"].values
    u = u.reshape(-1, u.shape[-1])
    n_sites = u.shape[1]

    if site_names is None:
        site_names = [f"Site {i}" for i in range(n_sites)]

    posteriors = [u[:, i] for i in range(n_sites)]

    # Color by deviation from zero.
    means = np.array([p.mean() for p in posteriors])
    max_abs = np.abs(means).max() + 1e-10
    colors = []
    for m in means:
        intensity = np.abs(m) / max_abs
        if m > 0:
            colors.append(_RED if intensity > 0.3 else _ORANGE)
        else:
            colors.append(_BLUE if intensity > 0.3 else _CYAN)

    fig, ax = plot_forest(
        site_names, posteriors, hdi_prob=0.94, ref_val=0.0,
        colors=colors, title=title, xlabel="Random intercept",
    )

    if save:
        _save(fig, save)
    return fig, ax


# ======================================================================
# §9  GP NORMATIVE TRAJECTORY (GaussianProcessNormative)
# ======================================================================

def plot_gp_trajectory(
    ages_train: np.ndarray,
    y_train: np.ndarray,
    ages_pred: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    *,
    patient_ages: Optional[np.ndarray] = None,
    patient_values: Optional[np.ndarray] = None,
    patient_labels: Optional[List[str]] = None,
    ci_levels: Tuple[float, ...] = (0.5, 0.9, 0.99),
    title: str = "GP Normative Trajectory",
    xlabel: str = "Age (years)",
    ylabel: str = "Descriptor",
    ax: Optional[Axes] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Gaussian Process age trajectory with uncertainty fans.

    Concentric bands show expanding uncertainty.  Individual
    patients plotted as coloured dots with deviation annotation.

    Parameters
    ----------
    ages_train : ndarray, shape (n,)
    y_train : ndarray, shape (n,)
    ages_pred : ndarray, shape (m,)
    y_pred_mean : ndarray, shape (m,)
    y_pred_std : ndarray, shape (m,)
    patient_ages, patient_values : ndarray, optional
    patient_labels : list of str, optional
    ci_levels : tuple of float
        Confidence bands (inner to outer).
    """
    _apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=DPI)
    else:
        fig = ax.figure

    from scipy.stats import norm

    # Reference cohort.
    ax.scatter(ages_train, y_train, s=8, alpha=0.25, color=_GREY,
               rasterized=True, zorder=1, label="Reference cohort")

    # GP mean.
    ax.plot(ages_pred, y_pred_mean, color=_BLUE, lw=2, zorder=3,
            label="GP mean")

    # Confidence fans (outer to inner for correct layering).
    alphas = np.linspace(0.08, 0.25, len(ci_levels))
    for ci, alpha in zip(reversed(sorted(ci_levels)), alphas):
        z = norm.ppf(0.5 + ci / 2)
        lo = y_pred_mean - z * y_pred_std
        hi = y_pred_mean + z * y_pred_std
        ax.fill_between(ages_pred, lo, hi, alpha=alpha, color=_BLUE,
                        zorder=2, label=f"{ci*100:.0f}% CI" if ci == max(ci_levels) else "")

    # Patients.
    if patient_ages is not None and patient_values is not None:
        pat_colors = [_RED, _ORANGE, _PURPLE, _GREEN, _TEAL]
        for i, (pa, pv) in enumerate(zip(patient_ages, patient_values)):
            c = pat_colors[i % len(pat_colors)]
            label = patient_labels[i] if patient_labels else f"Patient {i+1}"

            # Compute z-score at this age.
            idx = np.argmin(np.abs(ages_pred - pa))
            z_score = (pv - y_pred_mean[idx]) / (y_pred_std[idx] + 1e-10)

            ax.scatter(pa, pv, s=60, color=c, edgecolors="white",
                       linewidths=0.8, zorder=5)
            ax.annotate(
                f"{label}\nz={z_score:.1f}",
                xy=(pa, pv), xytext=(8, 8),
                textcoords="offset points", fontsize=6,
                color=c, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=c, lw=0.5),
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", frameon=False)

    if save:
        _save(fig, save)
    return fig, ax


# ======================================================================
# §10  CONNECTOME DIFFERENCE (BayesianConnectome)
# ======================================================================

def plot_connectome_posterior(
    edge_diff_matrix: np.ndarray,
    *,
    labels: Optional[List[str]] = None,
    network_boundaries: Optional[List[int]] = None,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
    title: str = "Bayesian Connectome — Edge Differences",
    ax: Optional[Axes] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Posterior mean edge-difference matrix.

    Parameters
    ----------
    edge_diff_matrix : ndarray, shape (R, R)
        From ``BayesianConnectome.edge_difference_matrix()``.
    """
    _apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    else:
        fig = ax.figure

    if vmax is None:
        vmax = np.abs(edge_diff_matrix).max()

    im = ax.imshow(edge_diff_matrix, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Posterior Δ (A − B)")

    if network_boundaries:
        for b in network_boundaries:
            ax.axhline(b - 0.5, color="white", lw=1)
            ax.axvline(b - 0.5, color="white", lw=1)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)

    if title:
        ax.set_title(title, fontweight="bold")

    if save:
        _save(fig, save)
    return fig, ax


# ======================================================================

__all__ = [
    # General Bayesian
    "plot_posterior",
    "plot_forest",
    "plot_prior_posterior",
    "plot_rope_decision",
    "plot_ridgeline",
    # Model-specific
    "plot_horseshoe_coefficients",
    "plot_best_posterior",
    "plot_site_effects",
    "plot_gp_trajectory",
    "plot_connectome_posterior",
]
