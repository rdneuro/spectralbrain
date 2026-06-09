"""Figure 4 — Statistical processing: frequentist + Bayesian tracks."""
import matplotlib.pyplot as plt
from _figstyle import *

def titled(ax, x, y, w, h, title, body, *, fc, ec, tc=SLATE, t_fs=8.6, b_fs=7.5):
    box(ax, x, y, w, h, "", fc=fc, ec=ec)
    ax.text(x+w/2, y+h-0.030, title, ha="center", va="center", fontsize=t_fs,
            weight="bold", color=tc, zorder=3)
    ax.text(x+w/2, y+(h-0.05)/2, body, ha="center", va="center", fontsize=b_fs,
            color=tc, zorder=3, linespacing=1.4)

fig, ax = plt.subplots(figsize=(8.4, 5.0))
clean(ax)
ax.text(0.5, 0.965, "Statistical processing", ha="center", va="center",
        fontsize=12.5, weight="bold", color=INK)

# ── input: descriptor fields ─────────────────────────────────────
box(ax, 0.30, 0.85, 0.40, 0.085,
    "descriptor fields  ·  group tensors  (from Fig. 3)",
    fc=AMBER_BG, ec=AMBER, tc=SLATE, fs=8.6, weight="bold")

# optional harmonization gate
box(ax, 0.345, 0.745, 0.31, 0.062,
    "optional harmonization — ComBat / ComBat-GAM",
    fc="#eef2f4", ec=GREYEDGE, tc=SLATE, fs=7.8)
arrow(ax, 0.5, 0.85, 0.5, 0.808, color=SLATE, lw=1.6)

# ── two tracks ───────────────────────────────────────────────────
header(ax, 0.045, 0.685, "Frequentist", color=ROSE, fs=10)
header(ax, 0.56, 0.685, "Bayesian  (PyMC)", color=INDIGO, fs=10)

# frequentist boxes
titled(ax, 0.045, 0.40, 0.43, 0.245, "vertex-wise inference",
       "permutation FWE (max-statistic)\nFDR  ·  TFCE  ·  Welch / Mann–Whitney\npartial correlation (correct d.f.)",
       fc=ROSE_BG, ec=ROSE, b_fs=7.6)
titled(ax, 0.045, 0.18, 0.43, 0.18, "effect size & comparison",
       "Cohen's d maps · analytic DeLong AUC\nBCa bootstrap CIs · ICC · non-inferiority",
       fc=ROSE_BG, ec=ROSE, b_fs=7.6)

# bayesian boxes
titled(ax, 0.525, 0.40, 0.43, 0.245, "regression & group models",
       "Horseshoe (sparse) regression\nBEST group comparison\nhierarchical linear model",
       fc=INDIGO_BG, ec=INDIGO, b_fs=7.6)
titled(ax, 0.525, 0.18, 0.43, 0.18, "normative & spatial models",
       "Gaussian-process normative (z-scores)\nspatial · connectome · cluster-confirmation",
       fc=INDIGO_BG, ec=INDIGO, b_fs=7.6)

# arrows from input to both tracks
arrow(ax, 0.42, 0.745, 0.26, 0.648, color=ROSE, lw=1.5)
arrow(ax, 0.58, 0.745, 0.74, 0.648, color=INDIGO, lw=1.5)
arrow(ax, 0.26, 0.40, 0.26, 0.362, color=ROSE, lw=1.4)
arrow(ax, 0.74, 0.40, 0.74, 0.362, color=INDIGO, lw=1.4)

# samplers ribbon (bayesian)
ax.text(0.74, 0.155, "samplers: PyMC · nutpie · NumPyro · BlackJAX",
        ha="center", va="top", fontsize=7.0, color=SLATE, style="italic")

# ── outputs (bottom) ─────────────────────────────────────────────
arrow(ax, 0.26, 0.18, 0.40, 0.105, color=ROSE, lw=1.5)
arrow(ax, 0.74, 0.18, 0.60, 0.105, color=INDIGO, lw=1.5)
box(ax, 0.27, 0.018, 0.46, 0.085,
    "outputs → significance maps · posteriors · normative z-scores → viz",
    fc=GREYBOX, ec=GREYEDGE, tc=SLATE, fs=8.2, weight="bold")

save(fig, "figures/fig4_statistics")
print("fig4 saved")
