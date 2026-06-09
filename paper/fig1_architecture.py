"""Figure 1 — SpectralBrain architecture / schematic design."""
import matplotlib.pyplot as plt
from _figstyle import *

fig, ax = plt.subplots(figsize=(7.4, 4.7))
clean(ax)

# Title strip
ax.text(0.5, 0.965, "SpectralBrain — library architecture", ha="center",
        va="center", fontsize=12.5, weight="bold", color=INK)

# ── Input layer (top) ─────────────────────────────────────────────
box(ax, 0.06, 0.83, 0.88, 0.085,
    "Neuroimaging inputs   ·   FreeSurfer · GIfTI · NIfTI/MGZ · HippUnfold v1/v2 · TractSeg · meshes · point clouds",
    fc=TEAL_BG, ec=TEAL, tc=SLATE, fs=8.6, weight="normal")

# ── io layer ──────────────────────────────────────────────────────
box(ax, 0.06, 0.705, 0.88, 0.085,
    "spectralbrain.io   —   loaders · BIDS / SUBJECTS_DIR discovery · load_group · resample_to_template",
    fc=INDIGO_BG, ec=INDIGO, tc=SLATE, fs=8.6)

# ── core layer ────────────────────────────────────────────────────
box(ax, 0.06, 0.58, 0.88, 0.085,
    "spectralbrain (core)   —   BrainMesh · BrainPointCloud · GroupData · LBO  decompose()",
    fc=INDIGO_BG, ec=INDIGO, tc=SLATE, fs=8.8, weight="bold")

# ── analysis subpackages (3 columns) ─────────────────────────────
cy, ch = 0.305, 0.21
box(ax, 0.06, cy, 0.265, ch,
    "spectral\n\nShapeDNA · HKS · SI-HKS\nWKS · GPS · BKS / iBKS\nwavelets · functional maps\ndistances",
    fc=AMBER_BG, ec=AMBER, tc=SLATE, fs=8.4)
box(ax, 0.3675, cy, 0.265, ch,
    "statistics\n\nvertex-wise FWE / FDR / TFCE\nDeLong · BCa bootstrap\nComBat / ComBat-GAM\n6 PyMC Bayesian models",
    fc=ROSE_BG, ec=ROSE, tc=SLATE, fs=8.4)
box(ax, 0.675, cy, 0.265, ch,
    "viz\n\ntemplate-free 6-view 3D\nunfolded flat-maps\ncluster overlays\nposterior plots",
    fc=GREYBOX, ec=GREYEDGE, tc=SLATE, fs=8.4)

# ── backends (bottom, cross-cutting) ─────────────────────────────
box(ax, 0.06, 0.085, 0.88, 0.12,
    "spectralbrain.backends   (pluggable, cross-cutting)\n"
    "eigensolvers: NumPy · PyTorch · CuPy · JAX        samplers: PyMC · nutpie · NumPyro · BlackJAX",
    fc="#eef2f4", ec=GREYEDGE, tc=SLATE, fs=8.6)

# ── vertical flow arrows ─────────────────────────────────────────
cx = 0.5
for y0, y1 in [(0.83, 0.792), (0.705, 0.667), (0.58, 0.517)]:
    arrow(ax, cx, y0, cx, y1, color=SLATE, lw=1.8)
# core -> three subpackages
for tx in (0.1925, 0.5, 0.8075):
    arrow(ax, cx, 0.58, tx, 0.516, color=SLATE, lw=1.3)
# backends serve core + analysis (dashed, upward)
for tx in (0.1925, 0.5, 0.8075):
    arrow(ax, tx, 0.205, tx, 0.305, color=AMBER, lw=1.1, style="-|>", ls=(0,(4,2)))

# legend for the colour semantics
handles = [
    Line2D([0],[0], marker="s", ls="", mfc=TEAL_BG, mec=TEAL, ms=11, label="input / I/O"),
    Line2D([0],[0], marker="s", ls="", mfc=INDIGO_BG, mec=INDIGO, ms=11, label="core data model"),
    Line2D([0],[0], marker="s", ls="", mfc=AMBER_BG, mec=AMBER, ms=11, label="spectral descriptors"),
    Line2D([0],[0], marker="s", ls="", mfc=ROSE_BG, mec=ROSE, ms=11, label="statistics"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
          ncol=4, frameon=False, fontsize=7.8, handletextpad=0.4, columnspacing=1.4)

save(fig, "figures/fig1_architecture")
print("fig1 saved")
