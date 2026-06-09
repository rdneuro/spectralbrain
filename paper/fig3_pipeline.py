"""Figure 3 — Processing pipeline and embedded analyses."""
import matplotlib.pyplot as plt
from _figstyle import *

def titled(ax, x, y, w, h, title, body, *, fc, ec, tc=SLATE, t_fs=8.6, b_fs=7.6):
    box(ax, x, y, w, h, "", fc=fc, ec=ec)
    ax.text(x+w/2, y+h-0.032, title, ha="center", va="center", fontsize=t_fs,
            weight="bold", color=tc, zorder=3)
    ax.text(x+w/2, y+(h-0.055)/2, body, ha="center", va="center", fontsize=b_fs,
            color=tc, zorder=3, linespacing=1.4)

fig, ax = plt.subplots(figsize=(8.4, 4.6))
clean(ax)
ax.text(0.5, 0.965, "Processing pipeline and embedded analyses",
        ha="center", va="center", fontsize=12.5, weight="bold", color=INK)

ytop = 0.55; H = 0.30   # main flow band
# 1) data model
titled(ax, 0.015, ytop, 0.155, H, "surface /\npoint cloud",
       "BrainMesh\nBrainPointCloud", fc=INDIGO_BG, ec=INDIGO, t_fs=8.2)
# 2) Laplacian + eigensolve
titled(ax, 0.205, ytop, 0.175, H, "LBO operator",
       "cotangent /\nrobust Laplacian\n→ decompose(k)\neigenpairs (λ, φ)", fc=INDIGO_BG, ec=INDIGO, t_fs=8.2)
# 3) descriptors
titled(ax, 0.415, ytop, 0.205, H, "spectral descriptors",
       "ShapeDNA · HKS\nSI-HKS · WKS · GPS\nBKS / iBKS\nwavelets · fmaps", fc=AMBER_BG, ec=AMBER, t_fs=8.4)
# 4) embedded analyses
titled(ax, 0.655, ytop, 0.33, H, "embedded analyses",
       "shape distances · descriptor recommendation\nharmonization (ComBat / ComBat-GAM)\nclustering · RSA · classification\nnormative deviation scoring",
       fc=ROSE_BG, ec=ROSE, t_fs=8.4, b_fs=7.0)

# flow arrows
for x0, x1 in [(0.17,0.205),(0.38,0.415),(0.62,0.655)]:
    arrow(ax, x0, ytop+H/2, x1, ytop+H/2, color=SLATE, lw=2.0, ms=12)

# backend ribbon under decompose
box(ax, 0.205, 0.30, 0.175, 0.16,
    "backend =\nNumPy · PyTorch\nCuPy · JAX", fc="#eef2f4", ec=GREYEDGE, tc=SLATE, fs=7.6)
arrow(ax, 0.2925, 0.46, 0.2925, 0.545, color=GREYEDGE, lw=1.1, ls=(0,(4,2)))

# point-cloud note under data model
ax.text(0.0925, 0.47, "meshes & raw\npoint clouds", ha="center", va="top",
        fontsize=7.0, color=SLATE, style="italic")

# downstream arrow to statistics/viz
arrow(ax, 0.82, ytop, 0.82, 0.205, color=ROSE, lw=1.8)
box(ax, 0.655, 0.075, 0.33, 0.125,
    "→  statistics  &  visualization\n(Fig. 4 · vertex-wise fields, group tensors)",
    fc=GREYBOX, ec=GREYEDGE, tc=SLATE, fs=8.0, weight="bold")

# per-subject vs group note
ax.text(0.015, 0.04, "Runs per structure or, via GroupData, across an entire cohort "
        "(load_group(mode='pipeline')), optionally on GPU.",
        ha="left", va="center", fontsize=7.4, color=SLATE, style="italic")

save(fig, "figures/fig3_pipeline")
print("fig3 saved")
