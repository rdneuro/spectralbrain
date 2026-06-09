"""Figure 2 — Data ingestion: heterogeneous inputs -> unified data model."""
import matplotlib.pyplot as plt
from _figstyle import *

def titled(ax, x, y, w, h, title, body, *, fc, ec, tc=SLATE, t_fs=8.4, b_fs=7.4):
    box(ax, x, y, w, h, "", fc=fc, ec=ec)
    ax.text(x+w/2, y+h-0.030, title, ha="center", va="center", fontsize=t_fs,
            weight="bold", color=tc, zorder=3)
    ax.text(x+w/2, y+(h-0.052)/2, body, ha="center", va="center", fontsize=b_fs,
            color=tc, zorder=3, linespacing=1.42)

fig, ax = plt.subplots(figsize=(8.2, 5.0))
clean(ax)
ax.text(0.5, 0.97, "Data ingestion", ha="center", va="center",
        fontsize=12.5, weight="bold", color=INK)

# ── Column headers (non-overlapping) ─────────────────────────────
header(ax, 0.02, 0.905, "Inputs", color=TEAL, fs=10)
header(ax, 0.455, 0.905, "Loaders & discovery", color=INDIGO, fs=10)
header(ax, 0.795, 0.905, "Data model", color=INDIGO, fs=10)

# ── Column 1: input sources ──────────────────────────────────────
srcs = [
    "FreeSurfer surfaces  (.pial/.white)",
    "FreeSurfer morphometry  (.thickness)",
    "GIfTI  (.surf/.func/.shape.gii)",
    "NIfTI / MGZ volumes & labels",
    "HippUnfold v1 / v2  (den-8k)",
    "TractSeg bundle masks",
    "meshes (.ply/.obj/.stl/.vtk) · HDF5",
    "point clouds  (N×3)",
]
y = 0.86; h = 0.068; gap = 0.0145; ys = []
for s in srcs:
    box(ax, 0.02, y-h, 0.40, h, s, fc=TEAL_BG, ec=TEAL, tc=SLATE, fs=7.7, align="left")
    ys.append(y - h/2); y -= (h+gap)

# ── Column 2 ─────────────────────────────────────────────────────
titled(ax, 0.45, 0.605, 0.245, 0.255, "single-file",
       "load()\nload_mesh() · load_nifti()\nload_gifti_*()\nload_freesurfer_*()\nload_hdf5()",
       fc=INDIGO_BG, ec=INDIGO)
titled(ax, 0.45, 0.315, 0.245, 0.255, "cohort",
       "discover_bids()\ndiscover_freesurfer()\ndiscover_tractseg_*()\nload_group()\nresample_to_template()",
       fc=INDIGO_BG, ec=INDIGO)
titled(ax, 0.45, 0.085, 0.245, 0.195, "volume → surface",
       "marching cubes\n(scikit-image)\nROI / label select",
       fc="#eef2f4", ec=GREYEDGE)

# ── Column 3 ─────────────────────────────────────────────────────
box(ax, 0.79, 0.70, 0.195, 0.10, "BrainMesh  (V, F)", fc=INDIGO_BG, ec=INDIGO, tc=INK, fs=8.8, weight="bold")
box(ax, 0.79, 0.55, 0.195, 0.10, "BrainPointCloud (P)", fc=INDIGO_BG, ec=INDIGO, tc=INK, fs=8.6, weight="bold")
box(ax, 0.79, 0.37, 0.195, 0.12, "GroupData\nstacked cohort\n+ covariates", fc=INDIGO_BG, ec=INDIGO, tc=INK, fs=8.4, weight="bold")
box(ax, 0.79, 0.205, 0.195, 0.105, "decompose()\nLBO eigenpairs", fc=AMBER_BG, ec=AMBER, tc=SLATE, fs=8.4, weight="bold")

# ── arrows ───────────────────────────────────────────────────────
for yy in ys[:5]:
    arrow(ax, 0.42, yy, 0.447, 0.73, color=TEAL, lw=0.8, alpha=0.5, ms=7)
for yy in ys[5:]:
    arrow(ax, 0.42, yy, 0.447, 0.44, color=TEAL, lw=0.8, alpha=0.5, ms=7)
arrow(ax, 0.42, ys[3], 0.447, 0.18, color=GREYEDGE, lw=0.8, alpha=0.6, ms=7)
# loaders -> model
arrow(ax, 0.695, 0.73, 0.787, 0.75, color=INDIGO, lw=1.6)
arrow(ax, 0.695, 0.60, 0.787, 0.60, color=INDIGO, lw=1.6)
arrow(ax, 0.695, 0.44, 0.787, 0.43, color=INDIGO, lw=1.6)
arrow(ax, 0.695, 0.18, 0.787, 0.72, color=GREYEDGE, lw=1.0, ls=(0,(4,2)), ms=8)
# model -> decompose
arrow(ax, 0.8875, 0.70, 0.8875, 0.312, color=AMBER, lw=1.6)
arrow(ax, 0.8875, 0.55, 0.8875, 0.312, color=AMBER, lw=1.1, alpha=0.45)
arrow(ax, 0.8875, 0.37, 0.8875, 0.312, color=AMBER, lw=1.1, alpha=0.45)

save(fig, "figures/fig2_ingestion")
print("fig2 saved")
