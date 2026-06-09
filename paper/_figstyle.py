"""Shared schematic style for SpectralBrain JOSS figures (matplotlib, vector)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# Muted, colourblind-safe scientific palette (no red/green pairing).
INK      = "#1f2430"   # near-black text
SLATE    = "#37474f"
TEAL     = "#2a7f8e"   # inputs
TEAL_BG  = "#e3f0f2"
INDIGO   = "#3f51b5"   # core
INDIGO_BG= "#e8eaf6"
AMBER    = "#b9770e"   # spectral
AMBER_BG = "#fbeed3"
ROSE     = "#9b2d5e"   # statistics
ROSE_BG  = "#f6e2ec"
GREYBOX  = "#eceff1"
GREYEDGE = "#90a4ae"

def box(ax, x, y, w, h, text, *, fc=GREYBOX, ec=GREYEDGE, tc=INK,
        fs=9, weight="normal", round_r=0.018, lw=1.2, align="center", pad=0.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={pad},rounding_size={round_r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ha = {"center":"center","left":"left"}[align]
    tx = x + (w/2 if align=="center" else 0.012)
    ax.text(tx, y + h/2, text, ha=ha, va="center", fontsize=fs,
            color=tc, weight=weight, zorder=3, linespacing=1.25)
    return p

def arrow(ax, x0, y0, x1, y1, *, color=SLATE, lw=1.6, style="-|>", ms=10, alpha=1.0, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=ms, lw=lw, color=color, alpha=alpha,
                        linestyle=ls, zorder=1, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a

def header(ax, x, y, text, *, color=SLATE, fs=10.5):
    ax.text(x, y, text, ha="left", va="center", fontsize=fs, weight="bold",
            color=color, zorder=3)

def clean(ax, xlim=(0,1), ylim=(0,1)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    ax.set_aspect("auto")

def save(fig, path):
    fig.savefig(path + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path + ".pdf", bbox_inches="tight", facecolor="white")
