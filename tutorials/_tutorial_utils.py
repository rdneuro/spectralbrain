"""Shared helpers for the SpectralBrain tutorial series.

Keeps each notebook focused on concepts rather than boilerplate: it locates the
bundled ``data/`` folder and offers a couple of inline-plotting conveniences.
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("VTK_USE_OFFSCREEN", "1")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

# data/ sits next to this file, inside tutorials/
DATA = Path(__file__).resolve().parent / "data"


def data_path(*parts) -> Path:
    """Build a path inside the bundled tutorial dataset."""
    p = DATA.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Is the tutorials/data folder present?")
    return p


def spectrum_plot(eigenvalues, ax=None, label=None, **kw):
    """Plot a Laplace-Beltrami eigenvalue spectrum (index vs lambda)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    ev = np.asarray(eigenvalues)
    ax.plot(np.arange(len(ev)), ev, marker="o", ms=3, lw=1.2, label=label, **kw)
    ax.set_xlabel("eigenvalue index $k$")
    ax.set_ylabel(r"eigenvalue $\lambda_k$")
    ax.grid(alpha=0.3)
    if label:
        ax.legend(fontsize=8)
    return ax
