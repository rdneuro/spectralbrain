"""Miscellaneous utilities: timing, reproducibility, I/O helpers.

Small tools used across SpectralBrain that don't belong in any
specific subpackage.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)


# ======================================================================
# §1  TIMING
# ======================================================================

@contextmanager
def timer(label: str = "Operation") -> Generator[None, None, None]:
    """Context manager that logs elapsed time.

    Parameters
    ----------
    label : str
        Description for the log message.

    Examples
    --------
    >>> with timer("Eigendecomposition"):
    ...     decomp = mesh.decompose(k=100)
    # [HH:MM:SS] INFO  Eigendecomposition: 12.34s
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if elapsed < 60:
        logger.info("%s: %.2fs", label, elapsed)
    elif elapsed < 3600:
        logger.info("%s: %dm %02ds", label, int(elapsed // 60), int(elapsed % 60))
    else:
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        logger.info("%s: %dh %02dm", label, h, m)


class Timer:
    """Reusable timer with lap support.

    Examples
    --------
    >>> t = Timer()
    >>> t.start()
    >>> process_a()
    >>> t.lap("step A")
    >>> process_b()
    >>> t.lap("step B")
    >>> t.report()
    """

    def __init__(self) -> None:
        """Initialise the timer with an optional label."""
        self._start: float = 0
        self._laps: List[tuple] = []

    def start(self) -> "Timer":
        """Start the timer and return self."""
        self._start = time.perf_counter()
        self._laps = []
        return self

    def lap(self, label: str = "") -> float:
        """Record a lap. Returns seconds since start."""
        elapsed = time.perf_counter() - self._start
        self._laps.append((label, elapsed))
        return elapsed

    def report(self) -> Dict[str, float]:
        """Return and log all laps."""
        result = {}
        prev = 0.0
        for label, cumulative in self._laps:
            delta = cumulative - prev
            result[label] = delta
            logger.info("  %s: %.2fs (cumulative: %.2fs)", label, delta, cumulative)
            prev = cumulative
        total = time.perf_counter() - self._start
        result["_total"] = total
        logger.info("  Total: %.2fs", total)
        return result


# ======================================================================
# §2  REPRODUCIBILITY
# ======================================================================

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for NumPy, Python, and optional frameworks.

    Parameters
    ----------
    seed : int
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import jax
        # JAX uses explicit PRNG keys, not global state.
        # Set env var for deterministic ops.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") +
            " --xla_gpu_deterministic_reductions"
        )
    except ImportError:
        pass

    logger.debug("Seeded everything with %d", seed)


def get_reproducibility_info() -> Dict[str, str]:
    """Collect version info for reproducibility metadata.

    Returns
    -------
    dict
        Keys: python, numpy, scipy, spectralbrain, platform, date.
    """
    import platform as plat
    import scipy

    from spectralbrain.runtime import __version__

    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "spectralbrain": __version__,
        "platform": plat.platform(),
        "date": datetime.now().isoformat(),
    }

    for pkg in ("pymc", "arviz", "nibabel", "sklearn", "rich"):
        try:
            mod = __import__(pkg)
            info[pkg] = mod.__version__
        except (ImportError, AttributeError):
            pass

    return info


# ======================================================================
# §3  FILE / PATH HELPERS
# ======================================================================

def ensure_dir(path: PathLike) -> Path:
    """Create directory if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_hash(path: PathLike, algorithm: str = "sha256") -> str:
    """Compute hash of a file.

    Parameters
    ----------
    path : PathLike
    algorithm : str
        ``"sha256"``, ``"md5"``, etc.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_files(
    directory: PathLike,
    pattern: str = "*",
    recursive: bool = True,
) -> List[Path]:
    """Glob for files in a directory.

    Parameters
    ----------
    directory : PathLike
    pattern : str
        Glob pattern (e.g. ``"*.nii.gz"``).
    recursive : bool

    Returns
    -------
    list of Path
    """
    d = Path(directory)
    if recursive:
        return sorted(d.rglob(pattern))
    return sorted(d.glob(pattern))


# ======================================================================
# §4  SUBJECTS / BIDS HELPERS
# ======================================================================

def parse_bids_filename(filename: str) -> Dict[str, str]:
    """Extract BIDS entities from a filename.

    Parameters
    ----------
    filename : str
        E.g. ``"sub-01_ses-pre_T1w.nii.gz"``.

    Returns
    -------
    dict
        E.g. ``{"sub": "01", "ses": "pre", "suffix": "T1w"}``.
    """
    stem = Path(filename).name.split(".")[0]
    parts = stem.split("_")
    entities: Dict[str, str] = {}
    for part in parts:
        if "-" in part:
            key, val = part.split("-", 1)
            entities[key] = val
        else:
            entities["suffix"] = part
    return entities


def collect_subjects(
    bids_dir: PathLike,
    pattern: str = "sub-*",
) -> List[str]:
    """List subject IDs in a BIDS directory.

    Parameters
    ----------
    bids_dir : PathLike
    pattern : str

    Returns
    -------
    list of str
        Subject IDs (e.g. ``["sub-01", "sub-02", ...]``).
    """
    d = Path(bids_dir)
    return sorted([
        p.name for p in d.glob(pattern)
        if p.is_dir()
    ])


# ======================================================================
# §5  PRETTY PRINTING
# ======================================================================

def print_dict(
    d: Dict[str, Any],
    *,
    title: Optional[str] = None,
    indent: int = 2,
) -> None:
    """Pretty-print a dict with Rich (fallback to plain)."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        if title:
            console.rule(f"[bold]{title}")
        table = Table(show_header=True)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for k, v in d.items():
            table.add_row(str(k), str(v))
        console.print(table)
    except ImportError:
        if title:
            print(f"--- {title} ---")
        for k, v in d.items():
            print(f"{' ' * indent}{k}: {v}")


def format_array_summary(arr: np.ndarray, name: str = "array") -> str:
    """One-line summary of an array."""
    if arr.size == 0:
        return f"{name}: empty"
    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"range=[{arr.min():.4g}, {arr.max():.4g}], "
        f"mean={arr.mean():.4g}, std={arr.std():.4g}"
    )


__all__ = [
    # Timing
    "timer", "Timer",
    # Reproducibility
    "seed_everything", "get_reproducibility_info",
    # File/path
    "ensure_dir", "file_hash", "find_files",
    # BIDS
    "parse_bids_filename", "collect_subjects",
    # Pretty printing
    "print_dict", "format_array_summary",
]
