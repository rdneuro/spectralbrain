"""CPU compute backend — NumPy/SciPy, Bayesian samplers, and parallelisation.

This module provides:

1. **NumpyBackend** — the default compute engine using SciPy sparse
   eigensolvers (ARPACK), sparse matrix operations, and NumPy array
   algebra.  Every other backend mirrors this interface.
2. **PyMCSampler / NutpieSampler** — Bayesian MCMC backends for the
   ``statistics/bayesian.py`` module.
3. **Joblib utilities** — composable parallelisation helpers with
   Rich progress integration.
4. **RAM management** — memory monitoring, garbage collection, and
   estimation helpers for multi-subject pipelines.

All optional dependencies (PyMC, nutpie, joblib) are lazy-imported.
Only NumPy and SciPy are hard requirements.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeVar,
)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spectralbrain.runtime import (
    Eigenvalues,
    Eigenvectors,
    MassMatrix,
    SparseMatrix,
    get_logger,
    progress_parallel,
)

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ======================================================================
# §1  NUMPY / SCIPY COMPUTE BACKEND
# ======================================================================


class NumpyBackend:
    """CPU compute backend using NumPy + SciPy.

    Provides the canonical interface that :class:`CupyBackend` and
    :class:`JaxBackend` mirror.  All ``core/`` and ``spectral/``
    modules call backend methods rather than importing NumPy or SciPy
    directly, enabling transparent GPU acceleration.

    Examples
    --------
    >>> from spectralbrain.backends.cpu import NumpyBackend
    >>> be = NumpyBackend()
    >>> evals, evecs = be.eigsh(L, M, k=100)
    >>> hks = be.exp(-evals[None, :] * t[:, None])  # broadcasting
    """

    name: str = "numpy"

    # ── Sparse eigensolvers ───────────────────────────────────────────

    @staticmethod
    def eigsh(
        L: SparseMatrix,
        M: MassMatrix | None = None,
        k: int = 100,
        *,
        sigma: float = -0.01,
        which: str = "LM",
        tol: float = 0.0,
        maxiter: int | None = None,
    ) -> tuple[Eigenvalues, Eigenvectors]:
        """Solve the generalised sparse eigenproblem L v = λ M v.

        Uses SciPy's ARPACK wrapper in shift-invert mode (default
        σ = −0.01) which is optimal for computing the *smallest*
        eigenvalues of the Laplacian.

        Parameters
        ----------
        L : sparse matrix, shape (N, N)
            Stiffness (Laplacian) matrix — symmetric positive
            semi-definite.
        M : sparse matrix, shape (N, N), optional
            Mass matrix.  If ``None``, the standard eigenproblem
            L v = λ v is solved.
        k : int
            Number of eigenpairs to compute.
        sigma : float
            Shift for shift-invert mode.  A small negative value
            avoids the singularity at λ = 0.
        which : str
            Which eigenvalues to target (``"LM"`` = largest magnitude
            *of the shifted operator*, yielding the smallest λ).
        tol : float
            Convergence tolerance (0 = machine precision).
        maxiter : int, optional
            Maximum ARPACK iterations.

        Returns
        -------
        eigenvalues : ndarray, shape (k,)
            Sorted ascending, float64.
        eigenvectors : ndarray, shape (N, k)
            Corresponding eigenvectors, M-orthonormal.

        Raises
        ------
        scipy.sparse.linalg.ArpackNoConvergence
            If ARPACK fails to converge within *maxiter* iterations.
        """
        L = sp.csc_matrix(L, dtype=np.float64)
        if M is not None:
            M = sp.csc_matrix(M, dtype=np.float64)

        eigenvalues, eigenvectors = spla.eigsh(
            L,
            k=k,
            M=M,
            sigma=sigma,
            which=which,
            tol=tol,
            maxiter=maxiter,
        )

        # Sort ascending (ARPACK returns in arbitrary order after
        # shift-invert).
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Clamp tiny negative eigenvalues from numerical noise.
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        return eigenvalues, eigenvectors

    # ── Sparse matrix construction ────────────────────────────────────

    @staticmethod
    def sparse_matrix(
        data: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
        shape: tuple[int, int],
        *,
        format: str = "csc",
    ) -> SparseMatrix:
        """Build a sparse matrix from COO triplets.

        Parameters
        ----------
        data : ndarray
            Non-zero values.
        row, col : ndarray
            Row and column indices.
        shape : (int, int)
            Matrix dimensions.
        format : str
            Output format (``"csc"``, ``"csr"``, ``"coo"``).

        Returns
        -------
        SparseMatrix
        """
        coo = sp.coo_matrix(
            (
                np.asarray(data, dtype=np.float64),
                (np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)),
            ),
            shape=shape,
        )
        if format == "csc":
            return coo.tocsc()
        elif format == "csr":
            return coo.tocsr()
        return coo

    # ── Dense array operations ────────────────────────────────────────
    # These thin wrappers exist so that CupyBackend / JaxBackend can
    # override them transparently.

    @staticmethod
    def array(data: Any, dtype: np.dtype = np.float64) -> np.ndarray:
        """Create a dense array."""
        return np.asarray(data, dtype=dtype)

    @staticmethod
    def zeros(shape: tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """Create a zero-filled array (mirrors numpy.zeros)."""
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape: tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """Create a ones-filled array (mirrors numpy.ones)."""
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def eye(n: int, dtype: np.dtype = np.float64) -> np.ndarray:
        """Create an identity matrix (mirrors numpy.eye)."""
        return np.eye(n, dtype=dtype)

    @staticmethod
    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiply (sparse- and dense-aware)."""
        if sp.issparse(a) or sp.issparse(b):
            return a @ b
        return np.matmul(a, b)

    @staticmethod
    def exp(x: np.ndarray) -> np.ndarray:
        """Element-wise exponential (mirrors numpy.exp)."""
        return np.exp(x)

    @staticmethod
    def log(x: np.ndarray) -> np.ndarray:
        """Element-wise safe log with clamp at 1e-300."""
        return np.log(np.clip(x, 1e-300, None))

    @staticmethod
    def sqrt(x: np.ndarray) -> np.ndarray:
        """Element-wise safe sqrt with clamp at 0."""
        return np.sqrt(np.clip(x, 0.0, None))

    @staticmethod
    def sum(x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Sum reduction (mirrors numpy.sum)."""
        return np.sum(x, axis=axis)

    @staticmethod
    def mean(x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Mean reduction (mirrors numpy.mean)."""
        return np.mean(x, axis=axis)

    @staticmethod
    def clip(x: np.ndarray, a_min: float | None, a_max: float | None) -> np.ndarray:
        """Element-wise clip (mirrors numpy.clip)."""
        return np.clip(x, a_min, a_max)

    @staticmethod
    def to_numpy(x: Any) -> np.ndarray:
        """Convert any array-like to a NumPy ndarray."""
        if isinstance(x, np.ndarray):
            return x
        if sp.issparse(x):
            return x.toarray()
        return np.asarray(x)

    @staticmethod
    def norm(x: np.ndarray, axis: int | None = None, ord: int | None = None) -> np.ndarray:
        """Vector/matrix norm (mirrors numpy.linalg.norm)."""
        return np.linalg.norm(x, axis=axis, ord=ord)

    @staticmethod
    def argsort(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Indirect sort indices (mirrors numpy.argsort)."""
        return np.argsort(x, axis=axis)

    @staticmethod
    def concatenate(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
        """Concatenate arrays along an axis."""
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def stack(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
        """Stack arrays along a new axis."""
        return np.stack(arrays, axis=axis)

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> np.ndarray:
        """Linearly spaced values (mirrors numpy.linspace)."""
        return np.linspace(start, stop, num, dtype=np.float64)

    @staticmethod
    def logspace(start: float, stop: float, num: int) -> np.ndarray:
        """Log-spaced values (mirrors numpy.logspace)."""
        return np.logspace(start, stop, num, dtype=np.float64)


# ======================================================================
# §2  BAYESIAN CPU SAMPLERS
# ======================================================================


def _require_pymc():
    """Lazy-import PyMC, raising ImportError if unavailable."""
    try:
        import pymc as pm

        return pm
    except ImportError as exc:
        raise ImportError("PyMC is required for Bayesian analysis.\n  pip install pymc") from exc


def _require_nutpie():
    """Lazy-import nutpie, raising ImportError if unavailable."""
    try:
        import nutpie

        return nutpie
    except ImportError as exc:
        raise ImportError(
            "nutpie is required for the nutpie sampler backend.\n  pip install nutpie"
        ) from exc


def _require_arviz():
    """Lazy-import ArviZ, raising ImportError if unavailable."""
    try:
        import arviz as az

        return az
    except ImportError as exc:
        raise ImportError(
            "ArviZ is required for Bayesian diagnostics.\n  pip install arviz"
        ) from exc


@dataclass
class SamplerConfig:
    """Configuration for Bayesian MCMC samplers.

    Parameters
    ----------
    draws : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning (burn-in) samples.
    chains : int
        Number of independent chains.
    cores : int
        CPU cores for parallel chains.
    target_accept : float
        Target acceptance probability for NUTS.
    random_seed : int or None
        RNG seed for reproducibility.
    """

    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    cores: int = 4
    target_accept: float = 0.95
    random_seed: int | None = 42


class PyMCSampler:
    """Bayesian sampler using PyMC's native NUTS implementation.

    This is the default CPU sampler.  It wraps ``pymc.sample()`` with
    SpectralBrain-compatible configuration and logging.

    Parameters
    ----------
    config : SamplerConfig, optional
        Sampling configuration.

    Examples
    --------
    >>> sampler = PyMCSampler(SamplerConfig(draws=1000, chains=2))
    >>> with pm.Model() as model:
    ...     mu = pm.Normal("mu", 0, 1)
    ...     obs = pm.Normal("obs", mu, 1, observed=data)
    >>> trace = sampler.sample(model)
    """

    name: str = "nuts"

    def __init__(self, config: SamplerConfig | None = None) -> None:
        """Initialise with optional SamplerConfig."""
        self.config = config or SamplerConfig()

    def sample(
        self,
        model: Any,  # pm.Model
        **kwargs: Any,
    ) -> Any:  # az.InferenceData
        """Run NUTS sampling on a PyMC model.

        Parameters
        ----------
        model : pymc.Model
            A fully specified PyMC model.
        **kwargs
            Overrides passed to ``pymc.sample()``.

        Returns
        -------
        arviz.InferenceData
            Posterior samples with diagnostics.
        """
        pm = _require_pymc()
        cfg = self.config

        sample_kwargs = dict(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            cores=cfg.cores,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
        sample_kwargs.update(kwargs)

        logger.info(
            "PyMC NUTS: %d draws × %d chains (%d tune)",
            cfg.draws,
            cfg.chains,
            cfg.tune,
        )

        with model:
            trace = pm.sample(**sample_kwargs)

        return trace


class NutpieSampler:
    """Bayesian sampler using nutpie (Rust-based NUTS).

    nutpie is a high-performance drop-in replacement for PyMC's
    default sampler.  It compiles the PyMC model to Rust and runs
    NUTS 2–10× faster on CPU.

    Parameters
    ----------
    config : SamplerConfig, optional
        Sampling configuration.

    Examples
    --------
    >>> sampler = NutpieSampler(SamplerConfig(draws=2000))
    >>> trace = sampler.sample(model)
    """

    name: str = "nutpie"

    def __init__(self, config: SamplerConfig | None = None) -> None:
        """Initialise with optional SamplerConfig."""
        self.config = config or SamplerConfig()

    def sample(
        self,
        model: Any,  # pm.Model
        **kwargs: Any,
    ) -> Any:  # az.InferenceData
        """Run nutpie NUTS on a PyMC model.

        Parameters
        ----------
        model : pymc.Model
            A fully specified PyMC model.
        **kwargs
            Overrides passed to ``nutpie.sample()``.

        Returns
        -------
        arviz.InferenceData
        """
        nutpie = _require_nutpie()
        cfg = self.config

        logger.info(
            "nutpie NUTS: %d draws × %d chains (%d tune)",
            cfg.draws,
            cfg.chains,
            cfg.tune,
        )

        compiled = nutpie.compile_pymc_model(model)
        trace = nutpie.sample(
            compiled,
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            seed=cfg.random_seed,
            progress_bar=True,
            **kwargs,
        )
        return trace


def get_bayesian_sampler(
    backend: Literal["nuts", "nutpie"] = "nuts",
    config: SamplerConfig | None = None,
) -> PyMCSampler | NutpieSampler:
    """Factory for CPU Bayesian samplers.

    Parameters
    ----------
    backend : ``"nuts"`` or ``"nutpie"``
        Which sampler to use.
    config : SamplerConfig, optional
        Sampling parameters.

    Returns
    -------
    PyMCSampler or NutpieSampler
    """
    if backend == "nuts":
        return PyMCSampler(config)
    elif backend == "nutpie":
        return NutpieSampler(config)
    else:
        raise ValueError(f"Unknown CPU Bayesian backend: {backend!r}")


# ======================================================================
# §3  JOBLIB PARALLELISATION UTILITIES
# ======================================================================


def _require_joblib():
    """Lazy-import joblib, raising ImportError if unavailable."""
    try:
        import joblib

        return joblib
    except ImportError as exc:
        raise ImportError("joblib is required for parallelisation.\n  pip install joblib") from exc


def parallel_map(
    func: Callable[..., R],
    items: Sequence[T],
    *,
    n_jobs: int = -1,
    backend: str = "loky",
    progress: bool = True,
    description: str = "Processing",
    **func_kwargs: Any,
) -> list[R]:
    """Apply *func* to each item in parallel with optional progress.

    A thin wrapper around ``joblib.Parallel`` that integrates with
    SpectralBrain's Rich progress bars.

    Parameters
    ----------
    func : callable
        Function to apply.  Must accept each item as its first
        positional argument.
    items : sequence
        Items to process.
    n_jobs : int
        Number of parallel workers (``-1`` = all cores).
    backend : str
        Joblib backend (``"loky"``, ``"threading"``, ``"multiprocessing"``).
    progress : bool
        Show a Rich progress bar.
    description : str
        Progress bar label.
    **func_kwargs
        Extra keyword arguments passed to *func*.

    Returns
    -------
    list
        Results in the same order as *items*.

    Examples
    --------
    >>> def compute(subj_id, k=100):
    ...     mesh = load(subj_id)
    ...     return decompose(mesh, k=k)
    >>> results = parallel_map(compute, subject_ids, n_jobs=8, k=50)
    """
    joblib = _require_joblib()
    total = len(items)

    if not progress:
        return joblib.Parallel(n_jobs=n_jobs, backend=backend)(
            joblib.delayed(func)(item, **func_kwargs) for item in items
        )

    # Stream results back in submission order and advance the progress bar
    # in THIS (parent) process.  The progress object holds a thread lock and
    # must never be captured in the worker closure — doing so breaks pickling
    # under the process-based ``loky`` backend.
    results: list[R | None] = [None] * total
    with progress_parallel(description, total=total) as tick:
        stream = joblib.Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")(
            joblib.delayed(func)(item, **func_kwargs) for item in items
        )
        for idx, result in enumerate(stream):
            results[idx] = result
            tick(1)

    return results  # type: ignore[return-value]


def parallel_batch(
    func: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    *,
    batch_size: int = 1000,
    n_jobs: int = -1,
    axis: int = 0,
    progress: bool = True,
    description: str = "Batch processing",
) -> np.ndarray:
    """Apply *func* to batches of an array in parallel.

    Splits *data* along *axis* into chunks of *batch_size*, applies
    *func* to each chunk in parallel, and concatenates the results.
    Useful for operations that are O(N²) per vertex but can be
    batched (e.g. geodesic distance computation).

    Parameters
    ----------
    func : callable
        Function that accepts an ndarray batch and returns an ndarray.
    data : ndarray
        Full array to process.
    batch_size : int
        Number of rows per batch.
    n_jobs : int
        Parallel workers.
    axis : int
        Axis along which to split.
    progress : bool
        Show progress bar.
    description : str
        Progress label.

    Returns
    -------
    ndarray
        Concatenated results.
    """
    joblib = _require_joblib()
    n = data.shape[axis]
    slices = [slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    batches = [np.take(data, range(*s.indices(n)), axis=axis) for s in slices]

    total = len(batches)
    if progress:
        # Stream results and tick in the parent; never pickle the progress
        # object into worker processes (see parallel_map for the rationale).
        with progress_parallel(description, total=total) as tick:
            stream = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
                joblib.delayed(func)(b) for b in batches
            )
            results = []
            for r in stream:
                results.append(r)
                tick(1)
    else:
        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(func)(b) for b in batches)

    return np.concatenate(results, axis=axis)


def batch_iterator(
    data: np.ndarray,
    batch_size: int = 1000,
    *,
    axis: int = 0,
) -> Iterator[np.ndarray]:
    """Iterate over an array in memory-safe batches.

    Unlike :func:`parallel_batch`, this is a sequential generator
    suitable for GPU-offloading loops where only one batch should
    be in memory at a time.

    Parameters
    ----------
    data : ndarray
        Array to iterate.
    batch_size : int
        Rows per batch.
    axis : int
        Axis to split along.

    Yields
    ------
    ndarray
        A view (not copy) of the batch.

    Examples
    --------
    >>> for batch in batch_iterator(big_array, batch_size=500):
    ...     result = expensive_compute(batch)
    ...     accumulate(result)
    """
    n = data.shape[axis]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = [slice(None)] * data.ndim
        idx[axis] = slice(start, end)
        yield data[tuple(idx)]


# ======================================================================
# §4  RAM MEMORY MANAGEMENT
# ======================================================================


@dataclass
class MemoryInfo:
    """Snapshot of system RAM usage.

    Attributes
    ----------
    total_gb : float
        Total physical RAM.
    available_gb : float
        Available (free + cached) RAM.
    used_gb : float
        Actively used RAM.
    percent_used : float
        Usage percentage (0–100).
    """

    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float

    def __repr__(self) -> str:
        """Return a human-readable summary."""
        return (
            f"RAM: {self.used_gb:.1f} / {self.total_gb:.1f} GB "
            """Return a human-readable RAM status summary."""
            f"({self.percent_used:.0f}% used, "
            f"{self.available_gb:.1f} GB free)"
        )


def ram_status() -> MemoryInfo:
    """Return current system RAM usage.

    Uses ``/proc/meminfo`` on Linux and ``psutil`` as fallback.

    Returns
    -------
    MemoryInfo
    """
    # Try /proc/meminfo first (no dependency, Linux only).
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        info: dict[str, int] = {}
        with open(meminfo_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val_kb = int(parts[1])
                    info[key] = val_kb
        total = info.get("MemTotal", 0) / (1024**2)
        available = info.get("MemAvailable", 0) / (1024**2)
        used = total - available
        pct = 100 * used / total if total > 0 else 0
        return MemoryInfo(total, available, used, pct)

    # Fallback: psutil.
    try:
        import psutil

        vm = psutil.virtual_memory()
        return MemoryInfo(
            vm.total / (1024**3),
            vm.available / (1024**3),
            vm.used / (1024**3),
            vm.percent,
        )
    except ImportError:
        logger.warning("Cannot read memory info: /proc/meminfo not found and psutil not installed.")
        return MemoryInfo(0, 0, 0, 0)


def gc_collect(generations: int = 2) -> int:
    """Force Python garbage collection and return bytes freed.

    Parameters
    ----------
    generations : int
        GC generations to collect (0, 1, or 2).

    Returns
    -------
    int
        Number of unreachable objects collected.

    Examples
    --------
    >>> del large_array
    >>> freed = gc_collect()
    >>> logger.info("GC freed %d objects", freed)
    """
    collected = 0
    for gen in range(generations + 1):
        collected += gc.collect(gen)
    logger.debug("GC collected %d objects (gen 0–%d)", collected, generations)
    return collected


def estimate_array_memory(
    shape: tuple[int, ...],
    dtype: np.dtype = np.float64,
) -> float:
    """Estimate memory for an array in gigabytes.

    Parameters
    ----------
    shape : tuple of int
    dtype : numpy dtype

    Returns
    -------
    float
        Estimated size in GB.

    Examples
    --------
    >>> estimate_array_memory((160_000, 300), np.float64)
    0.358  # ~358 MB for cortical eigenvectors
    """
    n_elements = 1
    for s in shape:
        n_elements *= s
    bytes_per_element = np.dtype(dtype).itemsize
    return n_elements * bytes_per_element / (1024**3)


@contextmanager
def memory_guard(
    min_available_gb: float = 2.0,
    error_on_low: bool = False,
) -> Generator[None, None, None]:
    """Context manager that checks RAM before and after a block.

    Parameters
    ----------
    min_available_gb : float
        Minimum free RAM required to proceed.
    error_on_low : bool
        Raise ``MemoryError`` if RAM is below threshold.
        If ``False`` (default), logs a warning instead.

    Examples
    --------
    >>> with memory_guard(min_available_gb=4.0):
    ...     big_result = compute_all_subjects()
    """
    info = ram_status()
    if info.available_gb < min_available_gb:
        msg = (
            f"Low RAM: {info.available_gb:.1f} GB available, "
            f"need {min_available_gb:.1f} GB. "
            f"Consider closing other applications."
        )
        if error_on_low:
            raise MemoryError(msg)
        logger.warning(msg)

    yield

    # Post-block: report if memory increased significantly.
    info_after = ram_status()
    delta = info.available_gb - info_after.available_gb
    if delta > 1.0:
        logger.info(
            "Block consumed ~%.1f GB RAM (%.1f → %.1f GB free)",
            delta,
            info.available_gb,
            info_after.available_gb,
        )


def shrink_array(
    arr: np.ndarray,
    target_dtype: np.dtype | None = None,
) -> np.ndarray:
    """Downcast an array to save memory.

    If *target_dtype* is ``None``, applies safe downcasting rules:
    float64 → float32, int64 → int32 (if values fit).

    Parameters
    ----------
    arr : ndarray
    target_dtype : dtype, optional

    Returns
    -------
    ndarray
        A (possibly) smaller copy.
    """
    if target_dtype is not None:
        return arr.astype(target_dtype, copy=False)

    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    if arr.dtype == np.int64:
        if arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max:
            return arr.astype(np.int32)
    return arr


# ======================================================================
# §5  __all__
# ======================================================================

__all__: list[str] = [
    # RAM management
    "MemoryInfo",
    # Compute backend
    "NumpyBackend",
    "NutpieSampler",
    "PyMCSampler",
    # Bayesian samplers
    "SamplerConfig",
    "batch_iterator",
    "estimate_array_memory",
    "gc_collect",
    "get_bayesian_sampler",
    "memory_guard",
    "parallel_batch",
    # Parallelisation
    "parallel_map",
    "ram_status",
    "shrink_array",
]
