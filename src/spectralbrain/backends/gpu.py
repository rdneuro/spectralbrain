"""GPU compute backends — CuPy, JAX, NumPyro, and VRAM management.

This module provides:

1. **CupyBackend** — drop-in GPU replacement for :class:`NumpyBackend`
   using CuPy (``cupy-cuda13x``).
2. **JaxBackend** — GPU backend with ``jit`` and ``vmap`` for
   batch-subject spectral descriptor computation.
3. **NumPyroSampler** — GPU-accelerated Bayesian MCMC via JAX.
4. **VRAM management** — monitoring, cache clearing, defragmentation,
   garbage collection, and a memory-guarded context manager.

All dependencies are lazy-imported.  If CuPy or JAX is missing, the
module still imports successfully — only instantiation of the backends
raises ``ImportError``.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import scipy.sparse as sp

from spectralbrain.runtime import (
    Eigenvalues,
    Eigenvectors,
    MassMatrix,
    SparseMatrix,
    get_logger,
)

logger = get_logger(__name__)


# ======================================================================
# Lazy imports
# ======================================================================

def _require_cupy():
    """Lazy-import CuPy, raising ImportError if unavailable."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpla
        return cp, cpsp, cpla
    except ImportError as exc:
        raise ImportError(
            "CuPy is required for the CuPy GPU backend.\n"
            "  pip install cupy-cuda13x"
        ) from exc


def _require_jax():
    """Lazy-import JAX, raising ImportError if unavailable."""
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy.sparse.linalg as jsla
        return jax, jnp, jsla
    except ImportError as exc:
        raise ImportError(
            "JAX is required for the JAX GPU backend.\n"
            "  pip install 'jax[cuda13]' jaxlib"
        ) from exc


def _require_numpyro():
    """Lazy-import NumPyro, raising ImportError if unavailable."""
    try:
        import numpyro
        import numpyro.infer as infer
        return numpyro, infer
    except ImportError as exc:
        raise ImportError(
            "NumPyro is required for GPU Bayesian inference.\n"
            "  pip install numpyro"
        ) from exc


# ======================================================================
# §1  CUPY BACKEND
# ======================================================================

class CupyBackend:
    """GPU compute backend using CuPy.

    Mirrors the :class:`NumpyBackend` interface.  Arrays live on
    the GPU; :meth:`to_numpy` copies back to host.

    Parameters
    ----------
    device_id : int
        CUDA device index.

    Examples
    --------
    >>> be = CupyBackend(device_id=0)
    >>> evals, evecs = be.eigsh(L, M, k=100)
    >>> type(evals)  # cupy.ndarray — lives on GPU
    """

    name: str = "cupy"

    def __init__(self, device_id: int = 0) -> None:
        """Initialise the CuPy GPU backend."""
        cp, cpsp, cpla = _require_cupy()
        self._cp = cp
        self._cpsp = cpsp
        self._cpla = cpla
        self.device_id = device_id
        self._cp.cuda.Device(device_id).use()
        logger.info(
            "CuPy backend initialised on GPU %d: %s",
            device_id, self._cp.cuda.runtime.getDeviceProperties(device_id)["name"],
        )

    # ── Sparse eigensolver ────────────────────────────────────────────

    def eigsh(
        self,
        L: SparseMatrix,
        M: Optional[MassMatrix] = None,
        k: int = 100,
        *,
        sigma: float = -0.01,
        which: str = "LM",
        tol: float = 0.0,
        maxiter: Optional[int] = None,
    ) -> Tuple[Eigenvalues, Eigenvectors]:
        """GPU sparse eigensolver via CuPy's ARPACK wrapper.

        Parameters and returns mirror :meth:`NumpyBackend.eigsh`.
        Results are returned as **host** (NumPy) arrays.
        """
        cp = self._cp

        # Transfer to GPU as CSC.
        L_gpu = self._cpsp.csc_matrix(L.tocsc().astype(np.float64))
        M_gpu = (
            self._cpsp.csc_matrix(M.tocsc().astype(np.float64))
            if M is not None else None
        )

        evals_gpu, evecs_gpu = self._cpla.eigsh(
            L_gpu, k=k, M=M_gpu,
            sigma=sigma, which=which,
            tol=tol, maxiter=maxiter,
        )

        # Back to host, sort ascending, clamp negatives.
        evals = cp.asnumpy(evals_gpu)
        evecs = cp.asnumpy(evecs_gpu)
        order = np.argsort(evals)
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]

        return evals, evecs

    # ── Sparse matrix ─────────────────────────────────────────────────

    def sparse_matrix(
        self,
        data: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
        shape: Tuple[int, int],
        **kwargs: Any,
    ) -> Any:
        """Build a sparse matrix from COO triplets on GPU."""
        cp = self._cp
        return self._cpsp.coo_matrix(
            (cp.asarray(data, dtype=cp.float64),
             (cp.asarray(row, dtype=cp.int64),
              cp.asarray(col, dtype=cp.int64))),
            shape=shape,
        ).tocsc()

    # ── Dense ops (GPU arrays) ────────────────────────────────────────

    def array(self, data: Any, dtype: Any = np.float64) -> Any:
        """Create a CuPy array on GPU."""
        return self._cp.asarray(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a zero-filled CuPy array."""
        return self._cp.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a ones-filled CuPy array."""
        return self._cp.ones(shape, dtype=dtype)

    def eye(self, n: int, dtype: Any = np.float64) -> Any:
        """Create a GPU identity matrix."""
        return self._cp.eye(n, dtype=dtype)

    def matmul(self, a: Any, b: Any) -> Any:
        """GPU matrix multiply."""
        return a @ b

    def exp(self, x: Any) -> Any:
        """Element-wise exponential on GPU."""
        return self._cp.exp(x)

    def log(self, x: Any) -> Any:
        """Element-wise safe log on GPU."""
        return self._cp.log(self._cp.clip(x, 1e-300, None))

    def sqrt(self, x: Any) -> Any:
        """Element-wise safe sqrt on GPU."""
        return self._cp.sqrt(self._cp.clip(x, 0.0, None))

    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        """Sum reduction on GPU."""
        return self._cp.sum(x, axis=axis)

    def mean(self, x: Any, axis: Optional[int] = None) -> Any:
        """Mean reduction on GPU."""
        return self._cp.mean(x, axis=axis)

    def clip(self, x: Any, a_min: Optional[float], a_max: Optional[float]) -> Any:
        """Element-wise clip on GPU."""
        return self._cp.clip(x, a_min, a_max)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Copy GPU array to host."""
        if isinstance(x, np.ndarray):
            return x
        return self._cp.asnumpy(x)

    def norm(self, x: Any, axis: Optional[int] = None, ord: Optional[int] = None) -> Any:
        """Vector/matrix norm on GPU."""
        return self._cp.linalg.norm(x, axis=axis, ord=ord)

    def argsort(self, x: Any, axis: int = -1) -> Any:
        """Indirect sort indices on GPU."""
        return self._cp.argsort(x, axis=axis)

    def concatenate(self, arrays: Sequence, axis: int = 0) -> Any:
        """Concatenate CuPy arrays."""
        return self._cp.concatenate(arrays, axis=axis)

    def stack(self, arrays: Sequence, axis: int = 0) -> Any:
        """Stack CuPy arrays along a new axis."""
        return self._cp.stack(arrays, axis=axis)

    def linspace(self, start: float, stop: float, num: int) -> Any:
        """Linearly spaced values on GPU."""
        return self._cp.linspace(start, stop, num, dtype=np.float64)

    def logspace(self, start: float, stop: float, num: int) -> Any:
        """Log-spaced values on GPU."""
        return self._cp.logspace(start, stop, num, dtype=np.float64)


# ======================================================================
# §2  JAX BACKEND
# ======================================================================

class JaxBackend:
    """GPU backend using JAX with ``jit`` and ``vmap``.

    The key advantage of JAX over CuPy for SpectralBrain is
    :func:`jax.vmap`, which vectorises descriptor computation across
    an entire cohort without explicit loops, and :func:`jax.jit`,
    which compiles hot paths for reuse.

    Parameters
    ----------
    device : str
        ``"gpu"`` or ``"cpu"``.

    Examples
    --------
    >>> be = JaxBackend()
    >>> # Batch HKS for 228 subjects:
    >>> batched_hks = be.vmap(compute_hks)(all_evals, all_evecs, t)
    """

    name: str = "jax"

    def __init__(self, device: str = "gpu") -> None:
        """Initialise the JaxBackend."""
        jax, jnp, jsla = _require_jax()
        self._jax = jax
        self._jnp = jnp
        self._jsla = jsla

        if device == "gpu":
            if not jax.devices("gpu"):
                logger.warning("No GPU found; JAX falling back to CPU.")
        logger.info("JAX backend: devices = %s", jax.devices())

    # ── Eigensolver ───────────────────────────────────────────────────

    def eigsh(
        self,
        L: SparseMatrix,
        M: Optional[MassMatrix] = None,
        k: int = 100,
        **kwargs: Any,
    ) -> Tuple[Eigenvalues, Eigenvectors]:
        """Sparse eigensolver via JAX's LOBPCG.

        For the generalised problem L v = λ M v, falls back to
        SciPy ARPACK on host and transfers results — JAX's sparse
        eigensolver does not yet support generalised problems
        natively.  The eigenvalues / vectors are returned as NumPy.

        Parameters
        ----------
        L, M, k : same as NumpyBackend.eigsh

        Returns
        -------
        eigenvalues, eigenvectors : NumPy arrays.
        """
        # JAX's sparse linalg is limited; delegate to SciPy for the
        # eigenproblem and use JAX for downstream descriptor math.
        import scipy.sparse.linalg as spla

        L_sp = sp.csc_matrix(L, dtype=np.float64)
        M_sp = sp.csc_matrix(M, dtype=np.float64) if M is not None else None

        sigma = kwargs.pop("sigma", -0.01)
        evals, evecs = spla.eigsh(
            L_sp, k=k, M=M_sp,
            sigma=sigma, which="LM",
        )
        order = np.argsort(evals)
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]
        return evals, evecs

    # ── JIT / VMAP helpers ────────────────────────────────────────────

    def jit(self, func: Callable, **kwargs: Any) -> Callable:
        """JIT-compile a function.

        Parameters
        ----------
        func : callable
            Pure function (no side effects).

        Returns
        -------
        callable
            JIT-compiled version.
        """
        return self._jax.jit(func, **kwargs)

    def vmap(
        self,
        func: Callable,
        in_axes: Any = 0,
        out_axes: Any = 0,
    ) -> Callable:
        """Auto-vectorise *func* over a batch axis.

        Parameters
        ----------
        func : callable
            Function operating on a single example.
        in_axes : int or tuple
            Which axes of each argument to vectorise over.
        out_axes : int or tuple
            Output batch axis.

        Returns
        -------
        callable
            Batched version of *func*.

        Examples
        --------
        >>> # Single-subject HKS: (k,), (N, k), (T,) → (N, T)
        >>> batched = be.vmap(compute_hks)
        >>> # Now: (S, k), (S, N, k), (T,) → (S, N, T)
        >>> all_hks = batched(all_evals, all_evecs, t_values)
        """
        return self._jax.vmap(func, in_axes=in_axes, out_axes=out_axes)

    # ── Dense array ops ───────────────────────────────────────────────

    def array(self, data: Any, dtype: Any = np.float64) -> Any:
        """Create a JAX array."""
        return self._jnp.asarray(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a zero-filled JAX array."""
        return self._jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a ones-filled JAX array."""
        return self._jnp.ones(shape, dtype=dtype)

    def eye(self, n: int, dtype: Any = np.float64) -> Any:
        """Create a JAX identity matrix."""
        return self._jnp.eye(n, dtype=dtype)

    def matmul(self, a: Any, b: Any) -> Any:
        """JAX matrix multiply."""
        return self._jnp.matmul(a, b)

    def exp(self, x: Any) -> Any:
        """Element-wise exponential via JAX."""
        return self._jnp.exp(x)

    def log(self, x: Any) -> Any:
        """Element-wise safe log via JAX."""
        return self._jnp.log(self._jnp.clip(x, 1e-300, None))

    def sqrt(self, x: Any) -> Any:
        """Element-wise safe sqrt via JAX."""
        return self._jnp.sqrt(self._jnp.clip(x, 0.0, None))

    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        """Sum reduction via JAX."""
        return self._jnp.sum(x, axis=axis)

    def mean(self, x: Any, axis: Optional[int] = None) -> Any:
        """Mean reduction via JAX."""
        return self._jnp.mean(x, axis=axis)

    def clip(self, x: Any, a_min: Optional[float], a_max: Optional[float]) -> Any:
        """Element-wise clip via JAX."""
        return self._jnp.clip(x, a_min, a_max)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Transfer JAX array to NumPy."""
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def norm(self, x: Any, axis: Optional[int] = None, ord: Optional[int] = None) -> Any:
        """Vector/matrix norm via JAX."""
        return self._jnp.linalg.norm(x, axis=axis, ord=ord)

    def argsort(self, x: Any, axis: int = -1) -> Any:
        """Indirect sort indices via JAX."""
        return self._jnp.argsort(x, axis=axis)

    def concatenate(self, arrays: Sequence, axis: int = 0) -> Any:
        """Concatenate JAX arrays."""
        return self._jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays: Sequence, axis: int = 0) -> Any:
        """Stack JAX arrays along a new axis."""
        return self._jnp.stack(arrays, axis=axis)

    def linspace(self, start: float, stop: float, num: int) -> Any:
        """Linearly spaced values via JAX."""
        return self._jnp.linspace(start, stop, num, dtype=np.float64)

    def logspace(self, start: float, stop: float, num: int) -> Any:
        """Log-spaced values via JAX."""
        return self._jnp.logspace(start, stop, num, dtype=np.float64)


# ======================================================================
# §3  NUMPYRO GPU BAYESIAN SAMPLER
# ======================================================================

class NumPyroSampler:
    """GPU-accelerated Bayesian MCMC using NumPyro + JAX.

    NumPyro runs NUTS on XLA-compiled JAX graphs, achieving
    substantial speedups over PyMC on GPU for models with many
    parameters (e.g. hierarchical normative models with thousands
    of vertex-level effects).

    Parameters
    ----------
    num_warmup : int
        Warmup (tuning) samples.
    num_samples : int
        Posterior draws.
    num_chains : int
        Independent chains.
    seed : int
        PRNG seed.

    Examples
    --------
    >>> sampler = NumPyroSampler(num_warmup=500, num_samples=2000)
    >>> # Define a NumPyro model function:
    >>> def model(x, y=None):
    ...     alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    ...     sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    ...     mu = alpha * x
    ...     numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    >>> trace = sampler.sample(model, x=x_data, y=y_data)
    """

    name: str = "numpyro"

    def __init__(
        self,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        seed: int = 42,
    ) -> None:
        """Initialise the NumPyro JAX sampler backend."""
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.seed = seed

    def sample(
        self,
        model: Callable,
        **model_kwargs: Any,
    ) -> Any:
        """Run NUTS on a NumPyro model function.

        Parameters
        ----------
        model : callable
            A NumPyro model function.
        **model_kwargs
            Data and hyperparameters passed to *model*.

        Returns
        -------
        numpyro.infer.MCMC
            MCMC object with ``.get_samples()`` and ``.print_summary()``.
        """
        numpyro, infer = _require_numpyro()
        jax, _, _ = _require_jax()

        kernel = infer.NUTS(model)
        mcmc = infer.MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )
        rng_key = jax.random.PRNGKey(self.seed)

        logger.info(
            "NumPyro NUTS: %d draws × %d chains (%d warmup) on %s",
            self.num_samples, self.num_chains, self.num_warmup,
            jax.devices()[0],
        )
        mcmc.run(rng_key, **model_kwargs)
        return mcmc

    def to_arviz(self, mcmc: Any) -> Any:
        """Convert NumPyro MCMC to ArviZ InferenceData.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC

        Returns
        -------
        arviz.InferenceData
        """
        try:
            import arviz as az
            return az.from_numpyro(mcmc)
        except ImportError as exc:
            raise ImportError(
                "ArviZ is required to convert NumPyro traces.\n"
                "  pip install arviz"
            ) from exc


# ======================================================================
# §4  VRAM MANAGEMENT
# ======================================================================

@dataclass
class VRAMInfo:
    """Snapshot of GPU VRAM usage.

    Attributes
    ----------
    device_name : str
        GPU model name.
    device_id : int
        CUDA device index.
    total_gb : float
        Total VRAM.
    used_gb : float
        Currently allocated VRAM.
    free_gb : float
        Available VRAM.
    percent_used : float
        Usage percentage.
    """

    device_name: str
    device_id: int
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float

    def __repr__(self) -> str:
        """Return a human-readable GPU status summary."""
        return (
            f"GPU {self.device_id} ({self.device_name}): "
            f"{self.used_gb:.2f} / {self.total_gb:.2f} GB "
            f"({self.percent_used:.0f}% used, "
            f"{self.free_gb:.2f} GB free)"
        )


def vram_status(device_id: int = 0) -> VRAMInfo:
    """Query current VRAM usage.

    Tries CuPy first, then ``nvidia-smi`` as fallback.

    Parameters
    ----------
    device_id : int
        CUDA device index.

    Returns
    -------
    VRAMInfo
    """
    # Try CuPy (fastest, most accurate).
    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            free, total = cp.cuda.runtime.memGetInfo()
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = props["name"]
        total_gb = total / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        used_gb = total_gb - free_gb
        pct = 100 * used_gb / total_gb if total_gb > 0 else 0
        return VRAMInfo(name, device_id, total_gb, used_gb, free_gb, pct)
    except (ImportError, Exception):
        pass

    # Fallback: nvidia-smi.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True,
        )
        parts = result.stdout.strip().split(", ")
        name = parts[0]
        total_mb, used_mb, free_mb = float(parts[1]), float(parts[2]), float(parts[3])
        total_gb = total_mb / 1024
        used_gb = used_mb / 1024
        free_gb = free_mb / 1024
        pct = 100 * used_gb / total_gb if total_gb > 0 else 0
        return VRAMInfo(name, device_id, total_gb, used_gb, free_gb, pct)
    except (FileNotFoundError, subprocess.CalledProcessError, Exception):
        pass

    logger.warning("Cannot query VRAM: no CuPy and nvidia-smi failed.")
    return VRAMInfo("unknown", device_id, 0, 0, 0, 0)


def vram_clear() -> None:
    """Clear CUDA memory caches across all known GPU frameworks.

    Calls cache-clearing functions for CuPy, JAX, and PyTorch (if
    installed).  Safe to call even when no GPU framework is loaded.
    """
    # CuPy
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        pinned_pool = cp.get_default_pinned_memory_pool()
        pinned_pool.free_all_blocks()
        logger.debug("CuPy memory pool cleared.")
    except (ImportError, Exception):
        pass

    # JAX
    try:
        import jax
        for dev in jax.devices("gpu"):
            # JAX doesn't expose a direct cache-clear API, but
            # deleting references + GC + re-checking frees memory.
            pass
        logger.debug("JAX: no explicit cache clear available.")
    except (ImportError, Exception):
        pass

    # PyTorch (in case someone loaded it)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("PyTorch CUDA cache cleared.")
    except (ImportError, Exception):
        pass


def vram_defrag(device_id: int = 0) -> None:
    """Attempt CUDA memory defragmentation.

    Frees cached blocks and triggers a synchronisation barrier that
    allows the CUDA driver to consolidate fragmented allocations.

    Parameters
    ----------
    device_id : int
        CUDA device index.

    Notes
    -----
    True defragmentation is limited by CUDA's memory model — once an
    allocation is placed, it cannot be moved.  This function does the
    best available: free caches, synchronise, and let the driver
    reclaim contiguous regions.
    """
    vram_clear()

    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            cp.cuda.Stream.null.synchronize()
            pool = cp.get_default_memory_pool()
            pool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
        logger.info("VRAM defrag (CuPy sync + free) on GPU %d", device_id)
    except (ImportError, Exception):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_id)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_id)
            logger.info("VRAM defrag (PyTorch sync + clear) on GPU %d", device_id)
    except (ImportError, Exception):
        pass


def vram_gc(device_id: int = 0) -> None:
    """Full GPU garbage collection: Python GC + VRAM clear + defrag.

    Parameters
    ----------
    device_id : int
        CUDA device index.
    """
    import gc
    gc.collect()
    gc.collect()
    vram_defrag(device_id)
    logger.debug("Full VRAM GC on GPU %d", device_id)


@contextmanager
def vram_guard(
    min_free_gb: float = 1.0,
    device_id: int = 0,
    error_on_low: bool = False,
    auto_clear: bool = True,
) -> Generator[None, None, None]:
    """Context manager for VRAM-safe GPU operations.

    Checks available VRAM before the block.  If below threshold,
    optionally clears caches or raises an error.  Reports delta
    after the block completes.

    Parameters
    ----------
    min_free_gb : float
        Minimum required free VRAM.
    device_id : int
        CUDA device index.
    error_on_low : bool
        Raise ``MemoryError`` if VRAM is insufficient.
    auto_clear : bool
        Attempt to clear caches if VRAM is low.

    Examples
    --------
    >>> with vram_guard(min_free_gb=4.0):
    ...     result = gpu_heavy_computation()
    """
    before = vram_status(device_id)

    if before.free_gb < min_free_gb:
        if auto_clear:
            logger.info(
                "VRAM low (%.2f GB free). Clearing caches…",
                before.free_gb,
            )
            vram_gc(device_id)
            before = vram_status(device_id)

        if before.free_gb < min_free_gb:
            msg = (
                f"Insufficient VRAM: {before.free_gb:.2f} GB free, "
                f"need {min_free_gb:.1f} GB on GPU {device_id} "
                f"({before.device_name})."
            )
            if error_on_low:
                raise MemoryError(msg)
            logger.warning(msg)

    yield

    after = vram_status(device_id)
    delta = before.free_gb - after.free_gb
    if abs(delta) > 0.1:
        logger.info(
            "GPU %d VRAM delta: %+.2f GB (%.2f → %.2f GB free)",
            device_id, -delta, before.free_gb, after.free_gb,
        )


def vram_monitor(device_id: int = 0, label: str = "") -> None:
    """Log current VRAM usage (one-shot, for debugging).

    Parameters
    ----------
    device_id : int
        CUDA device index.
    label : str
        Optional context label for the log message.

    Examples
    --------
    >>> vram_monitor(label="after eigsolve")
    # GPU 0 (RTX 3090): 3.42 / 24.00 GB (14%) [after eigsolve]
    """
    info = vram_status(device_id)
    suffix = f" [{label}]" if label else ""
    logger.info("%s%s", info, suffix)


# ======================================================================
# §5  BACKEND FACTORY
# ======================================================================

def get_gpu_backend(
    name: Literal["cupy", "jax"] = "cupy",
    **kwargs: Any,
) -> Union[CupyBackend, JaxBackend]:
    """Factory for GPU compute backends.

    Parameters
    ----------
    name : ``"cupy"`` or ``"jax"``
    **kwargs
        Passed to the backend constructor.

    Returns
    -------
    CupyBackend or JaxBackend
    """
    if name == "cupy":
        return CupyBackend(**kwargs)
    elif name == "jax":
        return JaxBackend(**kwargs)
    raise ValueError(f"Unknown GPU backend: {name!r}")


def get_gpu_bayesian_sampler(
    **kwargs: Any,
) -> NumPyroSampler:
    """Factory for the GPU Bayesian sampler.

    Returns
    -------
    NumPyroSampler
    """
    return NumPyroSampler(**kwargs)


# ======================================================================
# §6  __all__
# ======================================================================

__all__: List[str] = [
    # Compute backends
    "CupyBackend",
    "JaxBackend",
    "get_gpu_backend",
    # Bayesian sampler
    "NumPyroSampler",
    "get_gpu_bayesian_sampler",
    # VRAM management
    "VRAMInfo",
    "vram_status",
    "vram_clear",
    "vram_defrag",
    "vram_gc",
    "vram_guard",
    "vram_monitor",
]
