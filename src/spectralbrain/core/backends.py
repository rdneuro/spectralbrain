"""Compute backends -- NumPy/SciPy (CPU) and CuPy / JAX / Torch (GPU).

Every backend exposes the same small array-algebra interface plus a
generalised sparse eigensolver ``eigsh(L, M, k)`` used by
:meth:`spectralbrain.core.meshes.BrainMesh.decompose` and by the
operator family in :mod:`spectralbrain.spectral.operators`.

1. **NumpyBackend** -- the default engine (SciPy ARPACK shift-invert).
2. **CupyBackend** -- drop-in GPU replacement using CuPy.
3. **JaxBackend** -- GPU backend with ``jit`` / ``vmap`` for batch work.
4. **TorchBackend** -- GPU backend on PyTorch.
5. :func:`get_gpu_backend` -- factory.

All GPU dependencies are lazy-imported: the module imports cleanly
without them and only instantiating a GPU backend raises ``ImportError``.

Bayesian samplers live in :mod:`spectralbrain.statistics.bayesian`;
RAM/VRAM guards and joblib helpers live in :mod:`spectralbrain.runtime`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spectralbrain.runtime import (
    Eigenvalues,
    Eigenvectors,
    MassMatrix,
    SparseMatrix,
    get_logger,
)

# ======================================================================
# 1  NUMPY / SCIPY COMPUTE BACKEND
# ======================================================================


logger = get_logger(__name__)


class NumpyBackend:
    """CPU compute backend using NumPy + SciPy.

    Provides the canonical interface that :class:`CupyBackend` and
    :class:`JaxBackend` mirror.  All ``core/`` and ``spectral/``
    modules call backend methods rather than importing NumPy or SciPy
    directly, enabling transparent GPU acceleration.

    Examples
    --------
    >>> from spectralbrain.core.backends import NumpyBackend
    >>> be = NumpyBackend()
    >>> evals, evecs = be.eigsh(L, M, k=100)
    >>> hks = be.exp(-evals[None, :] * t[:, None])  # broadcasting
    """

    name: str = "numpy"

    # -- Sparse eigensolvers -------------------------------------------

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
        """Solve the generalised sparse eigenproblem L v = lambda M v.

        Uses SciPy's ARPACK wrapper in shift-invert mode (default
        sigma = -0.01) which is optimal for computing the *smallest*
        eigenvalues of the Laplacian.

        Parameters
        ----------
        L : sparse matrix, shape (N, N)
            Stiffness (Laplacian) matrix -- symmetric positive
            semi-definite.
        M : sparse matrix, shape (N, N), optional
            Mass matrix.  If ``None``, the standard eigenproblem
            L v = lambda v is solved.
        k : int
            Number of eigenpairs to compute.
        sigma : float
            Shift for shift-invert mode.  A small negative value
            avoids the singularity at lambda = 0.
        which : str
            Which eigenvalues to target (``"LM"`` = largest magnitude
            *of the shifted operator*, yielding the smallest lambda).
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

    # -- Sparse matrix construction ------------------------------------

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

    # -- Dense array operations ----------------------------------------
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
# 2  GPU LAZY IMPORTS AND SHARED EIGENSOLVER HELPERS
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
            "CuPy is required for the CuPy GPU backend.\n  pip install cupy-cuda13x"
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
            "JAX is required for the JAX GPU backend.\n  pip install 'jax[cuda13]' jaxlib"
        ) from exc


def _require_torch():
    """Lazy-import PyTorch, raising ImportError if unavailable."""
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the Torch GPU backend.\n  pip install torch"
        ) from exc


_EIGSH_DEFAULTS: dict[str, Any] = {"sigma": -0.01, "which": "LM", "tol": 0.0, "maxiter": None}


def _is_diagonal(M: Any) -> bool:
    """True if sparse/dense matrix *M* has no off-diagonal non-zeros."""
    if sp.issparse(M):
        C = sp.coo_matrix(M)
        off = C.row != C.col
        return not np.any(C.data[off] != 0)
    A = np.asarray(M)
    return bool(np.count_nonzero(A - np.diag(np.diag(A))) == 0)


def _cpu_fallback_reason(N: int, M: Any, dense_max: int) -> str | None:
    """Why the dense-GPU path cannot be used (``None`` if it can)."""
    if N > dense_max:
        return f"N={N} > dense_max={dense_max} (dense eigh would exhaust memory)"
    if M is not None and not _is_diagonal(M):
        return "the mass matrix is not diagonal (diagonal standardisation would be wrong)"
    return None


def _scipy_shift_invert(
    L: Any, M: Any, k: int, sigma: float, which: str, tol: float, maxiter: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """CPU ARPACK shift-invert eigensolve (fallback path)."""
    from scipy.sparse.linalg import eigsh as _scipy_eigsh

    return _scipy_eigsh(
        sp.csc_matrix(L, dtype=np.float64),
        M=(sp.csc_matrix(M, dtype=np.float64) if M is not None else None),
        k=k,
        sigma=sigma,
        which=which,
        tol=tol,
        maxiter=maxiter,
    )


def _warn_ignored_eigsh_args(backend: str, **given: Any) -> None:
    """Warn when shift-invert options are passed to the dense GPU path."""
    ignored = [n for n, v in given.items() if v != _EIGSH_DEFAULTS[n]]
    if ignored:
        logger.warning(
            "%s.eigsh: %s ignored on the dense GPU path (all eigenpairs are "
            "computed exactly; the k smallest are returned).",
            backend,
            ", ".join(ignored),
        )


# ======================================================================
# 3  CUPY BACKEND
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
    >>> type(evals)  # cupy.ndarray -- lives on GPU
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
            device_id,
            self._cp.cuda.runtime.getDeviceProperties(device_id)["name"],
        )

    # -- Sparse eigensolver --------------------------------------------

    def eigsh(
        self,
        L: SparseMatrix,
        M: MassMatrix | None = None,
        k: int = 100,
        *,
        sigma: float = -0.01,
        which: str = "LM",
        tol: float = 0.0,
        maxiter: int | None = None,
        dense_max: int = 20000,
    ) -> tuple[Eigenvalues, Eigenvectors]:
        """Smallest-k generalised eigenpairs ``L v = lambda M v`` on the GPU.

        CuPy's sparse ``eigsh`` supports neither the generalised problem (no
        ``M``) nor shift-invert (no ``sigma``), and recovering the *smallest*
        Laplacian eigenvalues by plain Lanczos is unreliable.  Because the FEM
        mass matrix is **diagonal** (lumped barycentric), the generalised
        problem standardises exactly to a symmetric one:

            A~ = D^{-1/2} L D^{-1/2},   D = diag(M),   psi = D^{1/2} v

        We solve the *dense* symmetric eigenproblem ``A~ psi = lambda psi`` on the GPU
        (``cupy.linalg.eigh`` -- robust, no ARPACK convergence issues), keep the
        ``k`` smallest, and recover the M-orthonormal eigenvectors
        ``v = D^{-1/2} psi``.  Validated against SciPy shift-invert and the
        analytic sphere spectrum.  Meshes with ``N > dense_max`` (or a
        non-diagonal ``M``) fall back to CPU sparse shift-invert -- a warning
        is logged.  ``sigma``/``which``/``tol``/``maxiter`` are honoured only
        on the fallback path (a warning is logged if they are set on the
        dense path); the signature mirrors :meth:`NumpyBackend.eigsh`.
        Returns **host** arrays.
        """
        cp = self._cp
        N = L.shape[0]
        d = (
            np.asarray(M.diagonal(), dtype=np.float64)
            if M is not None
            else np.ones(N, dtype=np.float64)
        )
        d = np.clip(d, 1e-20, None)
        dinv_sqrt = 1.0 / np.sqrt(d)

        reason = _cpu_fallback_reason(N, M, dense_max)
        if reason is not None:
            logger.warning(
                "CupyBackend.eigsh: %s -- falling back to CPU SciPy shift-invert "
                "(ARPACK); this solve does NOT run on the GPU.",
                reason,
            )
            evals, evecs = _scipy_shift_invert(L, M, k, sigma, which, tol, maxiter)
        else:
            _warn_ignored_eigsh_args(
                "CupyBackend", sigma=sigma, which=which, tol=tol, maxiter=maxiter
            )
            # Standardise on host, dense symmetric eigh on the GPU.
            A = sp.csr_matrix(L).astype(np.float64).toarray()
            A *= dinv_sqrt[:, None]
            A *= dinv_sqrt[None, :]
            A = 0.5 * (A + A.T)  # guard fp asymmetry
            A_gpu = cp.asarray(A)
            w_gpu, V_gpu = cp.linalg.eigh(A_gpu)  # ascending, orthonormal
            idx = cp.argsort(w_gpu)[:k]
            evals = cp.asnumpy(w_gpu[idx])
            psi = cp.asnumpy(V_gpu[:, idx])
            evecs = dinv_sqrt[:, None] * psi  # v = D^{-1/2} psi (M-orthonormal)
            del A_gpu, w_gpu, V_gpu
            self._cp.get_default_memory_pool().free_all_blocks()

        # Sort ascending, clamp tiny negatives from round-off.
        order = np.argsort(evals)
        evals = np.clip(np.asarray(evals)[order], 0.0, None)
        evecs = np.asarray(evecs)[:, order]
        return evals, evecs

    # -- Sparse matrix -------------------------------------------------

    def sparse_matrix(
        self,
        data: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
        shape: tuple[int, int],
        **kwargs: Any,
    ) -> Any:
        """Build a sparse matrix from COO triplets on GPU."""
        cp = self._cp
        return self._cpsp.coo_matrix(
            (
                cp.asarray(data, dtype=cp.float64),
                (cp.asarray(row, dtype=cp.int64), cp.asarray(col, dtype=cp.int64)),
            ),
            shape=shape,
        ).tocsc()

    # -- Dense ops (GPU arrays) ----------------------------------------

    def array(self, data: Any, dtype: Any = np.float64) -> Any:
        """Create a CuPy array on GPU."""
        return self._cp.asarray(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a zero-filled CuPy array."""
        return self._cp.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any = np.float64) -> Any:
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

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum reduction on GPU."""
        return self._cp.sum(x, axis=axis)

    def mean(self, x: Any, axis: int | None = None) -> Any:
        """Mean reduction on GPU."""
        return self._cp.mean(x, axis=axis)

    def clip(self, x: Any, a_min: float | None, a_max: float | None) -> Any:
        """Element-wise clip on GPU."""
        return self._cp.clip(x, a_min, a_max)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Copy GPU array to host."""
        if isinstance(x, np.ndarray):
            return x
        return self._cp.asnumpy(x)

    def norm(self, x: Any, axis: int | None = None, ord: int | None = None) -> Any:
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
# 4  JAX BACKEND
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

        # In modern JAX, ``jax.devices("gpu")`` raises a RuntimeError
        # (rather than returning an empty list) when no GPU platform is
        # present, so probe defensively and fall back to CPU instead of
        # crashing.  This keeps the backend usable on CPU-only installs.
        self.device = device
        if device == "gpu":
            try:
                has_gpu = bool(jax.devices("gpu"))
            except RuntimeError:
                has_gpu = False
            if not has_gpu:
                logger.warning("No GPU found; JAX backend using CPU.")
                self.device = "cpu"
        # Concrete device that every array created by this backend lives on.
        self._device = jax.devices(self.device)[0]

        # Without x64, JAX silently downcasts every float64 request to
        # float32 (precision loss; 1e-300 clamps underflow to 0).
        try:
            x64 = bool(jax.config.read("jax_enable_x64"))
        except Exception:  # pragma: no cover - very old/new config API
            x64 = bool(getattr(jax.config, "jax_enable_x64", False))
        self.x64 = x64
        if not x64:
            logger.warning(
                "JAX x64 mode is disabled: float64 arrays are computed in float32. "
                "Call jax.config.update('jax_enable_x64', True) before creating "
                "arrays for double precision."
            )
        logger.info("JAX backend: device = %s (all: %s)", self._device, jax.devices())

    def _put(self, x: Any) -> Any:
        """Place *x* on this backend's device."""
        return self._jax.device_put(x, self._device)

    # -- Eigensolver ---------------------------------------------------

    def eigsh(
        self,
        L: SparseMatrix,
        M: MassMatrix | None = None,
        k: int = 100,
        **kwargs: Any,
    ) -> tuple[Eigenvalues, Eigenvectors]:
        """Sparse eigensolver via JAX's LOBPCG.

        For the generalised problem L v = lambda M v, falls back to
        SciPy ARPACK on host and transfers results -- JAX's sparse
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
        which = kwargs.pop("which", "LM")
        tol = kwargs.pop("tol", 0.0)
        maxiter = kwargs.pop("maxiter", None)
        if kwargs:
            raise TypeError(f"JaxBackend.eigsh got unexpected arguments: {sorted(kwargs)}")
        evals, evecs = spla.eigsh(
            L_sp,
            k=k,
            M=M_sp,
            sigma=sigma,
            which=which,
            tol=tol,
            maxiter=maxiter,
        )
        order = np.argsort(evals)
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]
        return evals, evecs

    # -- JIT / VMAP helpers --------------------------------------------

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
        >>> # Single-subject HKS: (k,), (N, k), (T,) -> (N, T)
        >>> batched = be.vmap(compute_hks)
        >>> # Now: (S, k), (S, N, k), (T,) -> (S, N, T)
        >>> all_hks = batched(all_evals, all_evecs, t_values)
        """
        return self._jax.vmap(func, in_axes=in_axes, out_axes=out_axes)

    # -- Dense array ops -----------------------------------------------

    def array(self, data: Any, dtype: Any = np.float64) -> Any:
        """Create a JAX array."""
        return self._put(self._jnp.asarray(data, dtype=dtype))

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a zero-filled JAX array."""
        return self._put(self._jnp.zeros(shape, dtype=dtype))

    def ones(self, shape: tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a ones-filled JAX array."""
        return self._put(self._jnp.ones(shape, dtype=dtype))

    def eye(self, n: int, dtype: Any = np.float64) -> Any:
        """Create a JAX identity matrix."""
        return self._put(self._jnp.eye(n, dtype=dtype))

    def matmul(self, a: Any, b: Any) -> Any:
        """JAX matrix multiply."""
        return self._jnp.matmul(a, b)

    def exp(self, x: Any) -> Any:
        """Element-wise exponential via JAX."""
        return self._jnp.exp(x)

    def log(self, x: Any) -> Any:
        """Element-wise safe log via JAX (clamp at the dtype's smallest normal)."""
        x = self._jnp.asarray(x)
        dt = x.dtype if self._jnp.issubdtype(x.dtype, self._jnp.floating) else self._jnp.float32
        tiny = float(self._jnp.finfo(dt).tiny)
        return self._jnp.log(self._jnp.clip(x, tiny, None))

    def sqrt(self, x: Any) -> Any:
        """Element-wise safe sqrt via JAX."""
        return self._jnp.sqrt(self._jnp.clip(x, 0.0, None))

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum reduction via JAX."""
        return self._jnp.sum(x, axis=axis)

    def mean(self, x: Any, axis: int | None = None) -> Any:
        """Mean reduction via JAX."""
        return self._jnp.mean(x, axis=axis)

    def clip(self, x: Any, a_min: float | None, a_max: float | None) -> Any:
        """Element-wise clip via JAX."""
        return self._jnp.clip(x, a_min, a_max)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Transfer JAX array to NumPy."""
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def norm(self, x: Any, axis: int | None = None, ord: int | None = None) -> Any:
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
        return self._put(self._jnp.linspace(start, stop, num, dtype=np.float64))

    def logspace(self, start: float, stop: float, num: int) -> Any:
        """Log-spaced values via JAX."""
        return self._put(self._jnp.logspace(start, stop, num, dtype=np.float64))


# ======================================================================
# 5  TORCH BACKEND
# ======================================================================


class TorchBackend:
    """GPU compute backend using PyTorch.

    Mirrors the :class:`NumpyBackend` interface so it can be passed to
    :meth:`BrainMesh.decompose` via ``backend=``.  Dense ops run as Torch
    tensors on the selected device; :meth:`to_numpy` copies back to host.

    PyTorch has no robust sparse *generalised* eigensolver, so
    :meth:`eigsh` uses the same diagonal-mass standardisation as
    :class:`CupyBackend` -- ``A~ = D^{-1/2} L D^{-1/2}`` solved with the
    dense ``torch.linalg.eigh`` on the device -- and falls back to CPU
    sparse shift-invert for meshes above ``dense_max``.

    Parameters
    ----------
    device : str
        ``"cuda"`` or ``"cpu"``.  If ``"cuda"`` is requested but no GPU
        is available, the backend falls back to CPU.

    Examples
    --------
    >>> be = TorchBackend()
    >>> evals, evecs = be.eigsh(L, M, k=100)  # host NumPy arrays
    """

    name: str = "torch"

    def __init__(self, device: str = "cuda") -> None:
        """Initialise the Torch backend, falling back to CPU if needed."""
        torch = _require_torch()
        self._torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("No CUDA device found; Torch backend using CPU.")
            device = "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(self.device)
            logger.info("Torch backend on %s (%s)", self.device, name)
        else:
            logger.info("Torch backend on %s", self.device)

    # -- Sparse eigensolver --------------------------------------------

    def eigsh(
        self,
        L: SparseMatrix,
        M: MassMatrix | None = None,
        k: int = 100,
        *,
        sigma: float = -0.01,
        which: str = "LM",
        tol: float = 0.0,
        maxiter: int | None = None,
        dense_max: int = 20000,
    ) -> tuple[Eigenvalues, Eigenvectors]:
        """Smallest-k generalised eigenpairs ``L v = lambda M v`` on the device.

        Uses the diagonal-mass standardisation ``A~ = D^{-1/2} L D^{-1/2}``
        with a dense ``torch.linalg.eigh`` on the device, keeps the ``k``
        smallest, and recovers the M-orthonormal eigenvectors
        ``v = D^{-1/2} psi``.  Meshes with ``N > dense_max`` (or a non-diagonal
        ``M``) fall back to CPU sparse shift-invert with a logged warning.
        ``sigma``/``which``/``tol``/``maxiter`` are honoured only on that
        fallback path (a warning is logged otherwise).  Returns **host**
        (NumPy) arrays, matching :meth:`NumpyBackend.eigsh`.
        """
        torch = self._torch
        N = L.shape[0]
        d = (
            np.asarray(M.diagonal(), dtype=np.float64)
            if M is not None
            else np.ones(N, dtype=np.float64)
        )
        d = np.clip(d, 1e-20, None)
        dinv_sqrt = 1.0 / np.sqrt(d)

        reason = _cpu_fallback_reason(N, M, dense_max)
        if reason is not None:
            logger.warning(
                "TorchBackend.eigsh: %s -- falling back to CPU SciPy shift-invert "
                "(ARPACK); this solve does NOT run on %s.",
                reason,
                self.device,
            )
            evals, evecs = _scipy_shift_invert(L, M, k, sigma, which, tol, maxiter)
        else:
            _warn_ignored_eigsh_args(
                "TorchBackend", sigma=sigma, which=which, tol=tol, maxiter=maxiter
            )
            # Standardise on host, dense symmetric eigh on the device.
            A = sp.csr_matrix(L).astype(np.float64).toarray()
            A *= dinv_sqrt[:, None]
            A *= dinv_sqrt[None, :]
            A = 0.5 * (A + A.T)  # guard fp asymmetry
            A_t = torch.as_tensor(A, dtype=torch.float64, device=self.device)
            w_t, V_t = torch.linalg.eigh(A_t)  # ascending, orthonormal
            idx = torch.argsort(w_t)[:k]
            evals = w_t[idx].detach().cpu().numpy()
            psi = V_t[:, idx].detach().cpu().numpy()
            evecs = dinv_sqrt[:, None] * psi  # v = D^{-1/2} psi (M-orthonormal)
            del A_t, w_t, V_t
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Sort ascending, clamp tiny negatives from round-off.
        order = np.argsort(evals)
        evals = np.clip(np.asarray(evals)[order], 0.0, None)
        evecs = np.asarray(evecs)[:, order]
        return evals, evecs

    # -- Sparse matrix -------------------------------------------------

    def sparse_matrix(
        self,
        data: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
        shape: tuple[int, int],
        **kwargs: Any,
    ) -> Any:
        """Build a sparse COO tensor from triplets on the device."""
        torch = self._torch
        indices = torch.as_tensor(np.vstack([row, col]), dtype=torch.int64, device=self.device)
        values = torch.as_tensor(data, dtype=torch.float64, device=self.device)
        return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()

    # -- Dense ops (device tensors) ------------------------------------

    def _tdtype(self, dtype: Any) -> Any:
        """Map a NumPy/Torch dtype to the corresponding Torch dtype."""
        torch = self._torch
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.from_numpy(np.empty(0, dtype=np.dtype(dtype))).dtype

    def array(self, data: Any, dtype: Any = np.float64) -> Any:
        """Create a Torch tensor on the device."""
        return self._torch.as_tensor(np.asarray(data, dtype=dtype), device=self.device)

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a zero-filled tensor."""
        return self._torch.zeros(shape, dtype=self._tdtype(dtype), device=self.device)

    def ones(self, shape: tuple[int, ...], dtype: Any = np.float64) -> Any:
        """Create a ones-filled tensor."""
        return self._torch.ones(shape, dtype=self._tdtype(dtype), device=self.device)

    def eye(self, n: int, dtype: Any = np.float64) -> Any:
        """Create an identity tensor."""
        return self._torch.eye(n, dtype=self._tdtype(dtype), device=self.device)

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiply."""
        return a @ b

    def exp(self, x: Any) -> Any:
        """Element-wise exponential."""
        return self._torch.exp(x)

    def log(self, x: Any) -> Any:
        """Element-wise safe log (clamp at the dtype's smallest normal)."""
        tiny = self._torch.finfo(x.dtype).tiny if x.is_floating_point() else 1e-300
        return self._torch.log(self._torch.clamp(x, min=tiny))

    def sqrt(self, x: Any) -> Any:
        """Element-wise safe sqrt."""
        return self._torch.sqrt(self._torch.clamp(x, min=0.0))

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum reduction."""
        return self._torch.sum(x) if axis is None else self._torch.sum(x, dim=axis)

    def mean(self, x: Any, axis: int | None = None) -> Any:
        """Mean reduction."""
        return self._torch.mean(x) if axis is None else self._torch.mean(x, dim=axis)

    def clip(self, x: Any, a_min: float | None, a_max: float | None) -> Any:
        """Element-wise clamp."""
        return self._torch.clamp(x, min=a_min, max=a_max)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Copy a device tensor to host."""
        if isinstance(x, np.ndarray):
            return x
        if self._torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def norm(self, x: Any, axis: int | None = None, ord: int | None = None) -> Any:
        """Vector/matrix norm."""
        return self._torch.linalg.norm(x, ord=ord, dim=axis)

    def argsort(self, x: Any, axis: int = -1) -> Any:
        """Indices that sort the tensor."""
        return self._torch.argsort(x, dim=axis)

    def concatenate(self, arrays: Sequence, axis: int = 0) -> Any:
        """Concatenate along an existing axis."""
        return self._torch.cat(list(arrays), dim=axis)

    def stack(self, arrays: Sequence, axis: int = 0) -> Any:
        """Stack along a new axis."""
        return self._torch.stack(list(arrays), dim=axis)

    def linspace(self, start: float, stop: float, num: int) -> Any:
        """Evenly spaced values."""
        return self._torch.linspace(start, stop, num, dtype=self._torch.float64, device=self.device)

    def logspace(self, start: float, stop: float, num: int) -> Any:
        """Log-spaced values."""
        return self._torch.logspace(start, stop, num, dtype=self._torch.float64, device=self.device)

    # -- JIT / VMAP helpers --------------------------------------------

    def jit(self, func: Callable, **kwargs: Any) -> Callable:
        """Compile a function with ``torch.compile`` (Torch-native ops only)."""
        return self._torch.compile(func, **kwargs)

    def vmap(self, func: Callable, **kwargs: Any) -> Callable:
        """Vectorise a function with ``torch.func.vmap`` (Torch-native ops only)."""
        return self._torch.func.vmap(func, **kwargs)


# ======================================================================
# 6  BACKEND FACTORY
# ======================================================================


def get_gpu_backend(
    name: Literal["cupy", "jax", "torch"] = "cupy",
    **kwargs: Any,
) -> CupyBackend | JaxBackend | TorchBackend:
    """Factory for GPU compute backends.

    Parameters
    ----------
    name : ``"cupy"``, ``"jax"``, or ``"torch"``
    **kwargs
        Passed to the backend constructor.

    Returns
    -------
    CupyBackend, JaxBackend, or TorchBackend
    """
    if name == "cupy":
        return CupyBackend(**kwargs)
    elif name == "jax":
        return JaxBackend(**kwargs)
    elif name == "torch":
        return TorchBackend(**kwargs)
    raise ValueError(f"Unknown GPU backend: {name!r}")


__all__: list[str] = [
    "CupyBackend",
    "JaxBackend",
    "NumpyBackend",
    "TorchBackend",
    "get_gpu_backend",
]
