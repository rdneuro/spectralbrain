"""Spectral Graph Wavelets (SGW) for multi-resolution shape analysis.

Implements the wavelet framework of Hammond, Vandergheynst &
Gribonval (ACHA 2011): define a band-pass filter g on the spectral
domain and apply it via Chebyshev polynomial approximation of
g(t·L), **without** explicit eigendecomposition.

Also supports wavelet computation from precomputed eigenpairs (faster
for the same decomposition used by HKS/WKS).

Kernels
-------
- **Mexican hat** — g(x) = x · exp(-x) — the canonical SGW kernel.
- **Heat** — g(x) = exp(-x) — low-pass (same filter as HKS).
- **Meyer** — compactly supported in spectral domain.
- **Custom** — user-defined callable g(x).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    SparseMatrix,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  WAVELET KERNELS
# ======================================================================


def mexican_hat_kernel(x: np.ndarray) -> np.ndarray:
    """Mexican-hat (Ricker) wavelet kernel: g(x) = x · exp(-x).

    Parameters
    ----------
    x : ndarray
        Scaled spectral variable t·λ.

    Returns
    -------
    ndarray
    """
    return x * np.exp(-x)


def heat_kernel(x: np.ndarray) -> np.ndarray:
    """Heat kernel: g(x) = exp(-x).

    Low-pass filter — equivalent to the HKS filter but used here
    in the wavelet framework for completeness.
    """
    return np.exp(-x)


def meyer_kernel(x: np.ndarray) -> np.ndarray:
    """Simplified Meyer-type wavelet kernel.

    Compactly supported band-pass: peaks at x ≈ 1, decays to zero
    at x = 0 and x → ∞.
    """
    v = np.zeros_like(x)
    mask1 = (x >= 0.5) & (x < 1.0)
    mask2 = (x >= 1.0) & (x < 2.0)
    v[mask1] = np.sin(np.pi / 2 * _nu(2 * x[mask1] - 1)) ** 2
    v[mask2] = np.cos(np.pi / 2 * _nu(x[mask2] - 1)) ** 2
    return v


def _nu(x: np.ndarray) -> np.ndarray:
    """Smooth transition function for Meyer wavelet."""
    return x**4 * (35 - 84 * x + 70 * x**2 - 20 * x**3)


# ======================================================================
# §2  CHEBYSHEV APPROXIMATION OF g(t·L)
# ======================================================================


def _chebyshev_coefficients(
    kernel: Callable[[np.ndarray], np.ndarray],
    K: int,
    *,
    a: float = 0.0,
    b: float = 2.0,
) -> np.ndarray:
    """Compute Chebyshev coefficients for a kernel on [a, b].

    Uses the discrete cosine transform of kernel samples at
    Chebyshev nodes.

    Parameters
    ----------
    kernel : callable
        g(x) → y, defined on [a, b].
    K : int
        Number of Chebyshev coefficients (polynomial order + 1).
    a, b : float
        Interval bounds.

    Returns
    -------
    ndarray, shape (K,)
    """
    N = max(K + 1, 2 * K)
    nodes = np.cos(np.pi * (np.arange(N) + 0.5) / N)  # in [-1, 1]
    x = (nodes + 1) * (b - a) / 2 + a  # map to [a, b]
    vals = kernel(x)

    # DCT-based coefficient estimation.
    coeffs = np.zeros(K, dtype=np.float64)
    for k in range(K):
        coeffs[k] = (2.0 / N) * np.sum(vals * np.cos(np.pi * k * (np.arange(N) + 0.5) / N))
    coeffs[0] /= 2.0
    return coeffs


def _chebyshev_apply(
    L: SparseMatrix,
    signal: np.ndarray,
    coeffs: np.ndarray,
    *,
    a: float = 0.0,
    b: float = 2.0,
) -> np.ndarray:
    """Apply Chebyshev polynomial approximation of g(L) to a signal.

    Uses the three-term recurrence T_{k+1}(x) = 2x·T_k(x) − T_{k-1}(x).

    Parameters
    ----------
    L : SparseMatrix, shape (N, N), or callable
        The operator, or a function ``x ↦ L x`` (e.g. ``M⁻¹ L x``).
    signal : ndarray, shape (N,) or (N, d)
    coeffs : ndarray, shape (K,)
    a, b : float
        Spectral bounds.

    Returns
    -------
    ndarray, same shape as signal
    """
    K = len(coeffs)
    apply_L = L if callable(L) else (lambda x: L @ x)

    # Scale L to [-1, 1]: L̃ = (2L − (a+b)I) / (b−a)
    c = 2.0 / (b - a)
    d = -(a + b) / (b - a)

    # T_0 = I · signal
    T_prev = signal.copy()  # T_0(L̃) · s
    result = coeffs[0] * T_prev

    if K == 1:
        return result

    # T_1 = L̃ · signal
    T_curr = c * apply_L(signal) + d * signal
    result = result + coeffs[1] * T_curr

    # Recurrence.
    for k in range(2, K):
        T_next = 2.0 * (c * apply_L(T_curr) + d * T_curr) - T_prev
        result = result + coeffs[k] * T_next
        T_prev = T_curr
        T_curr = T_next

    return result


# ======================================================================
# §3  SPECTRAL GRAPH WAVELET TRANSFORM
# ======================================================================


def sgw_transform(
    L: SparseMatrix,
    scales: np.ndarray,
    *,
    signal: np.ndarray | None = None,
    kernel: Callable = mexican_hat_kernel,
    chebyshev_order: int = 30,
    lam_max: float | None = None,
    mass: SparseMatrix | None = None,
) -> np.ndarray:
    """Spectral Graph Wavelet Transform via Chebyshev approximation.

    Computes T_g^{t_j} f = g(t_j · Δ) · f for each scale t_j,
    **without** eigendecomposition, where Δ = M⁻¹L is the generalised
    (FEM) Laplace–Beltrami operator when *mass* is given, or Δ = L for a
    graph Laplacian.

    Parameters
    ----------
    L : SparseMatrix, shape (N, N)
        Laplacian (stiffness matrix).  For a FEM cotangent stiffness also
        pass *mass*; otherwise the stiffness alone is not the LBO and its
        spectrum (hence the wavelet scales) is meaningless.
    scales : ndarray, shape (S,)
        Wavelet scales (positive reals).
    signal : ndarray, shape (N,) or (N, d), optional
        Signal to transform.  Default = identity (delta at each
        vertex) — gives the wavelet coefficient matrix.
    kernel : callable
        Wavelet kernel g(x).  Default: Mexican hat.
    chebyshev_order : int
        Polynomial approximation order (higher = more accurate).
    lam_max : float, optional
        Upper bound on eigenvalues of Δ.  ``None`` = estimate via
        ARPACK (generalised problem ``L v = λ M v`` when *mass* is given).
    mass : SparseMatrix, shape (N, N), optional
        Mass matrix M.  When given, the wavelets are those of the
        generalised operator ``M⁻¹ L`` (the discrete LBO).

    Returns
    -------
    ndarray, shape (S, N) or (S, N, d)
        Wavelet coefficients at each scale.

    References
    ----------
    Hammond DK, Vandergheynst P, Gribonval R. Wavelets on graphs via
    spectral graph theory. *ACHA* 30(2):129–150, 2011.
    """
    N = L.shape[0]
    S = len(scales)

    if mass is None:
        apply_L = L
    else:
        M = sp.csc_matrix(mass, dtype=np.float64)
        m_diag = M.diagonal()
        if (M - sp.diags(m_diag)).count_nonzero() == 0:
            inv_m = 1.0 / np.clip(m_diag, 1e-300, None)

            def apply_L(x: np.ndarray, _L=L, _im=inv_m) -> np.ndarray:
                """Apply M⁻¹L for a diagonal (lumped) mass matrix."""
                y = _L @ x
                return y * (_im if y.ndim == 1 else _im[:, None])
        else:
            from scipy.sparse.linalg import factorized

            solve_m = factorized(M)

            def apply_L(x: np.ndarray, _L=L) -> np.ndarray:
                """Apply M⁻¹L for a consistent (non-diagonal) mass matrix."""
                y = _L @ x
                if y.ndim == 1:
                    return solve_m(y)
                return np.column_stack([solve_m(y[:, j]) for j in range(y.shape[1])])

    # Estimate λ_max if not provided.
    if lam_max is None:
        from scipy.sparse.linalg import eigsh

        if mass is None:
            lam_max = float(eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
        else:
            lam_max = float(
                eigsh(L, k=1, M=sp.csc_matrix(mass), which="LM", return_eigenvectors=False)[0]
            )
        lam_max *= 1.05  # safety margin

    if signal is None:
        if N > 20000:
            logger.warning(
                "sgw_transform(signal=None) builds a dense (S, N, N) array: "
                "N=%d → %.1f GB per scale.",
                N,
                N * N * 8 / 1e9,
            )
        signal = sp.eye(N, format="csc")

    is_sparse_signal = sp.issparse(signal)

    results: list[np.ndarray] = []
    with progress_simple("SGW transform", total=S) as tick:
        for _s_idx, t in enumerate(scales):
            # Scaled kernel: g_t(x) = g(t · x)
            def scaled_kernel(x: np.ndarray, _t=t) -> np.ndarray:
                """Evaluate the wavelet kernel scaled to a given level."""
                return kernel(_t * x)

            coeffs = _chebyshev_coefficients(
                scaled_kernel,
                chebyshev_order,
                a=0.0,
                b=lam_max,
            )

            if is_sparse_signal:
                # Process column by column for sparse identity.
                out = np.zeros((N, N), dtype=np.float64)
                for col in range(N):
                    e_col = np.zeros(N)
                    e_col[col] = 1.0
                    out[:, col] = _chebyshev_apply(
                        apply_L,
                        e_col,
                        coeffs,
                        a=0.0,
                        b=lam_max,
                    )
                results.append(out)
            else:
                out = _chebyshev_apply(
                    apply_L,
                    signal,
                    coeffs,
                    a=0.0,
                    b=lam_max,
                )
                results.append(out)

            tick(1)

    return np.stack(results, axis=0)  # (S, N, ...)


# ======================================================================
# §4  WAVELET DESCRIPTORS FROM EIGENPAIRS
# ======================================================================


def sgw_descriptor(
    decomp: SpectralDecomposition,
    scales: np.ndarray | None = None,
    *,
    n_scales: int = 5,
    kernel: Callable = mexican_hat_kernel,
    aggregate: Literal["energy", "raw", "abs_mean"] = "energy",
) -> DescriptorMatrix:
    """Spectral Graph Wavelet descriptor from precomputed eigenpairs.

    Faster than Chebyshev-based SGW when the eigenpairs are already
    available (from HKS/WKS computation).

    The descriptor is the wavelet *centred at x* evaluated at x (the
    diagonal of the wavelet operator, cf. HKS = diagonal of the heat
    kernel):

    .. math::

        \\psi_{t}(x) = \\psi_{t,x}(x) = \\sum_{i=0}^{k}
            g(t \\cdot \\lambda_i)\\, \\varphi_i(x)^2

    It depends on the eigenfunctions only through φ², so it is invariant
    to their arbitrary signs.  The per-vertex wavelet energy at scale *t*
    is W(x, t) = ψ_t(x)².

    Parameters
    ----------
    decomp : SpectralDecomposition
    scales : ndarray, shape (S,), optional
        Wavelet scales.  ``None`` = auto log-spaced.
    n_scales : int
        Number of auto scales.
    kernel : callable
        Wavelet kernel g(x).
    aggregate : str
        ``"energy"`` — ψ²(x, t), wavelet energy per vertex per scale.
        ``"raw"`` — ψ(x, t), raw wavelet coefficients (signed when g is).
        ``"abs_mean"`` — |ψ(x, t)|, absolute coefficients.

    Returns
    -------
    ndarray, shape (N, S)
        Multi-scale wavelet descriptor.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    if scales is None:
        lam_nz = evals[evals > 1e-10]
        if len(lam_nz) < 2:
            scales = np.logspace(-1, 2, n_scales)
        else:
            s_min = 1.0 / lam_nz[-1]
            s_max = 2.0 / lam_nz[0]
            scales = np.logspace(np.log10(s_min), np.log10(s_max), n_scales)

    scales = np.asarray(scales, dtype=np.float64)

    # g(t·λ) for each scale: (S, k)
    g_tl = np.array([kernel(t * evals) for t in scales])  # (S, k)

    # ψ_{t,x}(x) = Σᵢ g(t·λᵢ)·φᵢ(x)² = Φ² @ g_tl.T  (sign-invariant)
    psi = (evecs**2) @ g_tl.T  # (N, S)

    if aggregate == "energy":
        return psi**2
    elif aggregate == "abs_mean":
        return np.abs(psi)
    elif aggregate == "raw":
        return psi
    else:
        raise ValueError(f"Unknown aggregate: {aggregate!r}")


# ======================================================================

__all__: list[str] = [
    "heat_kernel",
    # Kernels
    "mexican_hat_kernel",
    "meyer_kernel",
    # Eigenpair-based
    "sgw_descriptor",
    # Chebyshev
    "sgw_transform",
]
