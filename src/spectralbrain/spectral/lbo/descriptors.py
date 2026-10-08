"""Spectral shape descriptors derived from the LBO eigenpairs.

Every function in this module consumes a
:class:`~spectralbrain.core.base.SpectralDecomposition` and produces
a :pydata:`ScalarMap` ``(N,)``, :pydata:`DescriptorMatrix` ``(N, T)``,
or :pydata:`GlobalDescriptor` ``(d,)``.

Once the eigendecomposition is computed (the expensive step), all
descriptors are algebraically cheap -- just array operations on
``eigenvalues`` and ``eigenvectors``.

Implemented descriptors
-----------------------
1. **ShapeDNA** -- eigenvalue fingerprint (Reuter et al. 2006)
2. **HKS** -- Heat Kernel Signature (Sun, Ovsjanikov & Guibas 2009)
3. **SI-HKS** -- Scale-Invariant HKS (Bronstein & Kokkinos 2010)
4. **WKS** -- Wave Kernel Signature (Aubry, Schlickewei & Cremers 2011)
5. **GPS** -- Global Point Signature (Rustamov 2007)
6. **Bates SP** -- Symmetric Polynomial Signatures (Bates et al. 2011)
7. **BKS** -- Biharmonic Kernel Signature (Lipman et al. 2010)
8. **IBKS** -- Improved BKS (Zhang et al. 2024)
9. **SI-WKS** -- Scale-Invariant WKS (WKS times ``1/lambda_K``)
10. **HFE** -- Heat Flow Entropy, a per-vertex saliency map
11. **M-GP landmarks** -- Morphometric Gaussian Process active-learning
    landmarking (Fan et al., Med. Image Anal. 2021): the vertices of maximal
    GP posterior variance under a SIWKS x HFE kernel, whose concatenated
    SIWKS form a compact per-subject global descriptor
    (:func:`mgp_landmarks`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    GlobalDescriptor,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)

_ZERO_TOL = 1e-10
"""Eigenvalues <= this are treated as the (numerically) zero modes."""

_WARNED_AUTO_TIMES = False


def _warn_auto_times_once() -> None:
    """Warn (once per session) that auto HKS times are shape-specific."""
    global _WARNED_AUTO_TIMES
    if not _WARNED_AUTO_TIMES:
        logger.warning(
            "HKS/Bates: time scales auto-derived from THIS shape's spectrum. "
            "For cross-subject / cross-parcel comparison pass a shared grid, "
            "e.g. t_values=shared_hks_times([decomp_a, decomp_b, ...])."
        )
        _WARNED_AUTO_TIMES = True


# ======================================================================
# S1  ShapeDNA  (Reuter, Wolter & Peinecke, 2006)
# ======================================================================


def compute_shapedna(
    decomp: SpectralDecomposition,
    *,
    normalize: Literal["none", "area", "volume", "fiedler"] = "area",
    skip_zero: bool = True,
) -> GlobalDescriptor:
    """ShapeDNA -- the LBO eigenvalue fingerprint.

    The simplest spectral descriptor: the truncated sequence of
    eigenvalues, optionally normalised for cross-subject comparison.

    .. math::

        \\text{ShapeDNA} = (\\lambda_1, \\lambda_2, \\ldots, \\lambda_k)

    Parameters
    ----------
    decomp : SpectralDecomposition
        Precomputed eigenpairs.
    normalize : str
        ``"none"`` -- raw eigenvalues.
        ``"area"`` -- multiply by surface area (Reuter convention).
        ``"volume"`` -- multiply by volume^{2/3}.
        ``"fiedler"`` -- divide by lambda_1.
    skip_zero : bool
        Exclude lambda_0 ~ 0 (the constant mode).

    Returns
    -------
    ndarray, shape (d,)
        Eigenvalue vector.  d = k-1 if *skip_zero*, else d = k.

    References
    ----------
    Reuter M, Wolter FE, Peinecke N. Laplace-Beltrami spectra as
    "Shape-DNA" of surfaces and solids. *Computer-Aided Design*
    38(4):342-366, 2006.
    """
    evals = decomp.eigenvalues.copy()
    start = 1 if skip_zero else 0
    dna = evals[start:]

    if normalize == "area":
        if decomp.surface_area is None or decomp.surface_area <= 0:
            raise ValueError("Surface area required for area normalisation.")
        dna = dna * decomp.surface_area
    elif normalize == "volume":
        raise NotImplementedError("Volume normalisation requires volumetric eigendecomposition.")
    elif normalize == "fiedler":
        # The Fiedler value is the first *non-zero* eigenvalue lambda_1 -- not
        # dna[0], which is lambda_0 ~ 0 when ``skip_zero=False``.
        nonzero = evals[evals > _ZERO_TOL]
        if nonzero.size == 0:
            raise ValueError("Fiedler value is zero.")
        dna = dna / nonzero[0]
    elif normalize != "none":
        raise ValueError(f"Unknown normalisation: {normalize!r}")

    return dna


# ======================================================================
# S2  HKS -- Heat Kernel Signature  (Sun, Ovsjanikov & Guibas, 2009)
# ======================================================================


def _auto_hks_times(
    eigenvalues: np.ndarray,
    n_times: int = 100,
) -> np.ndarray:
    """Auto-compute log-spaced time values for HKS.

    Following Sun et al. 2009:
        t_min = 4*ln(10) / lambda_max
        t_max = 4*ln(10) / lambda_1
    """
    lam = eigenvalues[eigenvalues > 1e-10]
    if len(lam) < 2:
        return np.logspace(-2, 2, n_times)
    c = 4.0 * np.log(10.0)
    t_min = c / lam[-1]
    t_max = c / lam[0]
    # Clamp to reasonable range.
    t_min = max(t_min, 1e-6)
    t_max = min(t_max, 1e6)
    return np.logspace(np.log10(t_min), np.log10(t_max), n_times)


def shared_hks_times(
    decomps: list[SpectralDecomposition] | SpectralDecomposition,
    n_times: int = 100,
) -> np.ndarray:
    """A common log-spaced HKS time grid for a collection of shapes.

    Uses the Sun et al. (2009) bounds over the *pooled* spectra:
    ``t_min = 4 ln10 / max_s lambda_max(s)`` and ``t_max = 4 ln10 / min_s lambda_1(s)``,
    so every shape is evaluated at the same times and the HKS columns are
    comparable across subjects / parcels.

    Parameters
    ----------
    decomps : SpectralDecomposition or list of them
    n_times : int

    Returns
    -------
    ndarray, shape (n_times,)
    """
    if isinstance(decomps, SpectralDecomposition):
        decomps = [decomps]
    lam_max: list[float] = []
    lam_1: list[float] = []
    for d in decomps:
        nz = np.asarray(d.eigenvalues)[np.asarray(d.eigenvalues) > _ZERO_TOL]
        if nz.size:
            lam_max.append(float(nz.max()))
            lam_1.append(float(nz.min()))
    if not lam_max:
        return np.logspace(-2, 2, n_times)
    pooled = np.array([min(lam_1), max(lam_max)])
    return _auto_hks_times(np.concatenate([[0.0], pooled]), n_times)


def compute_hks(
    decomp: SpectralDecomposition,
    t_values: np.ndarray | None = None,
    *,
    n_times: int = 100,
    normalize: bool = False,
) -> DescriptorMatrix:
    """Heat Kernel Signature -- multi-scale per-vertex descriptor.

    The HKS measures how much heat remains at a point after
    diffusing for time *t*.  Small *t* captures local geometry
    (curvature); large *t* captures global shape.

    .. math::

        \\text{HKS}(x, t) = \\sum_{i=0}^{k-1}
            e^{-\\lambda_i t}\\, \\varphi_i^2(x)

    Parameters
    ----------
    decomp : SpectralDecomposition
    t_values : ndarray, shape (T,), optional
        Time scales.  ``None`` = auto log-spaced from *this shape's*
        eigenvalues (a warning is logged once: such grids differ between
        shapes -- use :func:`shared_hks_times` for group comparisons).
    n_times : int
        Number of auto time scales (ignored if *t_values* given).
    normalize : bool
        If True, normalise each column (time slice) to unit L2 norm.

    Returns
    -------
    ndarray, shape (N, T)
        HKS evaluated at each vertex and time.

    References
    ----------
    Sun J, Ovsjanikov M, Guibas L. A concise and provably
    informative multi-scale signature based on heat diffusion.
    *SGP 2009*.
    """
    evals = decomp.eigenvalues  # (k,)
    evecs = decomp.eigenvectors  # (N, k)

    if t_values is None:
        _warn_auto_times_once()
        t_values = _auto_hks_times(evals, n_times)
    t_values = np.asarray(t_values, dtype=np.float64)

    # Phi^2 : (N, k) -- squared eigenfunctions.
    phi_sq = evecs**2  # (N, k)

    # exp(-lambda*t) : (T, k)
    exp_lt = np.exp(-evals[None, :] * t_values[:, None])  # (T, k)

    # HKS = Phi^2 @ exp(-lambda*t)^T : (N, T)
    hks = phi_sq @ exp_lt.T  # (N, T)

    if normalize:
        norms = np.linalg.norm(hks, axis=0, keepdims=True)
        hks = hks / np.clip(norms, 1e-30, None)

    logger.debug(
        "HKS: N=%d, T=%d, t in [%.2e, %.2e]",
        hks.shape[0],
        hks.shape[1],
        t_values[0],
        t_values[-1],
    )
    return hks


# ======================================================================
# S3  SI-HKS -- Scale-Invariant HKS  (Bronstein & Kokkinos, 2010)
# ======================================================================


def compute_si_hks(
    decomp: SpectralDecomposition,
    *,
    n_times: int = 256,
    n_frequencies: int = 8,
) -> DescriptorMatrix:
    """Scale-Invariant HKS -- removes scale dependence from HKS.

    Under uniform scaling beta the HKS undergoes a log-time shift and
    amplitude change.  SI-HKS eliminates both via:

    1. Sample HKS at log-spaced times tau = log(t).
    2. Take derivative w.r.t. tau (removes amplitude).
    3. Apply DFT; keep modulus of first *n_frequencies*
       coefficients (removes shift).

    .. math::

        \\text{SI-HKS}(x) = \\left|
            \\mathcal{F}\\left\\{
                \\frac{\\partial}{\\partial \\tau}
                \\text{HKS}(x, e^\\tau)
            \\right\\}
        \\right|_{1:n}

    Parameters
    ----------
    decomp : SpectralDecomposition
    n_times : int
        Number of log-time samples (FFT input length).
        Power of 2 recommended.
    n_frequencies : int
        Number of Fourier modulus coefficients to keep.

    Returns
    -------
    ndarray, shape (N, n_frequencies)
        Scale-invariant spectral descriptor.

    References
    ----------
    Bronstein MM, Kokkinos I. Scale-invariant heat kernel signatures
    for non-rigid shape recognition. *CVPR 2010*.
    """
    evals = decomp.eigenvalues

    # Log-spaced time values.
    t_vals = _auto_hks_times(evals, n_times)

    # Compute HKS at all times.
    hks = compute_hks(decomp, t_values=t_vals, normalize=False)  # (N, T)

    # Take log to linearise the amplitude scaling.
    hks_log = np.log(np.clip(hks, 1e-30, None))  # (N, T)

    # Derivative w.r.t. log-time (finite differences).
    dhks = np.diff(hks_log, axis=1)  # (N, T-1)

    # DFT along the time axis per vertex.
    fft_coeffs = np.fft.rfft(dhks, axis=1)  # (N, T//2+1)

    # Modulus of first n_frequencies (skip DC).
    n_freq = min(n_frequencies, fft_coeffs.shape[1] - 1)
    si_hks = np.abs(fft_coeffs[:, 1 : 1 + n_freq])  # (N, n_freq)

    logger.debug(
        "SI-HKS: N=%d, %d frequencies from %d time samples",
        si_hks.shape[0],
        n_freq,
        n_times,
    )
    return si_hks


# ======================================================================
# S4  WKS -- Wave Kernel Signature  (Aubry, Schlickewei & Cremers, 2011)
# ======================================================================


def _auto_wks_params(
    eigenvalues: np.ndarray,
    n_energies: int,
) -> tuple[np.ndarray, float]:
    """Auto-compute energy levels and bandwidth for WKS."""
    lam = eigenvalues[eigenvalues > 1e-10]
    if len(lam) < 2:
        return np.linspace(-2, 2, n_energies), 1.0

    log_lam = np.log(lam)
    e_min = log_lam[0]
    e_max = log_lam[-1]

    # Bandwidth sigma -- Aubry's recommendation.
    sigma = 7.0 * (e_max - e_min) / n_energies

    # Shift e_min/e_max inward by 2sigma to avoid boundary effects.
    e_min_shifted = e_min + 2 * sigma
    e_max_shifted = e_max - 2 * sigma

    if e_min_shifted >= e_max_shifted:
        # Fallback: use full range.
        e_min_shifted = e_min
        e_max_shifted = e_max
        sigma = (e_max - e_min) / (2 * n_energies)

    energies = np.linspace(e_min_shifted, e_max_shifted, n_energies)
    return energies, float(sigma)


def compute_wks(
    decomp: SpectralDecomposition,
    e_values: np.ndarray | None = None,
    *,
    n_energies: int = 100,
    sigma: float | None = None,
    normalize: bool = True,
) -> DescriptorMatrix:
    """Wave Kernel Signature -- band-pass per-vertex descriptor.

    Derived from the Schrodinger equation.  Acts as a bank of
    band-pass filters in log-eigenvalue space, giving balanced
    weight to all spectral frequencies (unlike HKS which is
    low-pass).

    .. math::

        \\text{WKS}(x, e) = C_e \\sum_{i=1}^{k}
            \\varphi_i^2(x)\\,
            \\exp\\!\\left(
                -\\frac{(e - \\log\\lambda_i)^2}{2\\sigma^2}
            \\right)

    where :math:`C_e` normalises so the filter weights sum to 1.

    Parameters
    ----------
    decomp : SpectralDecomposition
    e_values : ndarray, shape (E,), optional
        Log-energy levels.  ``None`` = auto from eigenvalues.
    n_energies : int
        Number of auto energy levels.
    sigma : float, optional
        Gaussian bandwidth.  ``None`` = auto (Aubry convention).
    normalize : bool
        Normalise each energy slice to unit L2 norm.

    Returns
    -------
    ndarray, shape (N, E)
        WKS evaluated at each vertex and energy level.

    References
    ----------
    Aubry M, Schlickewei U, Cremers D. The wave kernel signature:
    a quantum mechanical approach to shape analysis. *ICCV 2011*.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors
    N, _k = evecs.shape

    # Skip the zero eigenvalue.
    nz = evals > 1e-10
    evals_nz = evals[nz]
    evecs_nz = evecs[:, nz]
    log_lam = np.log(evals_nz)  # (k',)

    if e_values is None or sigma is None:
        auto_e, auto_sigma = _auto_wks_params(evals, n_energies)
        if e_values is None:
            e_values = auto_e
        if sigma is None:
            sigma = auto_sigma

    e_values = np.asarray(e_values, dtype=np.float64)
    E = len(e_values)

    # Gaussian filter weights: (E, k')
    #   g[j, i] = exp(-(e_j - log lambda_i)^2 / (2sigma^2))
    diff = e_values[:, None] - log_lam[None, :]  # (E, k')
    gauss = np.exp(-(diff**2) / (2 * sigma**2))  # (E, k')

    # Normalisation C_e: sum of weights per energy level.
    C = gauss.sum(axis=1, keepdims=True)  # (E, 1)
    C = np.clip(C, 1e-30, None)
    gauss_norm = gauss / C  # (E, k')

    # WKS = Phi^2 @ gauss_norm^T : (N, E)
    phi_sq = evecs_nz**2  # (N, k')
    wks = phi_sq @ gauss_norm.T  # (N, E)

    if normalize:
        norms = np.linalg.norm(wks, axis=0, keepdims=True)
        wks = wks / np.clip(norms, 1e-30, None)

    logger.debug(
        "WKS: N=%d, E=%d, sigma=%.4f, e in [%.2f, %.2f]",
        N,
        E,
        sigma,
        e_values[0],
        e_values[-1],
    )
    return wks


# ======================================================================
# S5  GPS -- Global Point Signature  (Rustamov, 2007)
# ======================================================================


def compute_gps(
    decomp: SpectralDecomposition,
    *,
    skip_zero: bool = True,
) -> DescriptorMatrix:
    """Global Point Signature -- spectral embedding of the surface.

    Embeds each point into a high-dimensional space where Euclidean
    distance equals diffusion distance (at t -> inf).

    .. math::

        \\text{GPS}(x) = \\left(
            \\frac{\\varphi_1(x)}{\\sqrt{\\lambda_1}},\\;
            \\frac{\\varphi_2(x)}{\\sqrt{\\lambda_2}},\\;
            \\ldots,\\;
            \\frac{\\varphi_k(x)}{\\sqrt{\\lambda_k}}
        \\right)

    .. warning::
        GPS is **not** sign/ordering invariant.  Eigenvectors have
        arbitrary sign (phi and -phi are both valid), so direct comparison
        between subjects requires sign alignment.  For group-level
        analysis, prefer HKS or WKS which use phi^2 and are
        sign-invariant.

    Parameters
    ----------
    decomp : SpectralDecomposition
    skip_zero : bool
        Exclude the constant eigenfunction (lambda_0 ~ 0).  Any remaining null
        mode (lambda <= 1e-10, e.g. lambda_0 when ``skip_zero=False`` or extra
        components) yields an all-zero column.

    Returns
    -------
    ndarray, shape (N, d)
        Spectral embedding.  d = k-1 if *skip_zero*, else d = k.

    References
    ----------
    Rustamov RM. Laplace-Beltrami eigenfunctions for deformation
    invariant shape representation. *SGP 2007*.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    start = 1 if skip_zero else 0
    evals_sel = evals[start:]
    evecs_sel = evecs[:, start:]

    # Null modes (lambda ~ 0: the constant mode, or one per extra connected
    # component) have no finite GPS coordinate -- 1/sqrt lambda diverges.  Their
    # columns are set to 0 (shape is preserved) instead of being blown up
    # by 1/sqrt (clip(lambda)) ~ 1e5.
    null = evals_sel <= _ZERO_TOL
    if np.any(null):
        logger.warning(
            "GPS: %d null eigenmode(s) (lambda <= %.0e) have no finite GPS "
            "coordinate; their columns are set to 0.",
            int(null.sum()),
            _ZERO_TOL,
        )
    inv_sqrt_lam = np.zeros_like(evals_sel, dtype=np.float64)
    inv_sqrt_lam[~null] = 1.0 / np.sqrt(evals_sel[~null])

    gps = evecs_sel * inv_sqrt_lam[None, :]  # (N, d)

    logger.debug("GPS: N=%d, d=%d", gps.shape[0], gps.shape[1])
    return gps


# ======================================================================
# S6  Bates Symmetric Polynomial Signatures  (Bates et al., 2011)
# ======================================================================


def compute_bates_signatures(
    decomp: SpectralDecomposition,
    t_values: np.ndarray | None = None,
    *,
    n_times: int = 10,
    order: int = 2,
) -> DescriptorMatrix:
    """Symmetric polynomial signatures -- sign/ordering invariant.

    Construct per-mode heat weights w_j(x, t) = exp(-lambda_j*t)*phi_j(x)^2, then
    compute elementary symmetric polynomials e_p of the weights.  Because
    each w_j depends on phi_j only through phi_j^2, the signatures are invariant
    under eigenvector sign flips; being symmetric polynomials, they are
    invariant under permutations of (e.g. degenerate) eigenfunctions.

    .. math::

        e_1(x, t) &= \\sum_j w_j(x, t) \\quad \\text{(= HKS)}

        e_2(x, t) &= \\sum_{j < k} w_j(x, t)\\, w_k(x, t)

        e_p(x, t) &= \\sum_{j_1 < \\cdots < j_p}
                       \\prod_{m=1}^{p} w_{j_m}(x, t)

    For order=2 via Newton's identity:
    e_2 = (e_1^2 - Sigma w_j^2) / 2

    Parameters
    ----------
    decomp : SpectralDecomposition
    t_values : ndarray, optional
        Time scales.  ``None`` = auto.
    n_times : int
        Number of auto time scales.
    order : int
        Maximum order of symmetric polynomials (1, 2, or 3).
        Higher orders are more informative but O(k^order).

    Returns
    -------
    ndarray, shape (N, order x T)
        Concatenated symmetric polynomial signatures across
        orders and time scales.

    References
    ----------
    Bates J, Pafundi D, Kanel P, Liu X, Mio W. Spectral signatures
    of point clouds and applications to detection of Alzheimer's
    disease through neuroimaging. *IEEE ISBI 2011*.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors
    N, _k = evecs.shape

    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2 or 3, got {order}.")
    if t_values is None:
        _warn_auto_times_once()
        t_values = _auto_hks_times(evals, n_times)
    t_values = np.asarray(t_values, dtype=np.float64)
    T = len(t_values)

    results: list[np.ndarray] = []

    with progress_simple("Bates SP signatures", total=T) as tick:
        for _ti, t in enumerate(t_values):
            # Sign-invariant weights: w_j(x) = exp(-lambda_j*t) * phi_j(x)^2
            weights = np.exp(-evals * t)  # (k,)
            w = (evecs**2) * weights[None, :]  # (N, k)

            # e_1 = Sigma w_j (= HKS at time t)
            e1 = w.sum(axis=1)  # (N,)
            results.append(e1)

            if order >= 2:
                # e_2 = (e_1^2 - Sigma w_j^2) / 2  (Newton's identity)
                sum_sq = np.sum(w**2, axis=1)  # (N,)
                e2 = (e1**2 - sum_sq) / 2.0  # (N,)
                results.append(e2)

            if order >= 3:
                # e_3 = (e_1^3 - 3*e_1*Sigmaw^2 + 2*Sigmaw^3) / 6
                sum_cu = np.sum(w**3, axis=1)  # (N,)
                e3 = (e1**3 - 3 * e1 * sum_sq + 2 * sum_cu) / 6.0
                results.append(e3)

            tick(1)

    # Stack: (N, order x T) -- columns alternate [e1_t0, e2_t0, e1_t1, e2_t1, ...]
    sig = np.column_stack(results)  # (N, order*T)

    logger.debug(
        "Bates SP: N=%d, order=%d, T=%d -> dim=%d",
        N,
        order,
        T,
        sig.shape[1],
    )
    return sig


# ======================================================================
# S7  BKS -- Biharmonic Kernel Signature  (Lipman et al., 2010)
# ======================================================================


def compute_bks(
    decomp: SpectralDecomposition,
) -> ScalarMap:
    """Biharmonic Kernel Signature -- parameter-free per-vertex scalar.

    Uses the biharmonic operator (Delta^2) instead of the heat operator.
    Unlike HKS and WKS, BKS has **no tuneable parameter** -- it is
    fully determined by the eigenpairs.

    .. math::

        \\text{BKS}(x) = \\sum_{i=1}^{k}
            \\frac{\\varphi_i^2(x)}{\\lambda_i^2}

    The 1/lambda^2 weighting gives dominant weight to low-frequency
    modes (global shape).

    Parameters
    ----------
    decomp : SpectralDecomposition

    Returns
    -------
    ndarray, shape (N,)
        Per-vertex BKS scalar.

    References
    ----------
    Lipman Y, Rustamov RM, Funkhouser TA. Biharmonic distance.
    *ACM Transactions on Graphics* 29(3):27, 2010.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    # Skip lambda_0 ~ 0.
    nz = evals > 1e-10
    evals_nz = evals[nz]
    evecs_nz = evecs[:, nz]

    inv_lam_sq = 1.0 / (evals_nz**2)  # (k',)
    bks = np.sum(evecs_nz**2 * inv_lam_sq[None, :], axis=1)  # (N,)

    logger.debug("BKS: N=%d, k'=%d non-zero eigenvalues", bks.shape[0], nz.sum())
    return bks


# ======================================================================
# S8  IBKS -- Improved Biharmonic Kernel Signature  (Zhang et al., 2024)
# ======================================================================


def compute_ibks(
    decomp: SpectralDecomposition,
    *,
    gaussian_curvature: ScalarMap | None = None,
    alpha: float = 0.1,
    k_neighbours: int = 10,
) -> ScalarMap:
    """Improved BKS with curvature-aware neighbourhood aggregation.

    Augments BKS with Gaussian curvature information to improve
    stability at articulation points and high-curvature regions.

    IBKS(x) = BKS(x) + alpha * mean_{y in N(x)} |K(y)| * BKS(y)

    where K is Gaussian curvature and N(x) is the k-nearest
    neighbourhood.

    Parameters
    ----------
    decomp : SpectralDecomposition
    gaussian_curvature : ndarray, shape (N,), optional
        Pre-computed Gaussian curvature.  If ``None``, the curvature
        term is approximated from the eigenvectors (less accurate
        but avoids requiring a mesh).
    alpha : float
        Blending weight for the curvature term.
    k_neighbours : int
        Neighbourhood size for local aggregation.

    Returns
    -------
    ndarray, shape (N,)
        Per-vertex IBKS.

    References
    ----------
    Zhang Y et al. Improved biharmonic kernel signature for 3D
    non-rigid shape matching and retrieval. *The Visual Computer*
    40:969-980, 2024.
    """
    bks = compute_bks(decomp)
    N = decomp.n_vertices

    if gaussian_curvature is not None:
        K_abs = np.abs(gaussian_curvature)
    else:
        # Approximate curvature from spectral gap: vertices with high
        # eigenfunction variation tend to have higher curvature.
        evals = decomp.eigenvalues
        evecs = decomp.eigenvectors
        nz = evals > 1e-10
        # Weighted variance of eigenfunctions as curvature proxy.
        weights = evals[nz][:10] if nz.sum() >= 10 else evals[nz]
        K_abs = np.sqrt(np.sum((evecs[:, nz][:, : len(weights)] ** 2) * weights[None, :], axis=1))

    # Neighbourhood aggregation via kNN on eigenvector embedding.
    from spectralbrain.core.base import knn_search

    # Use the first few eigenvectors as embedding for neighbourhood.
    n_emb = min(10, decomp.n_eigenvalues)
    emb = decomp.eigenvectors[:, :n_emb]
    _, indices = knn_search(emb, k=k_neighbours)

    # Aggregate: mean of curvature-weighted BKS in neighbourhood.
    nbr_bks = bks[indices]  # (N, k)
    nbr_K = K_abs[indices]  # (N, k)
    curvature_term = np.mean(nbr_K * nbr_bks, axis=1)  # (N,)

    ibks = bks + alpha * curvature_term

    logger.debug("IBKS: N=%d, alpha=%.2f, k_nn=%d", N, alpha, k_neighbours)
    return ibks


# ======================================================================
# S9  CONVENIENCE: compute all descriptors at once
# ======================================================================


def compute_all_descriptors(
    decomp: SpectralDecomposition,
    *,
    hks_n_times: int = 100,
    wks_n_energies: int = 100,
    si_hks_n_freq: int = 8,
    bates_order: int = 2,
    bates_n_times: int = 10,
    gaussian_curvature: ScalarMap | None = None,
) -> dict[str, GlobalDescriptor | DescriptorMatrix | ScalarMap]:
    """Compute all 8 spectral descriptors from one decomposition.

    Efficient because the eigendecomposition (the expensive step)
    is shared.  Each descriptor adds only O(N*k*T) work.

    Parameters
    ----------
    decomp : SpectralDecomposition
    hks_n_times : int
    wks_n_energies : int
    si_hks_n_freq : int
    bates_order : int
    bates_n_times : int
    gaussian_curvature : ndarray, optional
        For IBKS.

    Returns
    -------
    dict of {str: ndarray}
        Keys: ``"shapedna"``, ``"hks"``, ``"si_hks"``, ``"wks"``,
        ``"gps"``, ``"bates_sp"``, ``"bks"``, ``"ibks"``.
    """
    logger.info(
        "Computing all descriptors for %d vertices, k=%d",
        decomp.n_vertices,
        decomp.n_eigenvalues,
    )

    results: dict[str, GlobalDescriptor | DescriptorMatrix | ScalarMap] = {}

    results["shapedna"] = compute_shapedna(decomp, normalize="area")
    results["hks"] = compute_hks(decomp, n_times=hks_n_times)
    results["si_hks"] = compute_si_hks(decomp, n_frequencies=si_hks_n_freq)
    results["wks"] = compute_wks(decomp, n_energies=wks_n_energies)
    results["gps"] = compute_gps(decomp)
    results["bates_sp"] = compute_bates_signatures(
        decomp,
        order=bates_order,
        n_times=bates_n_times,
    )
    results["bks"] = compute_bks(decomp)
    results["ibks"] = compute_ibks(
        decomp,
        gaussian_curvature=gaussian_curvature,
    )

    logger.info(
        "All descriptors computed: %s",
        {k: v.shape for k, v in results.items()},
    )
    return results


# ======================================================================
# M-GP LANDMARKING -- SIWKS + HEAT-FLOW ENTROPY (Fan et al. 2021)
# ======================================================================


# ----------------------------- result container ------------------------------
@dataclass
class MGPLandmarkResult:
    """Output of :func:`mgp_landmarks`.

    Attributes
    ----------
    landmarks : ndarray (L,)
        Vertex indices, in selection order.
    descriptor : ndarray (L * E,)
        Concatenated SIWKS of the landmarks -- the concise subject descriptor.
    uncertainty : ScalarMap (N,)
        Residual GP posterior variance after the ``L`` landmarks.
    selection_scores : ndarray (L,)
        Posterior variance at each landmark's selection (saturation curve).
    siwks : DescriptorMatrix (N, E)
        Per-vertex SIWKS feature matrix.
    hfe : ScalarMap (N,)
        Heat Flow Entropy per vertex.
    metadata : dict
    """

    landmarks: np.ndarray
    descriptor: np.ndarray
    uncertainty: ScalarMap
    selection_scores: np.ndarray
    siwks: DescriptorMatrix
    hfe: ScalarMap
    metadata: dict = field(default_factory=dict)

    @property
    def n_landmarks(self) -> int:
        return int(self.landmarks.shape[0])


# ----------------------------- spectral descriptors --------------------------
def compute_si_wks(
    decomp: SpectralDecomposition,
    e_values: np.ndarray | None = None,
    *,
    n_energies: int = 100,
    sigma: float | None = None,
) -> DescriptorMatrix:
    r"""Scale-Invariant Wave Kernel Signature (SIWKS).

    The WKS rescaled by the inverse of the largest used eigenvalue,

    .. math::

        \mathrm{SIWKS}(x, e) = \frac{1}{\lambda_K}\, \mathrm{WKS}(x, e),

    which makes the descriptor invariant to global scaling of the shape
    (under :math:`\beta M`, :math:`\lambda \to \beta^2 \lambda`). Complements
    :func:`~spectralbrain.spectral.lbo.descriptors.compute_si_hks` on the WKS side.

    Parameters
    ----------
    decomp : SpectralDecomposition
    e_values : ndarray (E,), optional
        Log-energy levels; ``None`` -> auto from eigenvalues.
    n_energies : int
        Number of auto energy levels.
    sigma : float, optional
        Gaussian bandwidth; ``None`` -> Aubry convention.

    Returns
    -------
    DescriptorMatrix, shape (N, E)
    """
    wks = compute_wks(decomp, e_values, n_energies=n_energies, sigma=sigma, normalize=False)
    lam = np.asarray(decomp.eigenvalues, dtype=float)
    lam_K = float(lam[lam > 1e-10][-1])  # largest non-trivial eigenvalue
    return np.asarray(wks, dtype=float) / lam_K


def _adjacency_from_decomp(
    decomp: SpectralDecomposition,
    faces: np.ndarray | None,
    n_vertices: int,
) -> sp.csr_matrix:
    """1-ring adjacency: from ``faces`` if given, else from the stiffness pattern."""
    if faces is not None:
        F = np.asarray(faces, dtype=int)
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        A = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n_vertices, n_vertices))
    else:
        if decomp.stiffness is None:
            raise ValueError(
                "HFE needs mesh connectivity: pass `faces`, or provide a "
                "SpectralDecomposition that carries its `stiffness` matrix."
            )
        A = sp.csr_matrix(decomp.stiffness)
        A = (A != 0).astype(float)
    A = A.tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def heat_flow_entropy(
    decomp: SpectralDecomposition,
    *,
    t: float | None = None,
    heat: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> ScalarMap:
    r"""Heat Flow Entropy (HFE) -- per-vertex structural saliency.

    Diffuses heat on the surface (HKS field at time ``t``) and measures the
    Shannon entropy of the heat-gradient magnitudes across the 1-ring,

    .. math::

        \mathrm{HFE}(v_i) = -\sum_{j \in N(i)} p_{ij} \log p_{ij},
        \quad p_{ij} = \frac{|h_j - h_i|}{\sum_{j} |h_j - h_i|} \ge 0.

    High HFE marks structurally disordered regions (e.g. pial/white-matter
    transitions), used as the GP kernel weight in :func:`mgp_landmarks`.

    Parameters
    ----------
    decomp : SpectralDecomposition
    t : float, optional
        Heat diffusion time; ``None`` -> spectral mid-time.
    heat : ndarray (N,), optional
        Precomputed heat field (overrides ``t``).
    faces : ndarray (m, 3), optional
        Faces for the neighbourhood graph; if omitted, the stiffness sparsity
        pattern is used.

    Returns
    -------
    ScalarMap, shape (N,)
    """
    lam = np.asarray(decomp.eigenvalues, dtype=float)
    phi = np.asarray(decomp.eigenvectors, dtype=float)
    n = phi.shape[0]

    if heat is None:
        pos = lam > 1e-10
        lp, pp = lam[pos], phi[:, pos]
        if t is None:
            t = float(np.exp(0.5 * (np.log(1.0 / lp[-1]) + np.log(1.0 / lp[0]))))
        h = (pp**2) @ np.exp(-lp * t)
    else:
        h = np.asarray(heat, dtype=float)

    A = _adjacency_from_decomp(decomp, faces, n)
    hfe = np.zeros(n)
    indptr, indices = A.indptr, A.indices
    for i in range(n):
        nbr = indices[indptr[i] : indptr[i + 1]]
        if nbr.size == 0:
            continue
        d = np.abs(h[nbr] - h[i])
        s = d.sum()
        if s <= 1e-12:
            continue
        p = d / s
        nz = p > 0
        hfe[i] = -np.sum(p[nz] * np.log(p[nz]))
    return hfe


# ------------------------ distance map + kernel operator ---------------------
def _siwks_distance_map(S: np.ndarray, knn: int) -> sp.csr_matrix:
    n = S.shape[0]
    k = min(knn + 1, n)
    tree = cKDTree(S)
    _, idx = tree.query(S, k=k)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    w = np.abs(S[rows] - S[cols]).sum(axis=1)
    m = rows != cols
    M = sp.csr_matrix((w[m], (rows[m], cols[m])), shape=(n, n))
    return M.maximum(M.T).tocsr()


def _row_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    M = M.tocsr().copy()
    counts = np.diff(M.indptr).astype(float)
    counts[counts == 0] = 1.0
    return (sp.diags(1.0 / counts) @ M).tocsr()


class _MGPKernel:
    """K = Mbar H Mbar^T as an operator (diagonal + columns on demand, never dense)."""

    def __init__(self, Mbar: sp.csr_matrix, hfe: np.ndarray):
        self.Mbar = Mbar
        self.hfe = np.asarray(hfe, float)

    def diag(self) -> np.ndarray:
        return np.asarray(self.Mbar.multiply(self.Mbar) @ self.hfe).ravel()

    def column(self, j: int) -> np.ndarray:
        row_j = self.Mbar.getrow(j).toarray().ravel()
        return np.asarray(self.Mbar @ (self.hfe * row_j)).ravel()


def _pivoted_cholesky(kernel: _MGPKernel, n_landmarks: int, jitter: float):
    N = kernel.Mbar.shape[0]
    L = int(min(n_landmarks, N))
    d = kernel.diag().astype(float)
    G = np.zeros((N, L), float)
    landmarks = np.empty(L, int)
    scores = np.empty(L, float)
    chosen = np.zeros(N, bool)
    for l in range(L):
        j = int(np.argmax(np.where(chosen, -np.inf, d)))
        piv = d[j]
        landmarks[l], scores[l], chosen[j] = j, piv, True
        if piv <= jitter:
            landmarks, scores, G = landmarks[: l + 1], scores[: l + 1], G[:, : l + 1]
            break
        col = kernel.column(j)
        if l > 0:
            col = col - G[:, :l] @ G[j, :l]
        g = col / np.sqrt(piv)
        G[:, l] = g
        d = np.maximum(d - g**2, 0.0)
    return landmarks, d, scores


# --------------------------------- public API --------------------------------
def mgp_landmarks(
    decomp: SpectralDecomposition,
    *,
    faces: np.ndarray | None = None,
    n_landmarks: int = 50,
    n_energies: int = 100,
    knn: int = 20,
    sigma: float | None = None,
    heat_t: float | None = None,
    heat: np.ndarray | None = None,
    jitter: float = 1e-10,
) -> MGPLandmarkResult:
    """Select landmarks by M-GP active learning from a spectral decomposition.

    Parameters
    ----------
    decomp : SpectralDecomposition
        The central SpectralBrain object (eigenpairs; ``stiffness`` enables the
        HFE neighbourhood when ``faces`` is not supplied).
    faces : ndarray (m, 3), optional
        Triangular faces for the HFE 1-ring; if omitted, taken from the
        stiffness sparsity pattern.
    n_landmarks : int
        Number of landmarks ``L`` to select.
    n_energies : int
        SIWKS energy scales (per-landmark descriptor length).
    knn : int
        Neighbours in the SIWKS distance-map graph.
    sigma : float, optional
        SIWKS bandwidth (None -> Aubry rule).
    heat_t : float, optional
        Heat-field diffusion time for HFE (None -> spectral mid-time).
    heat : ndarray (N,), optional
        Precomputed heat field (overrides ``heat_t``).
    jitter : float
        Pivot floor for the Cholesky stop.

    Returns
    -------
    MGPLandmarkResult
    """
    S = compute_si_wks(decomp, n_energies=n_energies, sigma=sigma)
    hfe = heat_flow_entropy(decomp, t=heat_t, heat=heat, faces=faces)

    Mbar = _row_normalize(_siwks_distance_map(S, knn))
    kernel = _MGPKernel(Mbar, hfe)
    landmarks, residual_var, scores = _pivoted_cholesky(kernel, n_landmarks, jitter)

    logger.info(
        "M-GP landmarking: selected %d landmarks from %d vertices (SIWKS E=%d, knn=%d)",
        landmarks.shape[0],
        S.shape[0],
        n_energies,
        knn,
    )

    return MGPLandmarkResult(
        landmarks=landmarks,
        descriptor=S[landmarks].ravel(),
        uncertainty=residual_var,
        selection_scores=scores,
        siwks=S,
        hfe=hfe,
        metadata=dict(
            method="mgp_landmarking",
            reference="Fan et al., Med. Image Anal. 72 (2021) 102123",
            n_landmarks=int(landmarks.shape[0]),
            n_energies=int(n_energies),
            knn=int(knn),
            structure=(decomp.metadata or {}).get("structure"),
        ),
    )


__all__: list[str] = [
    "MGPLandmarkResult",
    "compute_all_descriptors",
    "compute_bates_signatures",
    "compute_bks",
    "compute_gps",
    "compute_hks",
    "compute_ibks",
    "compute_shapedna",
    "compute_si_hks",
    "compute_si_wks",
    "compute_wks",
    "heat_flow_entropy",
    "mgp_landmarks",
    "shared_hks_times",
]
