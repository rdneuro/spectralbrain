"""Spectral shape descriptors derived from the LBO eigenpairs.

Every function in this module consumes a
:class:`~spectralbrain.core.base.SpectralDecomposition` and produces
a :pydata:`ScalarMap` ``(N,)``, :pydata:`DescriptorMatrix` ``(N, T)``,
or :pydata:`GlobalDescriptor` ``(d,)``.

Once the eigendecomposition is computed (the expensive step), all
descriptors are algebraically cheap — just array operations on
``eigenvalues`` and ``eigenvectors``.

Implemented descriptors
-----------------------
1. **ShapeDNA** — eigenvalue fingerprint (Reuter et al. 2006)
2. **HKS** — Heat Kernel Signature (Sun, Ovsjanikov & Guibas 2009)
3. **SI-HKS** — Scale-Invariant HKS (Bronstein & Kokkinos 2010)
4. **WKS** — Wave Kernel Signature (Aubry, Schlickewei & Cremers 2011)
5. **GPS** — Global Point Signature (Rustamov 2007)
6. **Bates SP** — Symmetric Polynomial Signatures (Bates et al. 2011)
7. **BKS** — Biharmonic Kernel Signature (Lipman et al. 2010)
8. **IBKS** — Improved BKS (Zhang et al. 2024)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    GlobalDescriptor,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  ShapeDNA  (Reuter, Wolter & Peinecke, 2006)
# ======================================================================


def compute_shapedna(
    decomp: SpectralDecomposition,
    *,
    normalize: Literal["none", "area", "volume", "fiedler"] = "area",
    skip_zero: bool = True,
) -> GlobalDescriptor:
    """ShapeDNA — the LBO eigenvalue fingerprint.

    The simplest spectral descriptor: the truncated sequence of
    eigenvalues, optionally normalised for cross-subject comparison.

    .. math::

        \\text{ShapeDNA} = (\\lambda_1, \\lambda_2, \\ldots, \\lambda_k)

    Parameters
    ----------
    decomp : SpectralDecomposition
        Precomputed eigenpairs.
    normalize : str
        ``"none"`` — raw eigenvalues.
        ``"area"`` — multiply by surface area (Reuter convention).
        ``"volume"`` — multiply by volume^{2/3}.
        ``"fiedler"`` — divide by λ₁.
    skip_zero : bool
        Exclude λ₀ ≈ 0 (the constant mode).

    Returns
    -------
    ndarray, shape (d,)
        Eigenvalue vector.  d = k−1 if *skip_zero*, else d = k.

    References
    ----------
    Reuter M, Wolter FE, Peinecke N. Laplace–Beltrami spectra as
    "Shape-DNA" of surfaces and solids. *Computer-Aided Design*
    38(4):342–366, 2006.
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
        if dna[0] <= 0:
            raise ValueError("Fiedler value is zero.")
        dna = dna / dna[0]
    elif normalize != "none":
        raise ValueError(f"Unknown normalisation: {normalize!r}")

    return dna


# ======================================================================
# §2  HKS — Heat Kernel Signature  (Sun, Ovsjanikov & Guibas, 2009)
# ======================================================================


def _auto_hks_times(
    eigenvalues: np.ndarray,
    n_times: int = 100,
) -> np.ndarray:
    """Auto-compute log-spaced time values for HKS.

    Following Sun et al. 2009:
        t_min = 4·ln(10) / λ_max
        t_max = 4·ln(10) / λ_1
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


def compute_hks(
    decomp: SpectralDecomposition,
    t_values: np.ndarray | None = None,
    *,
    n_times: int = 100,
    normalize: bool = False,
) -> DescriptorMatrix:
    """Heat Kernel Signature — multi-scale per-vertex descriptor.

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
        Time scales.  ``None`` = auto log-spaced from eigenvalues.
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
        t_values = _auto_hks_times(evals, n_times)
    t_values = np.asarray(t_values, dtype=np.float64)

    # Φ² : (N, k) — squared eigenfunctions.
    phi_sq = evecs**2  # (N, k)

    # exp(-λ·t) : (T, k)
    exp_lt = np.exp(-evals[None, :] * t_values[:, None])  # (T, k)

    # HKS = Φ² @ exp(-λ·t)ᵀ : (N, T)
    hks = phi_sq @ exp_lt.T  # (N, T)

    if normalize:
        norms = np.linalg.norm(hks, axis=0, keepdims=True)
        hks = hks / np.clip(norms, 1e-30, None)

    logger.debug(
        "HKS: N=%d, T=%d, t∈[%.2e, %.2e]",
        hks.shape[0],
        hks.shape[1],
        t_values[0],
        t_values[-1],
    )
    return hks


# ======================================================================
# §3  SI-HKS — Scale-Invariant HKS  (Bronstein & Kokkinos, 2010)
# ======================================================================


def compute_si_hks(
    decomp: SpectralDecomposition,
    *,
    n_times: int = 256,
    n_frequencies: int = 8,
) -> DescriptorMatrix:
    """Scale-Invariant HKS — removes scale dependence from HKS.

    Under uniform scaling β the HKS undergoes a log-time shift and
    amplitude change.  SI-HKS eliminates both via:

    1. Sample HKS at log-spaced times τ = log(t).
    2. Take derivative w.r.t. τ (removes amplitude).
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
# §4  WKS — Wave Kernel Signature  (Aubry, Schlickewei & Cremers, 2011)
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

    # Bandwidth σ — Aubry's recommendation.
    sigma = 7.0 * (e_max - e_min) / n_energies

    # Shift e_min/e_max inward by 2σ to avoid boundary effects.
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
    """Wave Kernel Signature — band-pass per-vertex descriptor.

    Derived from the Schrödinger equation.  Acts as a bank of
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
    #   g[j, i] = exp(-(e_j - log λ_i)² / (2σ²))
    diff = e_values[:, None] - log_lam[None, :]  # (E, k')
    gauss = np.exp(-(diff**2) / (2 * sigma**2))  # (E, k')

    # Normalisation C_e: sum of weights per energy level.
    C = gauss.sum(axis=1, keepdims=True)  # (E, 1)
    C = np.clip(C, 1e-30, None)
    gauss_norm = gauss / C  # (E, k')

    # WKS = Φ² @ gauss_normᵀ : (N, E)
    phi_sq = evecs_nz**2  # (N, k')
    wks = phi_sq @ gauss_norm.T  # (N, E)

    if normalize:
        norms = np.linalg.norm(wks, axis=0, keepdims=True)
        wks = wks / np.clip(norms, 1e-30, None)

    logger.debug(
        "WKS: N=%d, E=%d, σ=%.4f, e∈[%.2f, %.2f]",
        N,
        E,
        sigma,
        e_values[0],
        e_values[-1],
    )
    return wks


# ======================================================================
# §5  GPS — Global Point Signature  (Rustamov, 2007)
# ======================================================================


def compute_gps(
    decomp: SpectralDecomposition,
    *,
    skip_zero: bool = True,
) -> DescriptorMatrix:
    """Global Point Signature — spectral embedding of the surface.

    Embeds each point into a high-dimensional space where Euclidean
    distance equals diffusion distance (at t → ∞).

    .. math::

        \\text{GPS}(x) = \\left(
            \\frac{\\varphi_1(x)}{\\sqrt{\\lambda_1}},\\;
            \\frac{\\varphi_2(x)}{\\sqrt{\\lambda_2}},\\;
            \\ldots,\\;
            \\frac{\\varphi_k(x)}{\\sqrt{\\lambda_k}}
        \\right)

    .. warning::
        GPS is **not** sign/ordering invariant.  Eigenvectors have
        arbitrary sign (φ and −φ are both valid), so direct comparison
        between subjects requires sign alignment.  For group-level
        analysis, prefer HKS or WKS which use φ² and are
        sign-invariant.

    Parameters
    ----------
    decomp : SpectralDecomposition
    skip_zero : bool
        Exclude the constant eigenfunction (λ₀ ≈ 0).

    Returns
    -------
    ndarray, shape (N, d)
        Spectral embedding.  d = k−1 if *skip_zero*, else d = k.

    References
    ----------
    Rustamov RM. Laplace–Beltrami eigenfunctions for deformation
    invariant shape representation. *SGP 2007*.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    start = 1 if skip_zero else 0
    evals_sel = evals[start:]
    evecs_sel = evecs[:, start:]

    # Avoid division by zero for near-zero eigenvalues.
    inv_sqrt_lam = 1.0 / np.sqrt(np.clip(evals_sel, 1e-10, None))

    gps = evecs_sel * inv_sqrt_lam[None, :]  # (N, d)

    logger.debug("GPS: N=%d, d=%d", gps.shape[0], gps.shape[1])
    return gps


# ======================================================================
# §6  Bates Symmetric Polynomial Signatures  (Bates et al., 2011)
# ======================================================================


def compute_bates_signatures(
    decomp: SpectralDecomposition,
    t_values: np.ndarray | None = None,
    *,
    n_times: int = 10,
    order: int = 2,
) -> DescriptorMatrix:
    """Symmetric polynomial signatures — sign/ordering invariant.

    Construct weighted eigenfunctions w_j(x, t) = exp(-λ_j·t)·φ_j(x),
    then compute elementary symmetric polynomials e_p of the weights.
    These are provably invariant under sign flips and permutations of
    the eigenfunctions.

    .. math::

        e_1(x, t) &= \\sum_j w_j(x, t) \\quad \\text{(= HKS)}

        e_2(x, t) &= \\sum_{j < k} w_j(x, t)\\, w_k(x, t)

        e_p(x, t) &= \\sum_{j_1 < \\cdots < j_p}
                       \\prod_{m=1}^{p} w_{j_m}(x, t)

    For order=2 via Newton's identity:
    e_2 = (e_1² − Σ w_j²) / 2

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
    ndarray, shape (N, order × T)
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

    if t_values is None:
        t_values = _auto_hks_times(evals, n_times)
    t_values = np.asarray(t_values, dtype=np.float64)
    T = len(t_values)

    results: list[np.ndarray] = []

    with progress_simple("Bates SP signatures", total=T) as tick:
        for _ti, t in enumerate(t_values):
            # Weighted eigenfunctions: w_j(x) = exp(-λ_j·t) · φ_j(x)
            weights = np.exp(-evals * t)  # (k,)
            w = evecs * weights[None, :]  # (N, k)

            # e_1 = Σ w_j (equivalent to HKS diagonal)
            e1 = w.sum(axis=1)  # (N,)
            results.append(e1)

            if order >= 2:
                # e_2 = (e_1² − Σ w_j²) / 2  (Newton's identity)
                sum_sq = np.sum(w**2, axis=1)  # (N,)
                e2 = (e1**2 - sum_sq) / 2.0  # (N,)
                results.append(e2)

            if order >= 3:
                # e_3 = (e_1³ − 3·e_1·Σw² + 2·Σw³) / 6
                sum_cu = np.sum(w**3, axis=1)  # (N,)
                e3 = (e1**3 - 3 * e1 * sum_sq + 2 * sum_cu) / 6.0
                results.append(e3)

            tick(1)

    # Stack: (N, order × T) — columns alternate [e1_t0, e2_t0, e1_t1, e2_t1, ...]
    sig = np.column_stack(results)  # (N, order*T)

    logger.debug(
        "Bates SP: N=%d, order=%d, T=%d → dim=%d",
        N,
        order,
        T,
        sig.shape[1],
    )
    return sig


# ======================================================================
# §7  BKS — Biharmonic Kernel Signature  (Lipman et al., 2010)
# ======================================================================


def compute_bks(
    decomp: SpectralDecomposition,
) -> ScalarMap:
    """Biharmonic Kernel Signature — parameter-free per-vertex scalar.

    Uses the biharmonic operator (Δ²) instead of the heat operator.
    Unlike HKS and WKS, BKS has **no tuneable parameter** — it is
    fully determined by the eigenpairs.

    .. math::

        \\text{BKS}(x) = \\sum_{i=1}^{k}
            \\frac{\\varphi_i^2(x)}{\\lambda_i^2}

    The 1/λ² weighting gives dominant weight to low-frequency
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

    # Skip λ₀ ≈ 0.
    nz = evals > 1e-10
    evals_nz = evals[nz]
    evecs_nz = evecs[:, nz]

    inv_lam_sq = 1.0 / (evals_nz**2)  # (k',)
    bks = np.sum(evecs_nz**2 * inv_lam_sq[None, :], axis=1)  # (N,)

    logger.debug("BKS: N=%d, k'=%d non-zero eigenvalues", bks.shape[0], nz.sum())
    return bks


# ======================================================================
# §8  IBKS — Improved Biharmonic Kernel Signature  (Zhang et al., 2024)
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

    IBKS(x) = BKS(x) + α · mean_{y ∈ N(x)} |K(y)| · BKS(y)

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
    40:969–980, 2024.
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

    logger.debug("IBKS: N=%d, α=%.2f, k_nn=%d", N, alpha, k_neighbours)
    return ibks


# ======================================================================
# §9  CONVENIENCE: compute all descriptors at once
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
    is shared.  Each descriptor adds only O(N·k·T) work.

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
# §10  __all__
# ======================================================================

__all__: list[str] = [
    "compute_all_descriptors",
    "compute_bates_signatures",
    "compute_bks",
    "compute_gps",
    "compute_hks",
    "compute_ibks",
    "compute_shapedna",
    "compute_si_hks",
    "compute_wks",
]
