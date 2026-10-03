"""Normative modeling, harmonization, and non-inferiority testing.

Build age-/sex-stratified normative distributions of spectral
descriptors from healthy reference cohorts, score individual patients
against the normative, and formally test whether spectral descriptors
are non-inferior to conventional morphometrics.

Sections
--------
§1  ComBat / ComBat-GAM harmonization
§2  NormativeModel — build, evaluate, persist
§3  Centile curves — age-trajectory percentile charts
§4  Individual deviation scoring
§5  Non-inferiority & equivalence testing (TOST, AUC comparison)
§6  Method comparison — spectral vs volumetric discrimination
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import stats as sp_stats

from spectralbrain.runtime import (
    PathLike,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  COMBAT / COMBAT-GAM HARMONIZATION
# ======================================================================


@dataclass
class HarmonizationResult:
    """Result container for ComBat / ComBat-GAM harmonization.

    Attributes
    ----------
    data_harmonized : np.ndarray
        Harmonized data matrix, shape ``(n_samples, n_features)``.
    method : str
        Method used: ``"combat"`` or ``"combat_gam"``.
    sites : np.ndarray
        Original site labels.
    n_sites : int
        Number of unique sites.
    site_counts : dict
        Per-site sample counts.
    estimates : dict
        Estimated batch parameters (gamma, delta, etc.) for
        reproducibility and inspection.
    """

    data_harmonized: np.ndarray
    method: str
    sites: np.ndarray
    n_sites: int
    site_counts: dict[str, int]
    estimates: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a compact summary string."""
        return (
            f"HarmonizationResult(method='{self.method}', "
            f"n_sites={self.n_sites}, shape={self.data_harmonized.shape})"
        )


def harmonize_combat(
    data: np.ndarray,
    sites: np.ndarray,
    *,
    covariates: np.ndarray | None = None,
    covariate_names: list[str] | None = None,
    empirical_bayes: bool = True,
    parametric: bool = True,
    mean_only: bool = False,
    reference_site: str | None = None,
) -> HarmonizationResult:
    """Remove multi-site batch effects using ComBat (Johnson et al., 2007).

    ComBat uses an empirical Bayesian framework to estimate and remove
    additive and multiplicative batch (site) effects while preserving
    biological variability associated with covariates of interest.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix with samples as rows and features as columns.
    sites : np.ndarray, shape (n_samples,)
        Site/batch labels for each sample.
    covariates : np.ndarray, shape (n_samples, n_covariates), optional
        Biological covariates to preserve (e.g., age, sex, diagnosis).
    covariate_names : list of str, optional
        Names for each covariate column (for logging).
    empirical_bayes : bool
        If ``True`` (default), use empirical Bayes to shrink batch
        estimates toward the grand mean.
    parametric : bool
        If ``True`` (default), assume parametric priors (inverse-gamma /
        normal). If ``False``, use non-parametric EB.
    mean_only : bool
        If ``True``, adjust only the mean (no variance adjustment).
    reference_site : str, optional
        Harmonize all other sites to match this reference site.

    Notes
    -----
    ``result.estimates`` holds everything needed to harmonise *new* samples
    from the training sites with :func:`combat_apply`.

    Returns
    -------
    HarmonizationResult

    References
    ----------
    Johnson WE, Li C, Rabinovic A. Adjusting batch effects in
        microarray expression data using empirical Bayes methods.
        *Biostatistics* 8(1):118-127, 2007.
    Fortin J-P et al. Harmonization of multi-site diffusion tensor
        imaging data. *NeuroImage* 161:149-170, 2018.

    Examples
    --------
    >>> result = harmonize_combat(
    ...     descriptors, sites=site_labels,
    ...     covariates=np.column_stack([ages, sex]),
    ... )
    >>> harmonized = result.data_harmonized
    """
    data = np.asarray(data, dtype=np.float64)
    sites = np.asarray(sites)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_features = data.shape

    if len(sites) != n_samples:
        raise ValueError(f"sites length ({len(sites)}) != data rows ({n_samples}).")

    unique_sites = np.unique(sites)
    n_sites = len(unique_sites)

    if n_sites < 2:
        logger.warning("Only 1 site found -- returning data unchanged.")
        return HarmonizationResult(
            data_harmonized=data.copy(),
            method="combat",
            sites=sites,
            n_sites=1,
            site_counts={str(unique_sites[0]): n_samples},
        )

    site_counts = _site_counts(sites, unique_sites, "ComBat")

    logger.info(
        "ComBat harmonization: %d samples x %d features, %d sites.",
        n_samples,
        n_features,
        n_sites,
    )

    covariate_design = None
    if covariates is not None:
        covariate_design = np.asarray(covariates, dtype=np.float64)
        if covariate_design.ndim == 1:
            covariate_design = covariate_design.reshape(-1, 1)
        if covariate_design.shape[0] != n_samples:
            raise ValueError("covariates must have one row per sample.")

    harmonized, est = _combat_core(
        data,
        sites,
        unique_sites,
        site_counts,
        covariate_design,
        empirical_bayes=empirical_bayes,
        parametric=parametric,
        mean_only=mean_only,
    )

    est["reference_shift"] = np.zeros(n_features)
    if reference_site is not None:
        ref_idx = np.where(unique_sites == reference_site)[0]
        if len(ref_idx) == 0:
            raise ValueError(
                f"Reference site '{reference_site}' not found in: {list(unique_sites)}"
            )
        ref_mask = sites == reference_site
        ref_mean = data[ref_mask].mean(axis=0)
        harm_ref_mean = harmonized[ref_mask].mean(axis=0)
        shift = ref_mean - harm_ref_mean
        harmonized = harmonized + shift
        est["reference_shift"] = shift
    est["reference_site"] = reference_site
    est["covariate_names"] = covariate_names

    return HarmonizationResult(
        data_harmonized=harmonized,
        method="combat",
        sites=sites,
        n_sites=n_sites,
        site_counts=site_counts,
        estimates=est,
    )


def _site_counts(sites: np.ndarray, unique_sites: np.ndarray, name: str) -> dict[str, int]:
    """Per-site sample counts; every site needs >= 2 samples."""
    site_counts = {}
    for s in unique_sites:
        count = int((sites == s).sum())
        if count < 2:
            raise ValueError(f"Site '{s}' has {count} sample(s); {name} requires >= 2.")
        site_counts[str(s)] = count
    return site_counts


def _combat_core(
    data: np.ndarray,
    sites: np.ndarray,
    unique_sites: np.ndarray,
    site_counts: dict[str, int],
    covariate_design: np.ndarray | None,
    *,
    empirical_bayes: bool,
    parametric: bool,
    mean_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """ComBat location/scale model (Johnson et al. 2007) on a given design.

    Data are standardised feature-wise by the pooled residual SD before the
    per-site location (gamma) and scale (delta) effects are estimated, so the
    empirical-Bayes hyperpriors -- which are pooled *across features within
    each site*, as in the original method -- operate on comparable scales.
    """
    n_samples, n_features = data.shape
    n_sites = len(unique_sites)
    site_idx = np.searchsorted(unique_sites, sites)
    site_design = np.zeros((n_samples, n_sites), dtype=np.float64)
    site_design[np.arange(n_samples), site_idx] = 1.0

    design = (
        site_design
        if covariate_design is None
        else np.column_stack([site_design, covariate_design])
    )
    beta_hat = np.linalg.pinv(design.T @ design) @ (design.T @ data)
    counts = np.array([site_counts[str(s)] for s in unique_sites], dtype=np.float64)
    # Sample-size weighted grand mean (Johnson et al. 2007).
    grand_mean = (counts / n_samples) @ beta_hat[:n_sites]
    beta_cov = beta_hat[n_sites:]
    if covariate_design is not None:
        covar_effects = covariate_design @ beta_cov
    else:
        covar_effects = np.zeros((n_samples, n_features))

    residuals = data - design @ beta_hat
    pooled_var = (residuals**2).sum(axis=0) / max(n_samples - n_sites, 1)
    pooled_std = np.sqrt(np.clip(pooled_var, 1e-10, None))

    Z = (data - grand_mean - covar_effects) / pooled_std

    gamma_hat = np.zeros((n_sites, n_features))
    delta_hat = np.ones((n_sites, n_features))
    for i in range(n_sites):
        zs = Z[site_idx == i]
        gamma_hat[i] = zs.mean(axis=0)
        if not mean_only:
            delta_hat[i] = np.clip(zs.var(axis=0, ddof=1), 1e-10, None)

    if empirical_bayes and n_features < 2:
        warnings.warn(
            "ComBat empirical Bayes pools information across features; with a "
            "single feature it is undefined, so un-shrunk estimates are used.",
            RuntimeWarning,
            stacklevel=3,
        )
        empirical_bayes = False

    if empirical_bayes:
        gamma_star, delta_star = _combat_eb_estimates(
            gamma_hat,
            delta_hat,
            site_counts,
            unique_sites,
            parametric=parametric,
            Z=Z,
            site_idx=site_idx,
            mean_only=mean_only,
        )
    else:
        gamma_star, delta_star = gamma_hat.copy(), delta_hat.copy()

    harmonized = np.empty_like(data)
    for i in range(n_sites):
        m = site_idx == i
        adjusted = (Z[m] - gamma_star[i]) / np.sqrt(delta_star[i])
        harmonized[m] = adjusted * pooled_std + grand_mean + covar_effects[m]

    est = {
        "gamma_hat": gamma_hat,
        "delta_hat": delta_hat,
        "gamma_star": gamma_star,
        "delta_star": delta_star,
        "grand_mean": grand_mean,
        "pooled_std": pooled_std,
        "beta_covariates": beta_cov,
        "unique_sites": unique_sites,
        "empirical_bayes": empirical_bayes,
        "parametric": parametric,
        "mean_only": mean_only,
    }
    return harmonized, est


def combat_apply(
    data: np.ndarray,
    sites: np.ndarray,
    estimates: dict[str, Any],
    *,
    covariates: np.ndarray | None = None,
) -> np.ndarray:
    """Apply previously estimated ComBat parameters to new samples.

    Parameters
    ----------
    data : np.ndarray, shape (n, n_features) or (n_features,)
        New (unharmonised) samples.
    sites : array-like, shape (n,) or scalar
        Site label of each sample; must be one of the training sites.
    estimates : dict
        ``HarmonizationResult.estimates`` of a :func:`harmonize_combat` fit.
    covariates : np.ndarray, shape (n, n_covariates), optional
        Covariates in the same column layout used for fitting. Required when
        the fit used covariates.

    Returns
    -------
    np.ndarray
        Harmonised samples with the same shape as ``data``.
    """
    x = np.asarray(data, dtype=np.float64)
    single = x.ndim == 1
    X = x.reshape(1, -1) if single else x
    site_arr = np.atleast_1d(np.asarray(sites))
    if site_arr.size == 1 and X.shape[0] > 1:
        site_arr = np.repeat(site_arr, X.shape[0])
    unique_sites = np.asarray(estimates["unique_sites"])
    beta_cov = np.asarray(estimates["beta_covariates"])
    if beta_cov.shape[0] > 0:
        if covariates is None:
            raise ValueError("This ComBat fit used covariates; pass `covariates`.")
        C = np.asarray(covariates, dtype=np.float64).reshape(X.shape[0], -1)
        cov_eff = C @ beta_cov
    else:
        cov_eff = np.zeros_like(X)
    out = np.empty_like(X)
    gm = estimates["grand_mean"]
    ps = estimates["pooled_std"]
    shift = estimates.get("reference_shift", 0.0)
    for r in range(X.shape[0]):
        hit = np.where(unique_sites == site_arr[r])[0]
        if hit.size == 0:
            raise ValueError(
                f"Site {site_arr[r]!r} was not part of the harmonisation fit "
                f"({list(unique_sites)})."
            )
        i = int(hit[0])
        z = (X[r] - gm - cov_eff[r]) / ps
        adj = (z - estimates["gamma_star"][i]) / np.sqrt(estimates["delta_star"][i])
        out[r] = adj * ps + gm + cov_eff[r] + shift
    return out[0] if single else out


def _combat_eb_estimates(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    site_counts: dict[str, int],
    unique_sites: np.ndarray,
    *,
    parametric: bool = True,
    Z: np.ndarray | None = None,
    site_idx: np.ndarray | None = None,
    mean_only: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical Bayes shrinkage of ComBat site effects (Johnson et al. 2007).

    Hyperpriors are estimated per site across features. The parametric
    branch iterates the normal / inverse-gamma conjugate posterior
    (``it.sol`` in the reference implementation); the non-parametric branch
    integrates the likelihood over the empirical distribution of the other
    features' estimates (``int.eprior``).

    Parameters
    ----------
    gamma_hat, delta_hat : np.ndarray, shape (n_sites, n_features)
        Naive per-site location / scale estimates on standardised data.
    site_counts : dict
    unique_sites : np.ndarray
    parametric : bool
    Z : np.ndarray, shape (n_samples, n_features)
        Standardised data (required for the iterative / non-parametric EB).
    site_idx : np.ndarray, shape (n_samples,)
        Site index of each row of ``Z``.
    mean_only : bool
        Shrink only the location (scale fixed at 1).

    Returns
    -------
    gamma_star, delta_star : np.ndarray, shape (n_sites, n_features)
    """
    n_sites = gamma_hat.shape[0]
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.ones_like(delta_hat)
    if Z is None or site_idx is None:
        raise ValueError("_combat_eb_estimates needs the standardised data Z and site_idx.")

    for i in range(n_sites):
        zs = Z[site_idx == i]
        n_i = zs.shape[0]
        g_hat, d_hat = gamma_hat[i], delta_hat[i]
        if parametric:
            g_bar = g_hat.mean()
            t2 = max(g_hat.var(ddof=1), 1e-10)
            if mean_only:
                gamma_star[i] = (t2 * n_i * g_hat + d_hat * g_bar) / (t2 * n_i + d_hat)
                continue
            m = d_hat.mean()
            s2 = max(d_hat.var(ddof=1), 1e-10)
            a_prior = (2 * s2 + m**2) / s2
            b_prior = (m * s2 + m**3) / s2
            g_old, d_old = g_hat.copy(), d_hat.copy()
            for _ in range(max_iter):
                g_new = (t2 * n_i * g_hat + d_old * g_bar) / (t2 * n_i + d_old)
                sum2 = ((zs - g_new) ** 2).sum(axis=0)
                d_new = (0.5 * sum2 + b_prior) / (n_i / 2.0 + a_prior - 1.0)
                change = max(
                    np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-12)),
                    np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-12)),
                )
                g_old, d_old = g_new, d_new
                if change < tol:
                    break
            gamma_star[i], delta_star[i] = g_old, np.clip(d_old, 1e-10, None)
        else:
            # Non-parametric EB: weights from the likelihood of feature g's data
            # under every *other* feature's (gamma_hat, delta_hat).
            s1 = zs.sum(axis=0)  # (F,)
            s2 = (zs**2).sum(axis=0)  # (F,)
            d_use = np.ones_like(d_hat) if mean_only else d_hat
            # sum_j (z_jg - gamma_h)^2 for all (g, h)
            ss = s2[:, None] - 2.0 * s1[:, None] * g_hat[None, :] + n_i * g_hat[None, :] ** 2
            ll = -0.5 * n_i * np.log(2 * np.pi * d_use)[None, :] - ss / (2.0 * d_use[None, :])
            np.fill_diagonal(ll, -np.inf)
            ll -= ll.max(axis=1, keepdims=True)
            w = np.exp(ll)
            w /= w.sum(axis=1, keepdims=True)
            gamma_star[i] = w @ g_hat
            if not mean_only:
                delta_star[i] = np.clip(w @ d_hat, 1e-10, None)

    return gamma_star, delta_star


def harmonize_combat_gam(
    data: np.ndarray,
    sites: np.ndarray,
    *,
    continuous_covariates: np.ndarray | None = None,
    continuous_names: list[str] | None = None,
    categorical_covariates: np.ndarray | None = None,
    categorical_names: list[str] | None = None,
    smooth_terms: list[str] | None = None,
    n_splines: int = 10,
    empirical_bayes: bool = True,
    parametric: bool = True,
) -> HarmonizationResult:
    """Remove multi-site batch effects using ComBat-GAM (Pomponio et al., 2020).

    Extends ComBat by modeling nonlinear covariate effects using
    Generalized Additive Models (GAMs) with penalized B-splines.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix (samples x features).
    sites : np.ndarray, shape (n_samples,)
        Site/batch labels.
    continuous_covariates : np.ndarray, shape (n_samples, n_cont), optional
        Continuous covariates (e.g., age).
    continuous_names : list of str, optional
        Names for continuous covariates.
    categorical_covariates : np.ndarray, shape (n_samples, n_cat), optional
        Categorical covariates (e.g., sex, diagnosis).
    categorical_names : list of str, optional
        Names for categorical covariates.
    smooth_terms : list of str, optional
        Which continuous covariates to model with splines.
    n_splines : int
        Number of B-spline basis functions.
    empirical_bayes : bool
        Use empirical Bayes shrinkage.
    parametric : bool
        Parametric (default) or non-parametric empirical Bayes.

    Returns
    -------
    HarmonizationResult

    References
    ----------
    Pomponio R et al. Harmonization of large MRI datasets for the
        analysis of brain imaging patterns throughout the lifespan.
        *NeuroImage* 208:116450, 2020.
    """
    data = np.asarray(data, dtype=np.float64)
    sites = np.asarray(sites)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples, n_features = data.shape

    unique_sites = np.unique(sites)
    n_sites = len(unique_sites)
    site_counts = _site_counts(sites, unique_sites, "ComBat-GAM")

    logger.info(
        "ComBat-GAM: %d samples x %d features, %d sites, %d splines.",
        n_samples,
        n_features,
        n_sites,
        n_splines,
    )

    covariate_parts = []

    if continuous_covariates is not None:
        continuous_covariates = np.atleast_2d(np.asarray(continuous_covariates, dtype=np.float64))
        if continuous_covariates.shape[0] == 1 and n_samples > 1:
            continuous_covariates = continuous_covariates.T
        if continuous_names is None:
            continuous_names = [f"cont_{i}" for i in range(continuous_covariates.shape[1])]
        if smooth_terms is None:
            smooth_terms = continuous_names

        for j, name in enumerate(continuous_names):
            col = continuous_covariates[:, j]
            if name in smooth_terms:
                basis = _bspline_basis(col, n_splines)
                # Drop one column: a B-spline basis sums to one and would be
                # collinear with the site intercepts.
                covariate_parts.append(basis[:, 1:] if basis.shape[1] > 1 else basis * 0.0)
            else:
                covariate_parts.append(col.reshape(-1, 1))

    if categorical_covariates is not None:
        categorical_covariates = np.atleast_2d(np.asarray(categorical_covariates))
        if categorical_covariates.shape[0] == 1 and n_samples > 1:
            categorical_covariates = categorical_covariates.T
        for j in range(categorical_covariates.shape[1]):
            col = categorical_covariates[:, j]
            uniq = np.unique(col)
            for val in uniq[1:]:
                covariate_parts.append((col == val).astype(np.float64).reshape(-1, 1))

    covariate_design = np.column_stack(covariate_parts) if covariate_parts else None

    harmonized, est = _combat_core(
        data,
        sites,
        unique_sites,
        site_counts,
        covariate_design,
        empirical_bayes=empirical_bayes,
        parametric=parametric,
        mean_only=False,
    )
    est["n_splines"] = n_splines
    # The spline design is data-dependent (knots), so ``combat_apply`` cannot
    # rebuild it for new samples; flag the estimates accordingly.
    est["apply_supported"] = False

    return HarmonizationResult(
        data_harmonized=harmonized,
        method="combat_gam",
        sites=sites,
        n_sites=n_sites,
        site_counts=site_counts,
        estimates=est,
    )


def _bspline_basis(x: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """Construct a B-spline basis matrix.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Input variable values.
    n_basis : int
        Number of basis functions.
    degree : int
        Spline degree (default 3 = cubic).

    Returns
    -------
    np.ndarray, shape (n, n_actual_basis)
        B-spline design matrix.
    """
    from scipy.interpolate import BSpline

    n = len(x)
    x_min, x_max = float(x.min()), float(x.max())
    x_range = x_max - x_min
    if x_range < 1e-10:
        return np.ones((n, 1))

    n_internal = max(n_basis - degree - 1, 1)
    internal_knots = np.linspace(x_min, x_max, n_internal + 2)[1:-1]
    knots = np.concatenate(
        [
            np.repeat(x_min - x_range * 0.01, degree + 1),
            internal_knots,
            np.repeat(x_max + x_range * 0.01, degree + 1),
        ]
    )

    n_actual = len(knots) - degree - 1
    basis = np.zeros((n, n_actual))
    for i in range(n_actual):
        coeffs = np.zeros(n_actual)
        coeffs[i] = 1.0
        spl = BSpline(knots, coeffs, degree, extrapolate=False)
        vals = spl(x)
        vals[np.isnan(vals)] = 0.0
        basis[:, i] = vals

    return basis


def harmonize(
    data: np.ndarray,
    sites: np.ndarray,
    *,
    method: Literal["combat", "combat_gam"] = "combat",
    **kwargs: Any,
) -> HarmonizationResult:
    """Unified harmonization interface dispatching to ComBat or ComBat-GAM.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix.
    sites : np.ndarray, shape (n_samples,)
        Site labels.
    method : str
        ``"combat"`` or ``"combat_gam"``.
    **kwargs
        Forwarded to the selected harmonization function.

    Returns
    -------
    HarmonizationResult
    """
    if method == "combat":
        return harmonize_combat(data, sites, **kwargs)
    elif method == "combat_gam":
        return harmonize_combat_gam(data, sites, **kwargs)
    raise ValueError(f"Unknown harmonization method: {method!r}")


# ======================================================================
# §2  NORMATIVE MODEL
# ======================================================================


class NormativeModel:
    """Age- and sex-stratified normative distribution of descriptors.

    Fits a normative model on a healthy reference cohort and scores
    individuals against it.  Supports parametric (Gaussian) and
    non-parametric (percentile) scoring.

    Parameters
    ----------
    method : str
        ``"gaussian"`` | ``"centile"`` | ``"gp"``.
    centile_neighbours : int, optional
        For ``method="centile"`` with ages: number of reference subjects
        closest in age used as the age-matched reference (default
        ``max(20, S // 5)``, capped at ``S``).

    Notes
    -----
    If the model is fitted with ``harmonize_method="combat"``, the ComBat
    parameters are stored and :meth:`score` / :meth:`score_batch` require
    the ``site`` of the scored subject(s) so that new data are harmonised
    into the same space as the reference cohort.

    Examples
    --------
    >>> norm = NormativeModel(method="gaussian")
    >>> norm.fit(descriptors_controls, ages=ages, sex=sex)
    >>> z = norm.score(descriptor_patient, age=45, sex=1)
    """

    def __init__(
        self,
        method: Literal["gaussian", "centile", "gp"] = "gaussian",
        *,
        centile_neighbours: int | None = None,
    ) -> None:
        """Initialise a normative model with the given parameters."""
        self.method = method
        self.centile_neighbours = centile_neighbours
        self._is_fitted: bool = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._age_coef: np.ndarray | None = None
        self._sex_coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None
        self._residual_std: np.ndarray | None = None
        self._reference_data: np.ndarray | None = None
        self._reference_ages: np.ndarray | None = None
        self._reference_sex: np.ndarray | None = None
        self._gp_model: Any | None = None
        self._harmonization: dict[str, Any] | None = None

    def fit(
        self,
        descriptors: np.ndarray,
        *,
        ages: np.ndarray | None = None,
        sex: np.ndarray | None = None,
        sites: np.ndarray | None = None,
        harmonize_method: Literal["combat", "combat_gam"] | None = None,
        harmonize_kwargs: dict[str, Any] | None = None,
    ) -> NormativeModel:
        """Fit the normative model on a healthy reference cohort.

        Parameters
        ----------
        descriptors : ndarray, shape (S, N) or (S, d)
            Per-subject descriptor values.
        ages : ndarray, shape (S,), optional
            Ages in years (enables age conditioning).
        sex : ndarray, shape (S,), optional
            Biological sex (0/1).
        sites : ndarray, shape (S,), optional
            Site labels. Used with ``harmonize_method``.
        harmonize_method : str, optional
            ``"combat"`` or ``"combat_gam"`` -- harmonize before fitting.
            With ``"combat"`` the fitted parameters are stored and re-applied
            to scored subjects (pass ``site`` to :meth:`score`).
        harmonize_kwargs : dict, optional
            Extra kwargs forwarded to the harmonization function.

        Returns
        -------
        self
        """
        desc = np.asarray(descriptors, dtype=np.float64)
        S = desc.shape[0]
        ages_arr = None if ages is None else np.asarray(ages, dtype=np.float64).ravel()
        sex_arr = None if sex is None else np.asarray(sex, dtype=np.float64).ravel()
        self._harmonization = None

        if harmonize_method is not None and sites is None:
            raise ValueError("harmonize_method requires `sites`.")
        if harmonize_method is not None and sites is not None:
            hkw = dict(harmonize_kwargs or {})
            if harmonize_method == "combat":
                covs = None
                cov_layout: list[str] = []
                if ages_arr is not None or sex_arr is not None:
                    parts = []
                    if ages_arr is not None:
                        parts.append(ages_arr.reshape(-1, 1))
                        cov_layout.append("age")
                    if sex_arr is not None:
                        parts.append(sex_arr.reshape(-1, 1))
                        cov_layout.append("sex")
                    covs = np.column_stack(parts)
                result = harmonize_combat(desc, sites, covariates=covs, **hkw)
                if result.n_sites > 1:
                    self._harmonization = {
                        "method": "combat",
                        "estimates": result.estimates,
                        "covariate_layout": cov_layout,
                    }
            elif harmonize_method == "combat_gam":
                result = harmonize_combat_gam(desc, sites, **hkw)
                self._harmonization = {
                    "method": "combat_gam",
                    "estimates": result.estimates,
                    "covariate_layout": [],
                }
            else:
                raise ValueError(f"Unknown harmonize_method: {harmonize_method!r}")
            desc = result.data_harmonized
            logger.info("Applied %s before normative fitting.", harmonize_method)

        self._reference_data = desc
        self._reference_ages = ages_arr
        self._reference_sex = sex_arr

        if self.method == "gaussian":
            if ages_arr is not None:
                self._fit_gaussian_regression(desc, ages_arr, sex_arr)
            else:
                if sex_arr is not None:
                    warnings.warn(
                        "Gaussian normative without ages ignores `sex`; pass ages "
                        "to fit the age + sex regression.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._mean = desc.mean(axis=0)
                self._std = np.clip(desc.std(axis=0, ddof=1), 1e-10, None)

        elif self.method == "centile":
            pass

        elif self.method == "gp":
            if ages_arr is None:
                raise ValueError("GP normative requires ages.")
            from spectralbrain.statistics.bayesian import GaussianProcessNormative

            if desc.ndim > 1 and desc.shape[1] > 1:
                warnings.warn(
                    "GP normative models the mean descriptor across features; "
                    "the returned z-score is a single global deviation broadcast "
                    "to every feature.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._gp_model = GaussianProcessNormative(kernel="matern52")
            y_mean = desc.mean(axis=1) if desc.ndim > 1 else desc
            self._gp_model.fit(ages_arr.reshape(-1, 1), y_mean)

        self._is_fitted = True
        logger.info(
            "Normative fitted: method=%s, S=%d, features=%s", self.method, S, desc.shape[1:]
        )
        return self

    def _fit_gaussian_regression(
        self, desc: np.ndarray, ages: np.ndarray, sex: np.ndarray | None
    ) -> None:
        """Fit vertex-wise linear regression: desc ~ age + sex.

        Parameters
        ----------
        desc : np.ndarray, shape (S, D)
            Descriptor matrix.
        ages : np.ndarray, shape (S,)
            Ages.
        sex : np.ndarray or None
            Sex coding.
        """
        S = desc.shape[0]
        ages = np.asarray(ages, dtype=np.float64)
        X = np.column_stack([np.ones(S), ages])
        if sex is not None:
            X = np.column_stack([X, np.asarray(sex, dtype=np.float64)])

        beta = np.linalg.lstsq(X, desc, rcond=None)[0]
        predicted = X @ beta
        residuals = desc - predicted

        self._intercept = beta[0]
        self._age_coef = beta[1]
        self._sex_coef = beta[2] if sex is not None else None
        self._residual_std = np.clip(residuals.std(axis=0, ddof=X.shape[1]), 1e-10, None)
        self._reference_ages = ages

    def _harmonize_input(
        self,
        desc: np.ndarray,
        site: Any,
        age: Any,
        sex: Any,
    ) -> np.ndarray:
        """Apply the stored ComBat parameters to subject data (if any)."""
        h = self._harmonization
        if h is None:
            if site is not None:
                warnings.warn(
                    "`site` was given but the model was not fitted with harmonisation; "
                    "it is ignored.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return desc
        if h["method"] != "combat":
            raise NotImplementedError(
                "Scoring new subjects through a ComBat-GAM harmonised model is not "
                "supported (the spline design cannot be rebuilt); harmonise the new "
                "data jointly with the reference cohort instead."
            )
        if site is None:
            raise ValueError(
                "This normative model was fitted on ComBat-harmonised data; pass the "
                "subject's `site` so the same harmonisation is applied before scoring."
            )
        cov = None
        if h["covariate_layout"]:
            vals = []
            for name in h["covariate_layout"]:
                v = age if name == "age" else sex
                if v is None:
                    raise ValueError(
                        f"The harmonisation used `{name}` as a covariate; pass it to score()."
                    )
                vals.append(float(v))
            cov = np.asarray(vals, dtype=np.float64).reshape(1, -1)
        return combat_apply(desc, site, h["estimates"], covariates=cov)

    def score(
        self,
        descriptor: np.ndarray,
        *,
        age: float | None = None,
        sex: int | None = None,
        site: Any | None = None,
    ) -> np.ndarray:
        """Score an individual against the normative.

        Parameters
        ----------
        descriptor : ndarray, shape (N,) or (d,)
            Individual's descriptor values.
        age : float, optional
        sex : int, optional
        site : optional
            Site label of the subject. Required when the model was fitted
            with ComBat harmonisation.

        Returns
        -------
        ndarray
            Z-scores (Gaussian) or percentiles (centile).
        """
        self._check_fitted()
        desc = np.asarray(descriptor, dtype=np.float64)
        desc = self._harmonize_input(desc, site, age, sex)

        if self.method == "gaussian":
            if self._age_coef is not None:
                if age is None:
                    raise ValueError("This normative model is age-conditioned; pass `age`.")
                predicted = self._intercept + self._age_coef * age
                if self._sex_coef is not None:
                    if sex is None:
                        raise ValueError("This normative model was fitted with sex; pass `sex`.")
                    predicted = predicted + self._sex_coef * sex
                elif sex is not None:
                    warnings.warn(
                        "`sex` was given but the model was fitted without sex; ignored.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return (desc - predicted) / self._residual_std
            if age is not None:
                warnings.warn(
                    "`age` was given but the model was fitted without ages; ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return (desc - self._mean) / self._std

        elif self.method == "centile":
            ref = self._reference_data
            sel = np.ones(ref.shape[0], dtype=bool)
            if self._reference_sex is not None and sex is not None:
                same = self._reference_sex == float(sex)
                if same.sum() >= 5:
                    sel &= same
                else:
                    warnings.warn(
                        "Too few same-sex reference subjects; centiles are not sex-stratified.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if self._reference_ages is not None:
                if age is None:
                    warnings.warn(
                        "Reference ages are available but `age` was not given; "
                        "centiles are not age-matched.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    idx = np.where(sel)[0]
                    S = idx.size
                    k = self.centile_neighbours or max(20, S // 5)
                    k = int(min(max(k, 1), S))
                    order = np.argsort(np.abs(self._reference_ages[idx] - float(age)))
                    keep = np.zeros_like(sel)
                    keep[idx[order[:k]]] = True
                    sel = keep
            elif age is not None:
                warnings.warn(
                    "`age` was given but the model was fitted without ages; centiles "
                    "are not age-matched.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            ref = ref[sel]
            D = desc.shape[0]
            pctiles = np.array([sp_stats.percentileofscore(ref[:, v], desc[v]) for v in range(D)])
            return pctiles

        elif self.method == "gp":
            if age is None:
                raise ValueError("GP scoring requires age.")
            mean_desc = float(desc.mean())
            z = self._gp_model.deviation(age, mean_desc)
            return np.full(desc.shape, z)

        raise ValueError(f"Unknown method: {self.method!r}")

    def score_batch(
        self,
        descriptors: np.ndarray,
        *,
        ages: np.ndarray | None = None,
        sex: np.ndarray | None = None,
        sites: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score multiple individuals.

        Parameters
        ----------
        descriptors : ndarray, shape (S, N)
        ages : ndarray, shape (S,), optional
        sex : ndarray, shape (S,), optional
        sites : ndarray, shape (S,), optional
            Site labels (required for a ComBat-harmonised model).

        Returns
        -------
        ndarray, shape (S, N)
        """
        descriptors = np.asarray(descriptors, dtype=np.float64)
        S = descriptors.shape[0]
        results = []
        with progress_simple("Normative scoring", total=S) as tick:
            for i in range(S):
                a = ages[i] if ages is not None else None
                s = sex[i] if sex is not None else None
                st = sites[i] if sites is not None else None
                results.append(self.score(descriptors[i], age=a, sex=s, site=st))
                tick(1)
        return np.array(results)

    def extreme_count(self, z_scores: np.ndarray, threshold: float = 2.0) -> dict[str, Any]:
        """Count extreme deviations in a z-score map.

        Parameters
        ----------
        z_scores : ndarray, shape (N,)
        threshold : float

        Returns
        -------
        dict
        """
        z = np.asarray(z_scores)
        high = (z > threshold).sum()
        low = (z < -threshold).sum()
        return {
            "n_extreme": int(high + low),
            "pct_extreme": float(100 * (high + low) / len(z)),
            "n_high": int(high),
            "n_low": int(low),
            "max_z": float(z.max()),
            "min_z": float(z.min()),
            "mean_abs_z": float(np.abs(z).mean()),
        }

    def save(self, path: PathLike) -> Path:
        """Save normative model to HDF5.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Path
        """
        from spectralbrain.io.export import save_hdf5

        out = Path(path)
        arrays = {}
        for key, val in [
            ("mean", self._mean),
            ("std", self._std),
            ("age_coef", self._age_coef),
            ("sex_coef", self._sex_coef),
            ("intercept", self._intercept),
            ("residual_std", self._residual_std),
        ]:
            if val is not None:
                arrays[key] = val
        save_hdf5(out, descriptors=arrays, metadata={"method": self.method, "type": "normative"})
        logger.info("Normative model saved -> %s", out)
        return out

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError("NormativeModel not fitted. Call .fit() first.")


# ======================================================================
# §3  CENTILE CURVES
# ======================================================================


def centile_curves(
    descriptors: np.ndarray,
    ages: np.ndarray,
    *,
    percentiles: Sequence[float] = (2.5, 5, 25, 50, 75, 95, 97.5),
    n_age_bins: int = 20,
    smooth: bool = True,
    smooth_window: int = 3,
) -> dict[str, np.ndarray]:
    """Compute age-binned centile curves for a descriptor.

    Parameters
    ----------
    descriptors : ndarray, shape (S,) or (S, N)
    ages : ndarray, shape (S,)
    percentiles : sequence of float
    n_age_bins : int
    smooth : bool
    smooth_window : int

    Returns
    -------
    dict with ``"age_centers"`` and ``"centiles"``.
    """
    desc = np.asarray(descriptors, dtype=np.float64)
    if desc.ndim > 1:
        desc = desc.mean(axis=1)
    ages = np.asarray(ages, dtype=np.float64)

    bin_edges = np.linspace(ages.min(), ages.max(), n_age_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    centile_dict: dict[float, np.ndarray] = {}
    for pct in percentiles:
        curve = np.zeros(n_age_bins)
        for b in range(n_age_bins):
            upper = ages <= bin_edges[b + 1] if b == n_age_bins - 1 else ages < bin_edges[b + 1]
            mask = (ages >= bin_edges[b]) & upper
            curve[b] = np.percentile(desc[mask], pct) if mask.sum() > 0 else np.nan
        if smooth:
            curve = _moving_average(curve, smooth_window)
        centile_dict[pct] = curve

    return {"age_centers": bin_centers, "centiles": centile_dict}


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """NaN-aware moving average.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    w : int

    Returns
    -------
    np.ndarray, shape (n,)
    """
    out = np.copy(x)
    for i in range(len(x)):
        lo, hi = max(0, i - w // 2), min(len(x), i + w // 2 + 1)
        window = x[lo:hi]
        valid = window[~np.isnan(window)]
        out[i] = valid.mean() if len(valid) > 0 else np.nan
    return out


# ======================================================================
# §4  INDIVIDUAL DEVIATION SCORING
# ======================================================================


def z_score_map(
    subject: np.ndarray, normative_mean: np.ndarray, normative_std: np.ndarray
) -> ScalarMap:
    """Simple z-score map (no covariates).

    Parameters
    ----------
    subject : ndarray, shape (N,)
    normative_mean, normative_std : ndarray, shape (N,)

    Returns
    -------
    ndarray, shape (N,)
    """
    return (subject - normative_mean) / (normative_std + 1e-30)


def extreme_value_map(z_scores: np.ndarray, *, threshold: float = 2.0) -> np.ndarray:
    """Binary map of extreme deviations: +1 above, -1 below, 0 within.

    Parameters
    ----------
    z_scores : ndarray, shape (N,)
    threshold : float

    Returns
    -------
    ndarray, shape (N,), int32
    """
    z = np.asarray(z_scores)
    result = np.zeros_like(z, dtype=np.int32)
    result[z > threshold] = 1
    result[z < -threshold] = -1
    return result


# ======================================================================
# §5  NON-INFERIORITY & EQUIVALENCE TESTING
# ======================================================================


@dataclass
class NonInferiorityResult:
    """Result of a non-inferiority or equivalence test.

    Attributes
    ----------
    test_type, metric_new, metric_reference, margin, difference,
    ci_lower, ci_upper, p_value, is_non_inferior, is_equivalent.
    """

    test_type: str
    metric_new: float
    metric_reference: float
    margin: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_non_inferior: bool
    is_equivalent: bool = False

    def __repr__(self) -> str:
        """Return a compact summary string."""
        status = (
            "EQUIVALENT"
            if self.is_equivalent
            else ("NON-INFERIOR" if self.is_non_inferior else "INCONCLUSIVE")
        )
        return (
            f"NonInferiority({status}: new={self.metric_new:.4f}, "
            f"ref={self.metric_reference:.4f}, d={self.margin:.4f}, "
            f"diff={self.difference:.4f}, CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4f})"
        )


def _paired_se(
    new: np.ndarray, ref: np.ndarray, cv_test_train_ratio: float | None
) -> tuple[float, float, int]:
    """Mean paired difference, its SE and df (Nadeau-Bengio corrected if asked)."""
    if new.shape != ref.shape:
        raise ValueError(f"Paired test needs equal-length inputs; got {new.shape} and {ref.shape}.")
    n = len(new)
    diff = new - ref
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if cv_test_train_ratio is None:
        se = sd / np.sqrt(n)
    else:
        # Nadeau & Bengio (2003) corrected resampled t: folds share training
        # data, so Var(mean) = (1/k + n_test/n_train) * s^2.
        se = sd * np.sqrt(1.0 / n + float(cv_test_train_ratio))
    return mean_diff, se, n - 1


def non_inferiority_test(
    metric_new: np.ndarray,
    metric_reference: np.ndarray,
    *,
    margin: float = 0.05,
    alpha: float = 0.025,
    paired: bool = True,
    cv_test_train_ratio: float | None = None,
) -> NonInferiorityResult:
    """Non-inferiority test for method comparison.

    Parameters
    ----------
    metric_new, metric_reference : ndarray
    margin : float
    alpha : float
    paired : bool
    cv_test_train_ratio : float, optional
        When the paired values are per-fold cross-validation scores, pass
        ``n_test / n_train`` to apply the Nadeau-Bengio variance correction
        (folds are not independent). ``None`` = ordinary paired t.

    Returns
    -------
    NonInferiorityResult
    """
    new = np.asarray(metric_new, dtype=np.float64)
    ref = np.asarray(metric_reference, dtype=np.float64)

    if paired:
        mean_diff, se, df = _paired_se(new, ref, cv_test_train_ratio)
    else:
        if cv_test_train_ratio is not None:
            raise ValueError("cv_test_train_ratio only applies to paired=True.")
        # Two independent samples (Welch): SE and Satterthwaite df.
        na, nb = len(new), len(ref)
        mean_diff = float(new.mean() - ref.mean())
        va, vb = new.var(ddof=1), ref.var(ddof=1)
        se = float(np.sqrt(va / na + vb / nb))
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) + 1e-30
        )

    t_stat = (mean_diff + margin) / (se + 1e-30)
    p_value = 1 - sp_stats.t.cdf(t_stat, df)
    t_crit = sp_stats.t.ppf(1 - alpha, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    return NonInferiorityResult(
        test_type="non_inferiority",
        metric_new=float(new.mean()),
        metric_reference=float(ref.mean()),
        margin=margin,
        difference=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_value),
        is_non_inferior=ci_lower > -margin,
    )


def equivalence_test_tost(
    metric_new: np.ndarray,
    metric_reference: np.ndarray,
    *,
    margin: float = 0.05,
    alpha: float = 0.05,
    cv_test_train_ratio: float | None = None,
) -> NonInferiorityResult:
    """Two One-Sided Tests (TOST) for equivalence.

    Parameters
    ----------
    metric_new, metric_reference : ndarray
        Paired values (same length).
    margin : float
    alpha : float
    cv_test_train_ratio : float, optional
        Nadeau-Bengio correction for per-fold CV scores (see
        :func:`non_inferiority_test`).

    Returns
    -------
    NonInferiorityResult
        The reported CI is the ``1 - 2*alpha`` interval that corresponds to
        the TOST decision.
    """
    new = np.asarray(metric_new, dtype=np.float64)
    ref = np.asarray(metric_reference, dtype=np.float64)
    mean_diff, se, df = _paired_se(new, ref, cv_test_train_ratio)
    t1 = (mean_diff + margin) / (se + 1e-30)
    p1 = 1 - sp_stats.t.cdf(t1, df)
    t2 = (mean_diff - margin) / (se + 1e-30)
    p2 = sp_stats.t.cdf(t2, df)
    p_tost = max(p1, p2)
    t_crit = sp_stats.t.ppf(1 - alpha, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    return NonInferiorityResult(
        test_type="equivalence_tost",
        metric_new=float(new.mean()),
        metric_reference=float(ref.mean()),
        margin=margin,
        difference=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_tost),
        is_non_inferior=ci_lower > -margin,
        is_equivalent=p_tost < alpha,
    )


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (ties get the average rank), 1-based."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong AUC + covariance (Sun & Xu 2014).

    Parameters
    ----------
    preds : ndarray, shape (k, n_pos + n_neg)
        Scores for each of ``k`` classifiers, positive cases first.
    n_pos : int
        Number of positive cases.

    Returns
    -------
    aucs : ndarray, shape (k,)
    cov : ndarray, shape (k, k)
        Covariance matrix of the AUC estimates.
    """
    m = n_pos
    n = preds.shape[1] - m
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def auc_comparison_delong(
    y_true: np.ndarray,
    scores_new: np.ndarray,
    scores_reference: np.ndarray,
) -> tuple[float, float, float]:
    """DeLong test for two correlated (paired) ROC AUCs.

    Implements the analytic DeLong test using the fast midrank algorithm
    of Sun & Xu (2014). It is deterministic (no resampling) and is the
    standard method for comparing two AUCs computed on the *same* samples.

    Parameters
    ----------
    y_true : ndarray, shape (n,)
        Binary labels (the positive class is the larger value, typically 1).
    scores_new, scores_reference : ndarray, shape (n,)
        Predicted scores from the two models on the same samples.

    Returns
    -------
    auc_new, auc_ref, p_value : float
        The two AUCs and the two-sided p-value for ``auc_new == auc_ref``.

    References
    ----------
    DeLong ER, DeLong DM, Clarke-Pearson DL. *Biometrics* 44(3):837–845,
    1988. Sun X, Xu W. *IEEE Signal Process Lett* 21(11):1389–1393, 2014.
    """
    y = np.asarray(y_true)
    s_new = np.asarray(scores_new, dtype=np.float64)
    s_ref = np.asarray(scores_reference, dtype=np.float64)

    pos_label = y.max()
    is_pos = y == pos_label
    if is_pos.all() or (~is_pos).all():
        raise ValueError("y_true must contain both classes for AUC comparison.")

    # Order positives first.
    order = np.argsort(~is_pos, kind="stable")
    n_pos = int(is_pos.sum())
    preds = np.vstack([s_new[order], s_ref[order]])

    aucs, cov = _fast_delong(preds, n_pos)
    lvec = np.array([[1.0, -1.0]])
    var = float((lvec @ cov @ lvec.T).item())
    if var <= 1e-30:
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2.0 * sp_stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


# ======================================================================
# §6  METHOD COMPARISON
# ======================================================================


@dataclass
class MethodComparisonResult:
    """Comprehensive comparison between two methods."""

    method_new: str
    method_reference: str
    auc_new: float
    auc_reference: float
    auc_p_value: float
    non_inferiority: NonInferiorityResult
    equivalence: NonInferiorityResult
    effect_size_new: float
    effect_size_reference: float

    def __repr__(self) -> str:
        """Return a compact summary string."""
        ni = "NI" if self.non_inferiority.is_non_inferior else "?"
        eq = "EQ" if self.equivalence.is_equivalent else "?"
        return (
            f"MethodComparison({self.method_new} vs {self.method_reference}: "
            f"AUC {self.auc_new:.3f} vs {self.auc_reference:.3f} "
            f"(p={self.auc_p_value:.4f}), [{ni}] [{eq}])"
        )


def compare_methods(
    y_true: np.ndarray,
    features_new: np.ndarray,
    features_reference: np.ndarray,
    *,
    method_new_name: str = "spectral",
    method_ref_name: str = "volumetric",
    n_folds: int = 10,
    margin: float = 0.05,
    seed: int | None = 42,
) -> MethodComparisonResult:
    """Full head-to-head comparison between two methods.

    Parameters
    ----------
    y_true : ndarray, shape (n,)
    features_new : ndarray, shape (n, d_new)
    features_reference : ndarray, shape (n, d_ref)
    method_new_name, method_ref_name : str
    n_folds : int
    margin : float
    seed : int

    Returns
    -------
    MethodComparisonResult
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y, X_new, X_ref = np.asarray(y_true), np.asarray(features_new), np.asarray(features_reference)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs_new, aucs_ref = [], []
    scores_new_all, scores_ref_all = np.zeros(len(y)), np.zeros(len(y))

    for train_idx, test_idx in cv.split(X_new, y):
        for X, scores_all, aucs_list in [
            (X_new, scores_new_all, aucs_new),
            (X_ref, scores_ref_all, aucs_ref),
        ]:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            )
            pipe.fit(X[train_idx], y[train_idx])
            prob = pipe.predict_proba(X[test_idx])[:, 1]
            scores_all[test_idx] = prob
            aucs_list.append(roc_auc_score(y[test_idx], prob))

    aucs_new, aucs_ref = np.array(aucs_new), np.array(aucs_ref)
    auc_new_full, auc_ref_full, p_delong = auc_comparison_delong(y, scores_new_all, scores_ref_all)
    # Per-fold AUCs share training data: Nadeau-Bengio corrected variance.
    ratio = 1.0 / max(n_folds - 1, 1)
    ni = non_inferiority_test(aucs_new, aucs_ref, margin=margin, cv_test_train_ratio=ratio)
    eq = equivalence_test_tost(aucs_new, aucs_ref, margin=margin, cv_test_train_ratio=ratio)
    d_new = _cohens_d(scores_new_all[y == 0], scores_new_all[y == 1])
    d_ref = _cohens_d(scores_ref_all[y == 0], scores_ref_all[y == 1])

    result = MethodComparisonResult(
        method_new=method_new_name,
        method_reference=method_ref_name,
        auc_new=auc_new_full,
        auc_reference=auc_ref_full,
        auc_p_value=p_delong,
        non_inferiority=ni,
        equivalence=eq,
        effect_size_new=d_new,
        effect_size_reference=d_ref,
    )
    logger.info("%s", result)
    return result


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d between two groups.

    Parameters
    ----------
    a, b : np.ndarray
        Sample values.

    Returns
    -------
    float
    """
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float(abs(a.mean() - b.mean()) / (pooled + 1e-30))


__all__: list[str] = [
    "HarmonizationResult",
    "MethodComparisonResult",
    "NonInferiorityResult",
    "NormativeModel",
    "auc_comparison_delong",
    "centile_curves",
    "combat_apply",
    "compare_methods",
    "equivalence_test_tost",
    "extreme_value_map",
    "harmonize",
    "harmonize_combat",
    "harmonize_combat_gam",
    "non_inferiority_test",
    "z_score_map",
]
