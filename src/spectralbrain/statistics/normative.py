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

    site_counts = {}
    for s in unique_sites:
        count = int((sites == s).sum())
        if count < 2:
            raise ValueError(f"Site '{s}' has {count} sample(s); ComBat requires >= 2.")
        site_counts[str(s)] = count

    logger.info(
        "ComBat harmonization: %d samples x %d features, %d sites.",
        n_samples,
        n_features,
        n_sites,
    )

    # --- Step 1: Design matrix ---
    site_idx = np.searchsorted(unique_sites, sites)
    site_design = np.zeros((n_samples, n_sites), dtype=np.float64)
    site_design[np.arange(n_samples), site_idx] = 1.0

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        design = np.column_stack([site_design, covariates])
    else:
        design = site_design

    # --- Step 2: Standardize data ---
    beta_hat = np.linalg.pinv(design.T @ design) @ (design.T @ data)
    grand_mean = beta_hat[:n_sites].mean(axis=0)

    if covariates is not None:
        covar_effects = covariates @ beta_hat[n_sites:]
    else:
        covar_effects = np.zeros((n_samples, n_features))

    stand_data = data - grand_mean - covar_effects

    gamma_hat = np.zeros((n_sites, n_features))
    delta_hat = np.zeros((n_sites, n_features))

    for i, s in enumerate(unique_sites):
        mask = sites == s
        site_data = stand_data[mask]
        gamma_hat[i] = site_data.mean(axis=0)
        delta_hat[i] = site_data.var(axis=0, ddof=1)

    pooled_var = np.zeros(n_features)
    for i, s in enumerate(unique_sites):
        mask = sites == s
        ni = mask.sum()
        pooled_var += (ni - 1) * delta_hat[i]
    pooled_var /= n_samples - n_sites
    pooled_std = np.sqrt(np.clip(pooled_var, 1e-10, None))

    # --- Step 3: Empirical Bayes estimation ---
    if empirical_bayes:
        gamma_star, delta_star = _combat_eb_estimates(
            gamma_hat,
            delta_hat,
            site_counts,
            unique_sites,
            parametric=parametric,
        )
    else:
        gamma_star = gamma_hat
        delta_star = delta_hat

    # --- Step 4: Adjust data ---
    # Johnson et al. 2007: Y*_ij = pooled_std · (stand_data_ij - γ*_i) / √δ²*_i + grand_mean + X·β
    harmonized = np.zeros_like(data)
    for i, s in enumerate(unique_sites):
        mask = sites == s
        adjusted = (stand_data[mask] - gamma_star[i]) / np.sqrt(delta_star[i] + 1e-30)
        harmonized[mask] = adjusted * pooled_std + grand_mean + covar_effects[mask]

    if reference_site is not None:
        ref_idx = np.where(unique_sites == reference_site)[0]
        if len(ref_idx) == 0:
            raise ValueError(
                f"Reference site '{reference_site}' not found in: {list(unique_sites)}"
            )
        ref_mask = sites == reference_site
        ref_mean = data[ref_mask].mean(axis=0)
        harm_ref_mean = harmonized[ref_mask].mean(axis=0)
        harmonized += ref_mean - harm_ref_mean

    return HarmonizationResult(
        data_harmonized=harmonized,
        method="combat",
        sites=sites,
        n_sites=n_sites,
        site_counts=site_counts,
        estimates={
            "gamma_hat": gamma_hat,
            "delta_hat": delta_hat,
            "gamma_star": gamma_star if empirical_bayes else gamma_hat,
            "delta_star": delta_star if empirical_bayes else delta_hat,
            "grand_mean": grand_mean,
            "pooled_std": pooled_std,
        },
    )


def _combat_eb_estimates(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    site_counts: dict[str, int],
    unique_sites: np.ndarray,
    *,
    parametric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical Bayes shrunken estimates for ComBat.

    Parameters
    ----------
    gamma_hat : np.ndarray, shape (n_sites, n_features)
        Naive per-site location (mean shift) estimates.
    delta_hat : np.ndarray, shape (n_sites, n_features)
        Naive per-site scale (variance) estimates.
    site_counts : dict
        Sample counts per site.
    unique_sites : np.ndarray
        Ordered array of unique site labels.
    parametric : bool
        If ``True``, use inverse-gamma/normal conjugate priors.

    Returns
    -------
    gamma_star : np.ndarray, shape (n_sites, n_features)
        Shrunken location estimates.
    delta_star : np.ndarray, shape (n_sites, n_features)
        Shrunken scale estimates.
    """
    n_sites, _n_features = gamma_hat.shape
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    gamma_bar = gamma_hat.mean(axis=0)
    tau2 = np.clip(gamma_hat.var(axis=0, ddof=1), 1e-10, None)

    delta_mean = delta_hat.mean(axis=0)
    delta_var = np.clip(delta_hat.var(axis=0, ddof=1), 1e-10, None)

    alpha_bar = (delta_mean**2) / delta_var + 2
    beta_bar = delta_mean * (alpha_bar - 1)

    ordered_counts = [site_counts[str(s)] for s in unique_sites]

    for i in range(n_sites):
        ni = ordered_counts[i]
        if parametric:
            precision_prior = 1.0 / tau2
            precision_data = ni / (delta_hat[i] + 1e-10)
            gamma_star[i] = (precision_prior * gamma_bar + precision_data * gamma_hat[i]) / (
                precision_prior + precision_data
            )
            alpha_post = alpha_bar + ni / 2
            beta_post = beta_bar + 0.5 * ni * delta_hat[i]
            delta_star[i] = beta_post / (alpha_post + 1)
        else:
            gamma_star[i] = gamma_hat[i]
            delta_star[i] = delta_hat[i]

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
    n_samples, n_features = data.shape

    unique_sites = np.unique(sites)
    n_sites = len(unique_sites)

    site_counts = {}
    for s in unique_sites:
        count = int((sites == s).sum())
        if count < 2:
            raise ValueError(f"Site '{s}' has {count} sample(s); ComBat-GAM requires >= 2.")
        site_counts[str(s)] = count

    logger.info(
        "ComBat-GAM: %d samples x %d features, %d sites, %d splines.",
        n_samples,
        n_features,
        n_sites,
        n_splines,
    )

    site_idx = np.searchsorted(unique_sites, sites)
    site_design = np.zeros((n_samples, n_sites), dtype=np.float64)
    site_design[np.arange(n_samples), site_idx] = 1.0

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
                covariate_parts.append(_bspline_basis(col, n_splines))
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

    if covariate_parts:
        covariate_design = np.column_stack(covariate_parts)
        design = np.column_stack([site_design, covariate_design])
    else:
        covariate_design = None
        design = site_design

    beta_hat = np.linalg.pinv(design.T @ design) @ (design.T @ data)
    grand_mean = beta_hat[:n_sites].mean(axis=0)

    if covariate_design is not None:
        covar_effects = covariate_design @ beta_hat[n_sites:]
    else:
        covar_effects = np.zeros((n_samples, n_features))

    stand_data = data - grand_mean - covar_effects

    gamma_hat = np.zeros((n_sites, n_features))
    delta_hat = np.zeros((n_sites, n_features))
    for i, s in enumerate(unique_sites):
        mask = sites == s
        site_data = stand_data[mask]
        gamma_hat[i] = site_data.mean(axis=0)
        delta_hat[i] = site_data.var(axis=0, ddof=1)

    pooled_var = np.zeros(n_features)
    for i, s in enumerate(unique_sites):
        ni = (sites == s).sum()
        pooled_var += (ni - 1) * delta_hat[i]
    pooled_var /= n_samples - n_sites
    pooled_std = np.sqrt(np.clip(pooled_var, 1e-10, None))

    if empirical_bayes:
        gamma_star, delta_star = _combat_eb_estimates(
            gamma_hat,
            delta_hat,
            site_counts,
            unique_sites,
        )
    else:
        gamma_star, delta_star = gamma_hat, delta_hat

    harmonized = np.zeros_like(data)
    for i, s in enumerate(unique_sites):
        mask = sites == s
        adjusted = (stand_data[mask] - gamma_star[i]) / np.sqrt(delta_star[i] + 1e-30)
        harmonized[mask] = adjusted * pooled_std + grand_mean + covar_effects[mask]

    return HarmonizationResult(
        data_harmonized=harmonized,
        method="combat_gam",
        sites=sites,
        n_sites=n_sites,
        site_counts=site_counts,
        estimates={
            "gamma_hat": gamma_hat,
            "delta_hat": delta_hat,
            "gamma_star": gamma_star if empirical_bayes else gamma_hat,
            "delta_star": delta_star if empirical_bayes else delta_hat,
            "grand_mean": grand_mean,
            "pooled_std": pooled_std,
            "n_splines": n_splines,
        },
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

    Examples
    --------
    >>> norm = NormativeModel(method="gaussian")
    >>> norm.fit(descriptors_controls, ages=ages, sex=sex)
    >>> z = norm.score(descriptor_patient, age=45, sex=1)
    """

    def __init__(
        self,
        method: Literal["gaussian", "centile", "gp"] = "gaussian",
    ) -> None:
        """Initialise a normative model with the given parameters."""
        self.method = method
        self._is_fitted: bool = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._age_coef: np.ndarray | None = None
        self._sex_coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None
        self._residual_std: np.ndarray | None = None
        self._reference_data: np.ndarray | None = None
        self._reference_ages: np.ndarray | None = None
        self._gp_model: Any | None = None

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
        harmonize_kwargs : dict, optional
            Extra kwargs forwarded to the harmonization function.

        Returns
        -------
        self
        """
        desc = np.asarray(descriptors, dtype=np.float64)
        S = desc.shape[0]

        if harmonize_method is not None and sites is not None:
            hkw = harmonize_kwargs or {}
            if harmonize_method == "combat":
                covs = None
                if ages is not None or sex is not None:
                    parts = []
                    if ages is not None:
                        parts.append(np.asarray(ages, dtype=np.float64).reshape(-1, 1))
                    if sex is not None:
                        parts.append(np.asarray(sex, dtype=np.float64).reshape(-1, 1))
                    covs = np.column_stack(parts)
                result = harmonize_combat(desc, sites, covariates=covs, **hkw)
            elif harmonize_method == "combat_gam":
                result = harmonize_combat_gam(desc, sites, **hkw)
            else:
                raise ValueError(f"Unknown harmonize_method: {harmonize_method!r}")
            desc = result.data_harmonized
            logger.info("Applied %s before normative fitting.", harmonize_method)

        self._reference_data = desc

        if self.method == "gaussian":
            if ages is not None:
                self._fit_gaussian_regression(desc, ages, sex)
            else:
                self._mean = desc.mean(axis=0)
                self._std = np.clip(desc.std(axis=0, ddof=1), 1e-10, None)

        elif self.method == "centile":
            self._reference_data = desc
            self._reference_ages = ages

        elif self.method == "gp":
            if ages is None:
                raise ValueError("GP normative requires ages.")
            from spectralbrain.statistics.bayesian import GaussianProcessNormative

            self._gp_model = GaussianProcessNormative(kernel="matern52")
            y_mean = desc.mean(axis=1)
            self._gp_model.fit(ages.reshape(-1, 1), y_mean)

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
        S, _D = desc.shape
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

    def score(
        self, descriptor: np.ndarray, *, age: float | None = None, sex: int | None = None
    ) -> np.ndarray:
        """Score an individual against the normative.

        Parameters
        ----------
        descriptor : ndarray, shape (N,) or (d,)
            Individual's descriptor values.
        age : float, optional
        sex : int, optional

        Returns
        -------
        ndarray
            Z-scores (Gaussian) or percentiles (centile).
        """
        self._check_fitted()
        desc = np.asarray(descriptor, dtype=np.float64)

        if self.method == "gaussian":
            if self._age_coef is not None and age is not None:
                predicted = self._intercept + self._age_coef * age
                if self._sex_coef is not None and sex is not None:
                    predicted += self._sex_coef * sex
                return (desc - predicted) / self._residual_std
            else:
                return (desc - self._mean) / self._std

        elif self.method == "centile":
            ref = self._reference_data
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
    ) -> np.ndarray:
        """Score multiple individuals.

        Parameters
        ----------
        descriptors : ndarray, shape (S, N)
        ages : ndarray, shape (S,), optional
        sex : ndarray, shape (S,), optional

        Returns
        -------
        ndarray, shape (S, N)
        """
        S = descriptors.shape[0]
        results = []
        with progress_simple("Normative scoring", total=S) as tick:
            for i in range(S):
                a = ages[i] if ages is not None else None
                s = sex[i] if sex is not None else None
                results.append(self.score(descriptors[i], age=a, sex=s))
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
            mask = (ages >= bin_edges[b]) & (ages < bin_edges[b + 1])
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


def non_inferiority_test(
    metric_new: np.ndarray,
    metric_reference: np.ndarray,
    *,
    margin: float = 0.05,
    alpha: float = 0.025,
    paired: bool = True,
) -> NonInferiorityResult:
    """Non-inferiority test for method comparison.

    Parameters
    ----------
    metric_new, metric_reference : ndarray
    margin : float
    alpha : float
    paired : bool

    Returns
    -------
    NonInferiorityResult
    """
    new = np.asarray(metric_new, dtype=np.float64)
    ref = np.asarray(metric_reference, dtype=np.float64)

    if paired:
        n = min(len(new), len(ref))
        new, ref = new[:n], ref[:n]
        diff = new - ref
        mean_diff = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(n))
        df = n - 1
    else:
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
) -> NonInferiorityResult:
    """Two One-Sided Tests (TOST) for equivalence.

    Parameters
    ----------
    metric_new, metric_reference : ndarray
    margin : float
    alpha : float

    Returns
    -------
    NonInferiorityResult
    """
    new = np.asarray(metric_new, dtype=np.float64)
    ref = np.asarray(metric_reference, dtype=np.float64)
    n = min(len(new), len(ref))
    new, ref = new[:n], ref[:n]
    diff = new - ref
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    df = n - 1
    t1 = (mean_diff + margin) / (se + 1e-30)
    p1 = 1 - sp_stats.t.cdf(t1, df)
    t2 = (mean_diff - margin) / (se + 1e-30)
    p2 = sp_stats.t.cdf(t2, df)
    p_tost = max(p1, p2)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
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
            try:
                aucs_list.append(roc_auc_score(y[test_idx], prob))
            except ValueError:
                pass

    aucs_new, aucs_ref = np.array(aucs_new), np.array(aucs_ref)
    auc_new_full, auc_ref_full, p_delong = auc_comparison_delong(y, scores_new_all, scores_ref_all)
    ni = non_inferiority_test(aucs_new, aucs_ref, margin=margin)
    eq = equivalence_test_tost(aucs_new, aucs_ref, margin=margin)
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
    "compare_methods",
    "equivalence_test_tost",
    "extreme_value_map",
    "harmonize",
    "harmonize_combat",
    "harmonize_combat_gam",
    "non_inferiority_test",
    "z_score_map",
]
