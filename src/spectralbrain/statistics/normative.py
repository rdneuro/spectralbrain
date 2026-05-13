"""Normative modeling and non-inferiority testing for spectral morphometry.

Build age-/sex-stratified normative distributions of spectral
descriptors from healthy reference cohorts, score individual patients
against the normative, and formally test whether spectral descriptors
are non-inferior to conventional morphometrics.

Sections
--------
§1  NormativeModel — build, evaluate, persist
§2  Centile curves — age-trajectory percentile charts
§3  Individual deviation scoring
§4  Non-inferiority & equivalence testing (TOST, AUC comparison)
§5  Method comparison — spectral vs volumetric discrimination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

from spectralbrain.runtime import (
    DescriptorMatrix,
    GlobalDescriptor,
    PathLike,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  NORMATIVE MODEL
# ======================================================================

class NormativeModel:
    """Age- and sex-stratified normative distribution of descriptors.

    Fits a normative model on a healthy reference cohort and scores
    individuals against it.  Supports parametric (Gaussian) and
    non-parametric (percentile) scoring.

    Parameters
    ----------
    method : str
        ``"gaussian"`` — per-vertex Gaussian (mean, std) optionally
        conditioned on age via linear regression.
        ``"centile"`` — non-parametric centile estimation.
        ``"gp"`` — Gaussian Process (delegates to
        :class:`~spectralbrain.statistics.bayesian.GaussianProcessNormative`).

    Examples
    --------
    >>> norm = NormativeModel(method="gaussian")
    >>> norm.fit(descriptors_controls, ages=ages, sex=sex)
    >>> z = norm.score(descriptor_patient, age=45, sex=1)
    >>> z.shape  # (N,) z-scores per vertex
    """

    def __init__(
        self,
        method: Literal["gaussian", "centile", "gp"] = "gaussian",
    ) -> None:
        self.method = method
        self._is_fitted: bool = False
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._age_coef: Optional[np.ndarray] = None
        self._sex_coef: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None
        self._residual_std: Optional[np.ndarray] = None
        self._reference_data: Optional[np.ndarray] = None
        self._reference_ages: Optional[np.ndarray] = None
        self._gp_model: Optional[Any] = None

    def fit(
        self,
        descriptors: np.ndarray,
        *,
        ages: Optional[np.ndarray] = None,
        sex: Optional[np.ndarray] = None,
        site: Optional[np.ndarray] = None,
    ) -> "NormativeModel":
        """Fit the normative model on a healthy reference cohort.

        Parameters
        ----------
        descriptors : ndarray, shape (S, N) or (S, d)
            Per-subject descriptor values.  S subjects, N vertices
            (or d global features).
        ages : ndarray, shape (S,), optional
            Age in years.  If provided, normative is age-conditioned.
        sex : ndarray, shape (S,), optional
            Biological sex (0/1).  If provided, included as covariate.
        site : ndarray, shape (S,), optional
            Site labels (for ComBat-style residualisation — future).

        Returns
        -------
        self
        """
        desc = np.asarray(descriptors, dtype=np.float64)
        S = desc.shape[0]
        self._reference_data = desc

        if self.method == "gaussian":
            if ages is not None:
                self._fit_gaussian_regression(desc, ages, sex)
            else:
                self._mean = desc.mean(axis=0)
                self._std = desc.std(axis=0, ddof=1)
                self._std = np.clip(self._std, 1e-10, None)

        elif self.method == "centile":
            self._reference_data = desc
            self._reference_ages = ages

        elif self.method == "gp":
            if ages is None:
                raise ValueError("GP normative requires ages.")
            from spectralbrain.statistics.bayesian import GaussianProcessNormative
            # Fit GP on mean descriptor.
            self._gp_model = GaussianProcessNormative(kernel="matern52")
            y_mean = desc.mean(axis=1)  # per-subject global mean
            self._gp_model.fit(ages.reshape(-1, 1), y_mean)

        self._is_fitted = True
        logger.info(
            "Normative model fitted: method=%s, S=%d, features=%s",
            self.method, S, desc.shape[1:],
        )
        return self

    def _fit_gaussian_regression(
        self,
        desc: np.ndarray,
        ages: np.ndarray,
        sex: Optional[np.ndarray],
    ) -> None:
        """Fit vertex-wise linear regression: desc ~ age + sex."""
        S, D = desc.shape
        ages = np.asarray(ages, dtype=np.float64)

        # Design matrix.
        X = np.column_stack([np.ones(S), ages])
        if sex is not None:
            X = np.column_stack([X, np.asarray(sex, dtype=np.float64)])

        # OLS per feature.
        beta = np.linalg.lstsq(X, desc, rcond=None)[0]     # (p, D)
        predicted = X @ beta                                 # (S, D)
        residuals = desc - predicted                         # (S, D)

        self._intercept = beta[0]                            # (D,)
        self._age_coef = beta[1]                             # (D,)
        self._sex_coef = beta[2] if sex is not None else None
        self._residual_std = residuals.std(axis=0, ddof=X.shape[1])
        self._residual_std = np.clip(self._residual_std, 1e-10, None)
        self._reference_ages = ages

    def score(
        self,
        descriptor: np.ndarray,
        *,
        age: Optional[float] = None,
        sex: Optional[int] = None,
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
            Z-scores (Gaussian method) or percentiles (centile method).
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
            pctiles = np.zeros(D)
            for v in range(D):
                pctiles[v] = sp_stats.percentileofscore(ref[:, v], desc[v])
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
        ages: Optional[np.ndarray] = None,
        sex: Optional[np.ndarray] = None,
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

    def extreme_count(
        self,
        z_scores: np.ndarray,
        threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """Count extreme deviations in a z-score map.

        Parameters
        ----------
        z_scores : ndarray, shape (N,)
        threshold : float

        Returns
        -------
        dict
            ``n_extreme``, ``pct_extreme``, ``n_high``, ``n_low``,
            ``max_z``, ``min_z``.
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
        """Save normative model to HDF5."""
        from spectralbrain.io.export import save_hdf5
        out = Path(path)
        arrays = {}
        if self._mean is not None:
            arrays["mean"] = self._mean
        if self._std is not None:
            arrays["std"] = self._std
        if self._age_coef is not None:
            arrays["age_coef"] = self._age_coef
        if self._sex_coef is not None:
            arrays["sex_coef"] = self._sex_coef
        if self._intercept is not None:
            arrays["intercept"] = self._intercept
        if self._residual_std is not None:
            arrays["residual_std"] = self._residual_std

        save_hdf5(
            out,
            descriptors=arrays,
            metadata={"method": self.method, "type": "normative"},
        )
        logger.info("Normative model saved → %s", out)
        return out

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("NormativeModel not fitted. Call .fit() first.")


# ======================================================================
# §2  CENTILE CURVES
# ======================================================================

def centile_curves(
    descriptors: np.ndarray,
    ages: np.ndarray,
    *,
    percentiles: Sequence[float] = (2.5, 5, 25, 50, 75, 95, 97.5),
    n_age_bins: int = 20,
    smooth: bool = True,
    smooth_window: int = 3,
) -> Dict[str, np.ndarray]:
    """Compute age-binned centile curves for a descriptor.

    Parameters
    ----------
    descriptors : ndarray, shape (S,) or (S, N)
        Per-subject descriptor (1D global or multi-vertex).
        If multi-vertex, uses the mean per subject.
    ages : ndarray, shape (S,)
        Ages in years.
    percentiles : sequence of float
        Centile levels to compute.
    n_age_bins : int
        Number of age bins.
    smooth : bool
        Apply moving-average smoothing to centile curves.
    smooth_window : int
        Smoothing window size.

    Returns
    -------
    dict
        ``"age_centers"`` : ndarray, shape (n_bins,)
        ``"centiles"`` : dict of {percentile: ndarray, shape (n_bins,)}
    """
    desc = np.asarray(descriptors, dtype=np.float64)
    if desc.ndim > 1:
        desc = desc.mean(axis=1)
    ages = np.asarray(ages, dtype=np.float64)

    # Bin ages.
    age_min, age_max = ages.min(), ages.max()
    bin_edges = np.linspace(age_min, age_max, n_age_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    centile_dict: Dict[float, np.ndarray] = {}
    for pct in percentiles:
        curve = np.zeros(n_age_bins)
        for b in range(n_age_bins):
            mask = (ages >= bin_edges[b]) & (ages < bin_edges[b + 1])
            if mask.sum() > 0:
                curve[b] = np.percentile(desc[mask], pct)
            else:
                curve[b] = np.nan

        # Smooth.
        if smooth:
            curve = _moving_average(curve, smooth_window)

        centile_dict[pct] = curve

    return {"age_centers": bin_centers, "centiles": centile_dict}


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Nan-aware moving average."""
    out = np.copy(x)
    for i in range(len(x)):
        lo = max(0, i - w // 2)
        hi = min(len(x), i + w // 2 + 1)
        window = x[lo:hi]
        valid = window[~np.isnan(window)]
        out[i] = valid.mean() if len(valid) > 0 else np.nan
    return out


# ======================================================================
# §3  INDIVIDUAL DEVIATION SCORING (convenience wrappers)
# ======================================================================

def z_score_map(
    subject: np.ndarray,
    normative_mean: np.ndarray,
    normative_std: np.ndarray,
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


def extreme_value_map(
    z_scores: np.ndarray,
    *,
    threshold: float = 2.0,
) -> np.ndarray:
    """Binary map of extreme deviations.

    Parameters
    ----------
    z_scores : ndarray, shape (N,)
    threshold : float

    Returns
    -------
    ndarray, shape (N,), int
        +1 = above, -1 = below, 0 = within norm.
    """
    z = np.asarray(z_scores)
    result = np.zeros_like(z, dtype=np.int32)
    result[z > threshold] = 1
    result[z < -threshold] = -1
    return result


# ======================================================================
# §4  NON-INFERIORITY & EQUIVALENCE TESTING
# ======================================================================

@dataclass
class NonInferiorityResult:
    """Result of a non-inferiority or equivalence test.

    Attributes
    ----------
    test_type : str
        ``"non_inferiority"`` or ``"equivalence"`` (TOST).
    metric_new : float
        Performance metric of the new method.
    metric_reference : float
        Performance metric of the reference method.
    margin : float
        Pre-specified non-inferiority margin Δ.
    difference : float
        new − reference.
    ci_lower : float
        Lower bound of CI for the difference.
    ci_upper : float
        Upper bound of CI for the difference.
    p_value : float
    is_non_inferior : bool
    is_equivalent : bool
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
        status = "NON-INFERIOR" if self.is_non_inferior else "INCONCLUSIVE"
        if self.is_equivalent:
            status = "EQUIVALENT"
        return (
            f"NonInferiority({status}: new={self.metric_new:.4f}, "
            f"ref={self.metric_reference:.4f}, Δ={self.margin:.4f}, "
            f"diff={self.difference:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
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

    Tests H₀: μ_new − μ_ref ≤ −Δ (new is inferior by ≥ margin)
    against H₁: μ_new − μ_ref > −Δ (new is non-inferior).

    Rejection of H₀ at α = 0.025 (one-sided) establishes
    non-inferiority.

    Parameters
    ----------
    metric_new : ndarray, shape (n_folds,) or (n_bootstrap,)
        Performance metric (e.g. AUC) of the new method across
        cross-validation folds or bootstrap samples.
    metric_reference : ndarray, shape (n_folds,)
        Same for the reference method.
    margin : float
        Non-inferiority margin Δ (in metric units).
        E.g. Δ=0.05 means "new is non-inferior if AUC drops by
        less than 0.05".
    alpha : float
        One-sided significance level (default 0.025 → 95% CI).
    paired : bool
        Paired test (same CV folds) vs unpaired.

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

    # One-sided t-test: H₀: μ_diff ≤ -margin
    t_stat = (mean_diff + margin) / (se + 1e-30)
    df = n - 1
    p_value = 1 - sp_stats.t.cdf(t_stat, df)

    # CI for the difference.
    t_crit = sp_stats.t.ppf(1 - alpha, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    is_ni = ci_lower > -margin

    return NonInferiorityResult(
        test_type="non_inferiority",
        metric_new=float(new.mean()),
        metric_reference=float(ref.mean()),
        margin=margin,
        difference=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_value),
        is_non_inferior=is_ni,
    )


def equivalence_test_tost(
    metric_new: np.ndarray,
    metric_reference: np.ndarray,
    *,
    margin: float = 0.05,
    alpha: float = 0.05,
) -> NonInferiorityResult:
    """Two One-Sided Tests (TOST) for equivalence.

    Tests both H₁: μ_diff > −Δ AND H₂: μ_diff < +Δ.
    Rejection of both establishes equivalence within ±Δ.

    Parameters
    ----------
    metric_new, metric_reference : ndarray
    margin : float
        Equivalence margin (symmetric: ±Δ).
    alpha : float

    Returns
    -------
    NonInferiorityResult
        ``is_equivalent=True`` if both one-sided tests pass.
    """
    new = np.asarray(metric_new, dtype=np.float64)
    ref = np.asarray(metric_reference, dtype=np.float64)
    n = min(len(new), len(ref))
    new, ref = new[:n], ref[:n]

    diff = new - ref
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    df = n - 1

    # Test 1: H₀: μ_diff ≤ -margin  (lower bound)
    t1 = (mean_diff + margin) / (se + 1e-30)
    p1 = 1 - sp_stats.t.cdf(t1, df)

    # Test 2: H₀: μ_diff ≥ +margin  (upper bound)
    t2 = (mean_diff - margin) / (se + 1e-30)
    p2 = sp_stats.t.cdf(t2, df)

    p_tost = max(p1, p2)
    is_equiv = p_tost < alpha

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
        is_equivalent=is_equiv,
    )


def auc_comparison_delong(
    y_true: np.ndarray,
    scores_new: np.ndarray,
    scores_reference: np.ndarray,
) -> Tuple[float, float, float]:
    """DeLong test for comparing two AUCs from paired samples.

    Tests H₀: AUC_new = AUC_ref.

    Parameters
    ----------
    y_true : ndarray, shape (n,)
        True binary labels.
    scores_new : ndarray, shape (n,)
        Predicted scores from the new method.
    scores_reference : ndarray, shape (n,)
        Predicted scores from the reference method.

    Returns
    -------
    auc_new : float
    auc_ref : float
    p_value : float
        Two-sided p-value for the difference.

    References
    ----------
    DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas
    under two or more correlated receiver operating characteristic
    curves. *Biometrics* 44(3):837–845, 1988.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    s_new = np.asarray(scores_new)
    s_ref = np.asarray(scores_reference)

    auc_new = roc_auc_score(y, s_new)
    auc_ref = roc_auc_score(y, s_ref)

    # Simplified DeLong via bootstrap (exact DeLong is complex).
    n_boot = 2000
    rng = np.random.default_rng(42)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            a_new = roc_auc_score(y[idx], s_new[idx])
            a_ref = roc_auc_score(y[idx], s_ref[idx])
            diffs.append(a_new - a_ref)
        except ValueError:
            continue

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return float(auc_new), float(auc_ref), 1.0

    se = diffs.std()
    if se < 1e-10:
        return float(auc_new), float(auc_ref), 1.0

    z = (auc_new - auc_ref) / se
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return float(auc_new), float(auc_ref), float(p_value)


# ======================================================================
# §5  METHOD COMPARISON — spectral vs volumetric
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
        ni = "NI" if self.non_inferiority.is_non_inferior else "?"
        eq = "EQ" if self.equivalence.is_equivalent else "?"
        return (
            f"MethodComparison({self.method_new} vs {self.method_reference}: "
            f"AUC {self.auc_new:.3f} vs {self.auc_reference:.3f} "
            f"(p={self.auc_p_value:.4f}), "
            f"[{ni}] [{eq}])"
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
    seed: Optional[int] = 42,
) -> MethodComparisonResult:
    """Full head-to-head comparison between two methods.

    Runs cross-validated classification with both feature sets,
    compares AUCs (DeLong), and performs non-inferiority + TOST
    equivalence tests.

    Parameters
    ----------
    y_true : ndarray, shape (n,)
        Binary labels.
    features_new : ndarray, shape (n, d_new)
        Features from the new method (spectral descriptors).
    features_reference : ndarray, shape (n, d_ref)
        Features from the reference method (e.g. volumes).
    method_new_name, method_ref_name : str
    n_folds : int
    margin : float
        Non-inferiority margin for AUC comparison.
    seed : int

    Returns
    -------
    MethodComparisonResult
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    y = np.asarray(y_true)
    X_new = np.asarray(features_new)
    X_ref = np.asarray(features_reference)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    aucs_new = []
    aucs_ref = []
    scores_new_all = np.zeros(len(y))
    scores_ref_all = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X_new, y):
        # New method.
        pipe_new = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
        pipe_new.fit(X_new[train_idx], y[train_idx])
        prob_new = pipe_new.predict_proba(X_new[test_idx])[:, 1]
        scores_new_all[test_idx] = prob_new

        # Reference method.
        pipe_ref = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
        pipe_ref.fit(X_ref[train_idx], y[train_idx])
        prob_ref = pipe_ref.predict_proba(X_ref[test_idx])[:, 1]
        scores_ref_all[test_idx] = prob_ref

        from sklearn.metrics import roc_auc_score
        try:
            aucs_new.append(roc_auc_score(y[test_idx], prob_new))
            aucs_ref.append(roc_auc_score(y[test_idx], prob_ref))
        except ValueError:
            pass

    aucs_new = np.array(aucs_new)
    aucs_ref = np.array(aucs_ref)

    # DeLong AUC comparison.
    auc_new_full, auc_ref_full, p_delong = auc_comparison_delong(
        y, scores_new_all, scores_ref_all,
    )

    # Non-inferiority.
    ni = non_inferiority_test(aucs_new, aucs_ref, margin=margin)

    # Equivalence (TOST).
    eq = equivalence_test_tost(aucs_new, aucs_ref, margin=margin)

    # Effect sizes (Cohen's d of the scores).
    d_new = _cohens_d(
        scores_new_all[y == 0], scores_new_all[y == 1],
    )
    d_ref = _cohens_d(
        scores_ref_all[y == 0], scores_ref_all[y == 1],
    )

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
    """Cohen's d between two groups."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
        / (na + nb - 2)
    )
    return float(abs(a.mean() - b.mean()) / (pooled + 1e-30))


# ======================================================================

__all__: List[str] = [
    # Normative model
    "NormativeModel",
    # Centile curves
    "centile_curves",
    # Deviation scoring
    "z_score_map",
    "extreme_value_map",
    # Non-inferiority & equivalence
    "NonInferiorityResult",
    "non_inferiority_test",
    "equivalence_test_tost",
    "auc_comparison_delong",
    # Method comparison
    "MethodComparisonResult",
    "compare_methods",
]
