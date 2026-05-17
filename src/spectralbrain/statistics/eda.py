"""Exploratory data analysis and quality control for spectral morphometry.

Five diagnostic blocks plus the descriptor recommendation engine:

1. **Spectral QC** — validate eigendecomposition quality.
2. **Optimal k** — how many eigenpairs are enough?
3. **Descriptor profiling** — summary statistics, normality, outliers.
4. **Reliability** — ICC test-retest, batch-effect detection.
5. **Report** — integrated markdown/Rich output.
6. **recommend_descriptor()** — surrogate-based descriptor selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DESCRIPTOR_ELIGIBILITY,
    AnalysisObjective,
    DescriptorMatrix,
    DescriptorType,
    GlobalDescriptor,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  SPECTRAL QC
# ======================================================================

@dataclass
class SpectralQCReport:
    """Quality-control diagnostics for a spectral decomposition.

    All fields are populated by :func:`spectral_qc`.
    """

    n_vertices: int = 0
    n_eigenvalues: int = 0
    lambda_0: float = 0.0
    lambda_0_ok: bool = True
    fiedler_value: float = 0.0
    spectral_gap: float = 0.0
    eigenvalues_nonneg: bool = True
    n_negative_eigenvalues: int = 0
    max_negative_eigenvalue: float = 0.0
    orthonormality_error: float = 0.0
    orthonormality_ok: bool = True
    laplacian_row_sum_max: float = 0.0
    laplacian_row_sum_ok: bool = True
    near_degenerate_pairs: int = 0
    recommended_k: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    passed: bool = True

    def __repr__(self) -> str:
        """Return a human-readable summary of the QC report."""
        status = "✓ PASSED" if self.passed else "✗ ISSUES FOUND"
        return (
            f"SpectralQC({status}, N={self.n_vertices}, "
            f"k={self.n_eigenvalues}, λ₀={self.lambda_0:.2e}, "
            f"Fiedler={self.fiedler_value:.4f})"
        )


def spectral_qc(
    decomp: SpectralDecomposition,
    *,
    lambda_0_tol: float = 1e-4,
    ortho_tol: float = 1e-3,
    row_sum_tol: float = 1e-2,
    degeneracy_tol: float = 1e-6,
) -> SpectralQCReport:
    """Run quality-control diagnostics on a spectral decomposition.

    Parameters
    ----------
    decomp : SpectralDecomposition
    lambda_0_tol : float
        Tolerance for λ₀ ≈ 0.
    ortho_tol : float
        Tolerance for M-orthonormality of eigenvectors.
    row_sum_tol : float
        Tolerance for Laplacian row-sum ≈ 0.
    degeneracy_tol : float
        Relative gap below which eigenvalue pairs are flagged
        as near-degenerate.

    Returns
    -------
    SpectralQCReport
    """
    rpt = SpectralQCReport()
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    rpt.n_vertices = decomp.n_vertices
    rpt.n_eigenvalues = decomp.n_eigenvalues
    rpt.lambda_0 = float(evals[0])
    rpt.fiedler_value = float(evals[1]) if len(evals) > 1 else 0.0
    rpt.spectral_gap = decomp.spectral_gap

    # Check λ₀ ≈ 0.
    if abs(evals[0]) > lambda_0_tol:
        rpt.lambda_0_ok = False
        rpt.warnings.append(
            f"λ₀ = {evals[0]:.2e} (expected ≈ 0, tol={lambda_0_tol:.0e})"
        )

    # Check non-negativity.
    neg_mask = evals < -1e-10
    rpt.n_negative_eigenvalues = int(neg_mask.sum())
    if rpt.n_negative_eigenvalues > 0:
        rpt.eigenvalues_nonneg = False
        rpt.max_negative_eigenvalue = float(evals[neg_mask].min())
        rpt.warnings.append(
            f"{rpt.n_negative_eigenvalues} negative eigenvalues "
            f"(min={rpt.max_negative_eigenvalue:.2e})"
        )

    # Check M-orthonormality: Φᵀ M Φ ≈ I.
    if decomp.mass is not None:
        M_dense = decomp.mass
        # Sample a subset of columns for large k.
        k_check = min(decomp.n_eigenvalues, 20)
        Phi = evecs[:, :k_check]
        gram = Phi.T @ (M_dense @ Phi)
        identity = np.eye(k_check)
        rpt.orthonormality_error = float(np.max(np.abs(gram - identity)))
        if rpt.orthonormality_error > ortho_tol:
            rpt.orthonormality_ok = False
            rpt.warnings.append(
                f"Eigenvectors not M-orthonormal "
                f"(max error={rpt.orthonormality_error:.2e})"
            )

    # Check Laplacian row sum.
    if decomp.stiffness is not None:
        row_sums = np.abs(
            np.asarray(decomp.stiffness.sum(axis=1)).ravel()
        )
        rpt.laplacian_row_sum_max = float(row_sums.max())
        if rpt.laplacian_row_sum_max > row_sum_tol:
            rpt.laplacian_row_sum_ok = False
            rpt.warnings.append(
                f"Laplacian row-sum max={rpt.laplacian_row_sum_max:.2e} "
                f"(should be ≈ 0)"
            )

    # Near-degenerate eigenvalue pairs.
    for i in range(1, len(evals) - 1):
        if evals[i] > 1e-10:
            rel_gap = abs(evals[i + 1] - evals[i]) / evals[i]
            if rel_gap < degeneracy_tol:
                rpt.near_degenerate_pairs += 1

    if rpt.near_degenerate_pairs > 0:
        rpt.warnings.append(
            f"{rpt.near_degenerate_pairs} near-degenerate eigenvalue pairs "
            f"(GPS sign ambiguity risk)"
        )

    rpt.passed = (
        rpt.lambda_0_ok
        and rpt.eigenvalues_nonneg
        and rpt.orthonormality_ok
    )

    return rpt


# ======================================================================
# §2  OPTIMAL k SELECTION
# ======================================================================

@dataclass
class OptimalKResult:
    """Recommended number of eigenpairs by multiple criteria."""

    k_elbow: int = 0
    k_energy_95: int = 0
    k_energy_99: int = 0
    k_gap: int = 0
    k_recommended: int = 0
    eigenvalues: Optional[np.ndarray] = None
    cumulative_energy: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """Return a compact summary of the optimal-k recommendation."""
        return (
            f"OptimalK(recommended={self.k_recommended}, "
            f"elbow={self.k_elbow}, energy95={self.k_energy_95}, "
            f"gap={self.k_gap})"
        )


def optimal_k(
    eigenvalues: np.ndarray,
    *,
    energy_thresholds: Tuple[float, float] = (0.95, 0.99),
) -> OptimalKResult:
    """Determine optimal number of eigenpairs.

    Three criteria:
    1. **Elbow** — maximum curvature of log(λ) vs index.
    2. **Energy** — Σᵢλᵢ / Σλ > threshold.
    3. **Max gap** — largest relative gap between consecutive λ.

    Parameters
    ----------
    eigenvalues : ndarray
        Full eigenvalue sequence.
    energy_thresholds : tuple of float
        Thresholds for cumulative energy (default 95% and 99%).

    Returns
    -------
    OptimalKResult
    """
    evals = np.asarray(eigenvalues)
    evals_pos = evals[evals > 1e-10]
    n = len(evals_pos)
    result = OptimalKResult(eigenvalues=evals)

    if n < 3:
        result.k_recommended = n
        return result

    # Cumulative energy.
    total = evals_pos.sum()
    cum = np.cumsum(evals_pos) / total
    result.cumulative_energy = cum

    result.k_energy_95 = int(np.searchsorted(cum, energy_thresholds[0]) + 1)
    result.k_energy_99 = int(np.searchsorted(cum, energy_thresholds[1]) + 1)

    # Elbow: maximum second derivative of log(λ).
    log_lam = np.log(evals_pos + 1e-30)
    d2 = np.diff(log_lam, n=2)
    result.k_elbow = int(np.argmax(np.abs(d2)) + 2)  # +2 for diff offset

    # Max relative gap.
    gaps = np.diff(evals_pos) / (evals_pos[:-1] + 1e-30)
    result.k_gap = int(np.argmax(gaps) + 1)

    # Consensus: median of the three.
    candidates = [result.k_elbow, result.k_energy_95, result.k_gap]
    result.k_recommended = int(np.median(candidates))
    result.k_recommended = max(10, min(result.k_recommended, n))

    return result


# ======================================================================
# §3  DESCRIPTOR PROFILING
# ======================================================================

def descriptor_profile(
    descriptors: Dict[str, np.ndarray],
    *,
    normality_samples: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Summary statistics for each descriptor.

    Parameters
    ----------
    descriptors : dict of {name: ndarray}
        Descriptor arrays (any shape — handles ScalarMap, DescriptorMatrix,
        GlobalDescriptor).
    normality_samples : int
        Subsample size for Shapiro-Wilk test.
    seed : int, optional

    Returns
    -------
    dict of {name: {stat: value}}
        Keys per descriptor: mean, std, min, max, skew, kurtosis,
        q25, q50, q75, shapiro_p, n_outliers_3sigma, shape.
    """
    from scipy.stats import shapiro, skew, kurtosis

    rng = np.random.default_rng(seed)
    profiles: Dict[str, Dict[str, Any]] = {}

    for name, arr in descriptors.items():
        arr = np.asarray(arr, dtype=np.float64)
        flat = arr.ravel()

        p = {
            "shape": arr.shape,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "q25": float(np.percentile(flat, 25)),
            "q50": float(np.percentile(flat, 50)),
            "q75": float(np.percentile(flat, 75)),
            "skewness": float(skew(flat)),
            "kurtosis": float(kurtosis(flat)),
        }

        # Outlier count (beyond 3σ).
        z = (flat - p["mean"]) / (p["std"] + 1e-30)
        p["n_outliers_3sigma"] = int(np.sum(np.abs(z) > 3))
        p["pct_outliers"] = 100 * p["n_outliers_3sigma"] / len(flat)

        # Shapiro-Wilk on subsample.
        n = min(normality_samples, len(flat))
        sample = rng.choice(flat, size=n, replace=False) if len(flat) > n else flat
        try:
            _, p_val = shapiro(sample)
            p["shapiro_p"] = float(p_val)
            p["normally_distributed"] = p_val > 0.05
        except Exception:
            p["shapiro_p"] = None
            p["normally_distributed"] = None

        profiles[name] = p

    return profiles


def descriptor_correlation(
    descriptors: Dict[str, np.ndarray],
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> Tuple[np.ndarray, List[str]]:
    """Correlation matrix between descriptors (redundancy check).

    For multi-column descriptors, uses the mean across columns.

    Parameters
    ----------
    descriptors : dict of {name: ndarray}
    method : str

    Returns
    -------
    corr_matrix : ndarray, shape (D, D)
    names : list of str
    """
    from scipy.stats import spearmanr

    names = sorted(descriptors.keys())
    vectors = []
    for name in names:
        arr = np.asarray(descriptors[name], dtype=np.float64)
        if arr.ndim > 1:
            vectors.append(arr.mean(axis=1))
        else:
            vectors.append(arr)

    # Ensure all same length.
    min_len = min(len(v) for v in vectors)
    mat = np.column_stack([v[:min_len] for v in vectors])

    if method == "pearson":
        corr = np.corrcoef(mat, rowvar=False)
    else:
        corr, _ = spearmanr(mat)
        if corr.ndim == 0:
            corr = np.array([[1.0]])

    return corr, names


# ======================================================================
# §4  TEST-RETEST RELIABILITY
# ======================================================================

def compute_icc(
    test: np.ndarray,
    retest: np.ndarray,
    *,
    icc_type: Literal["ICC2,1", "ICC3,1"] = "ICC3,1",
) -> float:
    """Intraclass Correlation Coefficient for test-retest.

    Parameters
    ----------
    test : ndarray, shape (N,) or (N, T)
        Descriptor values at time 1.
    retest : ndarray, shape (N,) or (N, T)
        Descriptor values at time 2.
    icc_type : str
        ``"ICC2,1"`` — two-way random, single measures.
        ``"ICC3,1"`` — two-way mixed, single measures
        (recommended for neuroimaging).

    Returns
    -------
    float
        ICC value in [-1, 1].  >0.75 = excellent, 0.60–0.75 = good,
        0.40–0.60 = fair, <0.40 = poor.
    """
    test = np.asarray(test, dtype=np.float64).ravel()
    retest = np.asarray(retest, dtype=np.float64).ravel()
    n = min(len(test), len(retest))
    test, retest = test[:n], retest[:n]

    # Two-way ANOVA decomposition.
    k = 2  # two measurements
    grand_mean = (test.mean() + retest.mean()) / 2

    # Between-subjects SS.
    subject_means = (test + retest) / 2
    SS_between = k * np.sum((subject_means - grand_mean) ** 2)

    # Within-subjects SS.
    SS_within = np.sum((test - subject_means) ** 2) + np.sum(
        (retest - subject_means) ** 2
    )

    # Between-measures SS.
    measure_means = np.array([test.mean(), retest.mean()])
    SS_measures = n * np.sum((measure_means - grand_mean) ** 2)

    # Error SS.
    SS_error = SS_within - SS_measures

    # Mean squares.
    MS_between = SS_between / (n - 1)
    MS_within = SS_within / (n * (k - 1))
    MS_measures = SS_measures / (k - 1) if k > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-10

    if icc_type == "ICC3,1":
        # ICC(3,1) = (MS_between - MS_error) / (MS_between + (k-1)·MS_error)
        icc = (MS_between - MS_error) / (MS_between + (k - 1) * MS_error)
    elif icc_type == "ICC2,1":
        icc = (MS_between - MS_error) / (
            MS_between + (k - 1) * MS_error + k * (MS_measures - MS_error) / n
        )
    else:
        raise ValueError(f"Unknown ICC type: {icc_type!r}")

    return float(np.clip(icc, -1.0, 1.0))


def batch_effect_scan(
    descriptors: Dict[str, np.ndarray],
    site_labels: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Scan for batch/site effects in spectral descriptors.

    For each descriptor, tests whether distributions differ
    significantly across sites using Kruskal-Wallis.

    Parameters
    ----------
    descriptors : dict of {name: ndarray}
        Per-subject descriptor values.
    site_labels : ndarray, shape (n_subjects,)
        Site/scanner labels.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict of {name: {statistic, p_value, has_batch_effect, effect_size}}
    """
    from scipy.stats import kruskal

    site_labels = np.asarray(site_labels)
    unique_sites = np.unique(site_labels)
    results: Dict[str, Dict[str, Any]] = {}

    for name, arr in descriptors.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        groups = [arr[site_labels == s] for s in unique_sites]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            results[name] = {
                "statistic": 0.0, "p_value": 1.0,
                "has_batch_effect": False, "effect_size": 0.0,
            }
            continue

        try:
            stat, p_val = kruskal(*groups)
            # Effect size: η² = H / (N - 1)
            N = sum(len(g) for g in groups)
            eta_sq = float(stat / (N - 1)) if N > 1 else 0.0
            results[name] = {
                "statistic": float(stat),
                "p_value": float(p_val),
                "has_batch_effect": p_val < alpha,
                "effect_size_eta2": eta_sq,
            }
        except Exception:
            results[name] = {
                "statistic": 0.0, "p_value": 1.0,
                "has_batch_effect": False, "effect_size_eta2": 0.0,
            }

    return results


def eigenvalue_stability(
    decomps: List[SpectralDecomposition],
    *,
    n_eigenvalues: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Cross-subject eigenvalue stability analysis.

    Parameters
    ----------
    decomps : list of SpectralDecomposition
        Decompositions from multiple subjects.
    n_eigenvalues : int, optional
        Number of eigenvalues to compare.

    Returns
    -------
    dict
        Keys: ``"mean"``, ``"std"``, ``"cv"`` (coefficient of
        variation), ``"eigenvalue_matrix"`` (subjects × k).
    """
    k = n_eigenvalues or min(d.n_eigenvalues for d in decomps)
    matrix = np.array([d.eigenvalues[:k] for d in decomps])  # (S, k)

    mean_evals = matrix.mean(axis=0)
    std_evals = matrix.std(axis=0)
    cv = std_evals / (mean_evals + 1e-30)

    return {
        "mean": mean_evals,
        "std": std_evals,
        "cv": cv,
        "eigenvalue_matrix": matrix,
    }


# ======================================================================
# §5  RECOMMEND_DESCRIPTOR — surrogate-based selection
# ======================================================================

@dataclass
class DescriptorRecommendation:
    """Output of :func:`recommend_descriptor`.

    Attributes
    ----------
    recommended : str
        Name of the top-ranked descriptor.
    objective : str
        The analysis objective used.
    ranking : list of dict
        Top descriptors with scores and metrics.
    surrogate_details : dict
        Information about the surrogates generated.
    """

    recommended: str
    objective: str
    ranking: List[Dict[str, Any]]
    surrogate_details: Dict[str, Any]

    def __repr__(self) -> str:
        """Return a summary showing the top-5 ranked descriptors."""
        top5 = ", ".join(
            f"{r['descriptor']}({r['score']:.3f})"
            for r in self.ranking[:5]
        )
        return (
            f"Recommendation('{self.recommended}' for "
            f"{self.objective}) — top 5: [{top5}]"
        )


def _generate_surrogates(
    points: np.ndarray,
    objective: str,
    *,
    n_surrogates: int = 30,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate synthetic deformations for descriptor evaluation.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Reference geometry (vertex/point coordinates).
    objective : str
    n_surrogates : int
    seed : int

    Returns
    -------
    surrogate_points : list of ndarray
        Deformed point sets.
    labels : ndarray, shape (n_surrogates,)
        0 = control (undeformed), 1 = deformed.
    """
    rng = np.random.default_rng(seed)
    n_half = n_surrogates // 2
    surrogates: List[np.ndarray] = []
    labels = np.zeros(n_surrogates, dtype=np.int32)

    N = points.shape[0]
    centroid = points.mean(axis=0)

    for i in range(n_surrogates):
        if i < n_half:
            # Controls: add small Gaussian noise (no systematic deformation).
            noise = rng.normal(0, 0.01, points.shape)
            surrogates.append(points + noise)
            labels[i] = 0
        else:
            # Deformed: apply objective-specific deformations.
            if objective == "group_discrimination":
                # Focal atrophy: shrink a random subregion.
                center = points[rng.integers(N)]
                dists = np.linalg.norm(points - center, axis=1)
                radius = np.percentile(dists, 30)
                mask = dists < radius
                scale = 0.7 + 0.3 * rng.random()
                deformed = points.copy()
                deformed[mask] = center + (deformed[mask] - center) * scale
                surrogates.append(deformed + rng.normal(0, 0.01, points.shape))

            elif objective == "lateralization":
                # Asymmetric deformation: scale one half differently.
                mid = centroid[0]
                left = points[:, 0] < mid
                deformed = points.copy()
                scale = 0.8 + 0.2 * rng.random()
                deformed[left] *= np.array([scale, 1, 1])
                surrogates.append(deformed + rng.normal(0, 0.01, points.shape))

            elif objective == "longitudinal_change":
                # Progressive uniform shrinkage.
                scale = 0.85 + 0.15 * rng.random()
                deformed = centroid + (points - centroid) * scale
                surrogates.append(deformed + rng.normal(0, 0.01, points.shape))

            elif objective == "subregion_detection":
                # Localised bump (add outward displacement to a patch).
                center = points[rng.integers(N)]
                dists = np.linalg.norm(points - center, axis=1)
                radius = np.percentile(dists, 20)
                mask = dists < radius
                displacement = rng.uniform(0.5, 2.0)
                deformed = points.copy()
                direction = deformed[mask] - centroid
                direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
                deformed[mask] += direction * displacement
                surrogates.append(deformed + rng.normal(0, 0.01, points.shape))

            labels[i] = 1

    return surrogates, labels


def _evaluate_descriptor(
    descriptor_values: List[np.ndarray],
    labels: np.ndarray,
    *,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate a descriptor's discriminative power on surrogates.

    Returns AUC, balanced accuracy, and Cohen's d.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    # Aggregate each surrogate's descriptor to a single vector.
    features = []
    for dv in descriptor_values:
        arr = np.asarray(dv, dtype=np.float64)
        if arr.ndim > 1:
            # Use mean + std per column as global summary.
            feat = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])
        else:
            feat = np.array([arr.mean(), arr.std(), np.median(arr)])
        features.append(feat)

    X = np.array(features)                                  # (n_surr, d)
    y = labels

    # Handle constant features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Remove zero-variance columns.
    var_mask = X_scaled.std(axis=0) > 1e-10
    if not var_mask.any():
        return {"auc": 0.5, "accuracy": 0.5, "cohens_d": 0.0}
    X_scaled = X_scaled[:, var_mask]

    n_splits_actual = min(n_splits, min(np.bincount(y)))
    n_splits_actual = max(2, n_splits_actual)

    try:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        auc_scores = cross_val_score(
            clf, X_scaled, y, cv=n_splits_actual, scoring="roc_auc",
        )
        acc_scores = cross_val_score(
            clf, X_scaled, y, cv=n_splits_actual, scoring="balanced_accuracy",
        )
    except Exception:
        auc_scores = np.array([0.5])
        acc_scores = np.array([0.5])

    # Cohen's d between groups.
    X0 = X_scaled[y == 0]
    X1 = X_scaled[y == 1]
    pooled_std = np.sqrt(
        ((len(X0) - 1) * X0.var(axis=0).mean() +
         (len(X1) - 1) * X1.var(axis=0).mean())
        / (len(X0) + len(X1) - 2)
    )
    cohens_d = float(
        np.abs(X0.mean() - X1.mean()) / (pooled_std + 1e-10)
    )

    return {
        "auc": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "accuracy": float(np.mean(acc_scores)),
        "cohens_d": cohens_d,
    }


def recommend_descriptor(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    objective: Union[str, AnalysisObjective] = "group_discrimination",
    *,
    n_surrogates: int = 30,
    k_eigenpairs: int = 30,
    n_jobs: int = 1,
    seed: Optional[int] = 42,
) -> DescriptorRecommendation:
    """Recommend the best spectral descriptor for an analysis objective.

    Generates synthetic surrogates with controlled deformations,
    computes all eligible descriptors, evaluates each descriptor's
    discriminative power, and ranks by consensus.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Representative geometry (e.g. mean mesh vertices, or one
        subject's point cloud).
    labels : ndarray, optional
        Not used by the surrogate engine (surrogates generate their
        own labels).  Reserved for future data-driven evaluation.
    objective : str or AnalysisObjective
        Analysis goal.  Determines eligible descriptors and
        surrogate deformation type.
    n_surrogates : int
        Number of synthetic shapes to generate.
    k_eigenpairs : int
        Eigenpairs per surrogate decomposition.
    n_jobs : int
        Number of parallel workers for surrogate decomposition.
        ``1`` = sequential (default), ``-1`` = all cores.  Requires
        ``joblib`` when > 1.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    DescriptorRecommendation
        Contains ``.recommended``, ``.ranking`` (top descriptors
        with AUC, accuracy, effect size), and ``.surrogate_details``.

    Notes
    -----
    This function is computationally heavy (30 surrogates × k
    eigenpairs × all descriptors by default).  For large meshes,
    consider using ``n_jobs=-1`` to parallelise the surrogate
    decomposition across CPU cores.

    Examples
    --------
    >>> rec = sb.statistics.recommend_descriptor(
    ...     mesh.vertices,
    ...     objective="group_discrimination",
    ...     n_jobs=-1,
    ... )
    >>> print(rec.recommended)
    'wks'
    >>> print(rec.ranking[:3])
    """
    from spectralbrain.core.base import knn_search
    from spectralbrain.spectral.descriptors import (
        compute_bks,
        compute_gps,
        compute_hks,
        compute_shapedna,
        compute_si_hks,
        compute_wks,
        compute_bates_signatures,
    )

    if isinstance(objective, AnalysisObjective):
        obj_str = objective.value
    else:
        obj_str = objective

    eligible = DESCRIPTOR_ELIGIBILITY.get(obj_str, [])
    if not eligible:
        raise ValueError(
            f"Unknown objective: {obj_str!r}. "
            f"Available: {list(DESCRIPTOR_ELIGIBILITY.keys())}"
        )

    # Map descriptor names to compute functions.
    compute_fns: Dict[str, Callable] = {
        "shapedna": lambda d: compute_shapedna(d, normalize="area"),
        "hks": lambda d: compute_hks(d, n_times=20),
        "si_hks": lambda d: compute_si_hks(d, n_frequencies=6),
        "wks": lambda d: compute_wks(d, n_energies=20),
        "gps": lambda d: compute_gps(d),
        "bates_sp": lambda d: compute_bates_signatures(d, order=2, n_times=5),
        "bks": lambda d: compute_bks(d),
    }

    # Filter to eligible + available.
    active_descs = {
        name: fn
        for name, fn in compute_fns.items()
        if name in eligible
    }

    logger.info(
        "recommend_descriptor: objective='%s', %d eligible, "
        "%d surrogates, k=%d",
        obj_str, len(active_descs), n_surrogates, k_eigenpairs,
    )

    # Generate surrogates.
    surrogates, surr_labels = _generate_surrogates(
        points, obj_str, n_surrogates=n_surrogates, seed=seed,
    )

    # Decompose all surrogates.
    from spectralbrain.core.pointclouds import BrainPointCloud

    def _decompose_single(pts: np.ndarray) -> SpectralDecomposition:
        """Decompose a single surrogate point cloud."""
        pc = BrainPointCloud(pts)
        return pc.decompose(k=k_eigenpairs, laplacian_method="knn")

    decomps: List[SpectralDecomposition] = []
    if n_jobs == 1:
        with progress_simple("Decomposing surrogates", total=n_surrogates) as tick:
            for pts in surrogates:
                decomps.append(_decompose_single(pts))
                tick(1)
    else:
        from spectralbrain.backends.cpu import parallel_map
        decomps = parallel_map(
            _decompose_single,
            surrogates,
            n_jobs=n_jobs,
            description="Decomposing surrogates (parallel)",
        )

    # Compute each descriptor on all surrogates and evaluate.
    scores: List[Dict[str, Any]] = []

    with progress_simple("Evaluating descriptors", total=len(active_descs)) as tick:
        for desc_name, compute_fn in active_descs.items():
            try:
                desc_values = [compute_fn(d) for d in decomps]
                metrics = _evaluate_descriptor(desc_values, surr_labels)
                # Composite score: weighted combination.
                score = (
                    0.5 * metrics["auc"]
                    + 0.3 * metrics["accuracy"]
                    + 0.2 * min(metrics["cohens_d"] / 2.0, 1.0)
                )
                scores.append({
                    "descriptor": desc_name,
                    "score": score,
                    **metrics,
                })
            except Exception as exc:
                logger.warning(
                    "Descriptor '%s' failed: %s", desc_name, exc,
                )
                scores.append({
                    "descriptor": desc_name,
                    "score": 0.0,
                    "auc": 0.0,
                    "accuracy": 0.0,
                    "cohens_d": 0.0,
                    "error": str(exc),
                })
            tick(1)

    # Rank by composite score.
    scores.sort(key=lambda x: x["score"], reverse=True)

    return DescriptorRecommendation(
        recommended=scores[0]["descriptor"] if scores else "hks",
        objective=obj_str,
        ranking=scores,
        surrogate_details={
            "n_surrogates": n_surrogates,
            "n_controls": int((surr_labels == 0).sum()),
            "n_deformed": int((surr_labels == 1).sum()),
            "k_eigenpairs": k_eigenpairs,
            "seed": seed,
        },
    )


# ======================================================================
# §6  __all__
# ======================================================================

__all__: List[str] = [
    # QC
    "SpectralQCReport",
    "spectral_qc",
    # Optimal k
    "OptimalKResult",
    "optimal_k",
    # Descriptor profiling
    "descriptor_profile",
    "descriptor_correlation",
    # Reliability
    "compute_icc",
    "batch_effect_scan",
    "eigenvalue_stability",
    # Recommendation
    "DescriptorRecommendation",
    "recommend_descriptor",
]
