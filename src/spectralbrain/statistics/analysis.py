"""Statistical analysis toolkit for spectral morphometry.

Covers the full analytical pipeline from vertex-wise group comparison
to connectome-level network analysis, including dimension-collapsing
methods for converting per-vertex descriptors into per-shape global
vectors.

Sections
--------
S1  Vertex-wise group comparison (t-test, Mann-Whitney, TFCE, FDR, permutation)
S2  Effect sizes (Cohen's d, Hedges' g -- vertex-wise maps)
S3  Vertex-wise correlation with clinical scores
S4  Surprise / anomaly maps (z-score against normative)
S5  Classification (SVM, LogReg + CV)
S6  Dimension collapsing (Fisher vectors, Bag-of-Spectral-Words, kernel mean embedding)
S7  Dissimilarity measures (EMD, KL, JS divergence, energy distance)
S8  RSA -- Representational Similarity Analysis
S9  Connectome & network analysis (modularity, participation, NBS, Mantel)
S10 Asymmetry analysis (lateralisation indices)
S11 Dimensionality reduction (PCA, MDS, UMAP)
S12 Exploratory data analysis and QC (spectral QC, optimal k, descriptor
    profiling, ICC / batch-effect reliability, eigenvalue stability)
S13 Descriptor recommendation engine (surrogate-based, on meshes)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats as sp_stats

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DESCRIPTOR_ELIGIBILITY,
    AnalysisObjective,
    ConnectomeMatrix,
    DescriptorMatrix,
    DistanceMatrix,
    GlobalDescriptor,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# S1  VERTEX-WISE GROUP COMPARISON
# ======================================================================


@dataclass
class VertexWiseResult:
    """Results of a vertex-wise statistical test."""

    statistic: np.ndarray  # (N,) test statistic per vertex
    p_values: np.ndarray  # (N,) uncorrected p-values
    p_corrected: np.ndarray  # (N,) corrected p-values
    correction: str  # method used
    significant: np.ndarray  # (N,) bool mask at alpha
    alpha: float
    effect_size: np.ndarray | None = None  # (N,) Cohen's d

    @property
    def n_significant(self) -> int:
        """Return the count of significant results after correction."""
        return int(self.significant.sum())

    def __repr__(self) -> str:
        """Return a compact summary string."""
        return (
            f"VertexWiseResult({self.n_significant} significant / "
            f"{len(self.statistic)} vertices, "
            f"correction='{self.correction}', alpha={self.alpha})"
        )


def vertexwise_ttest(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    correction: Literal["fdr", "bonferroni", "none"] = "fdr",
    alpha: float = 0.05,
    equal_var: bool = False,
) -> VertexWiseResult:
    """Independent two-sample t-test at each vertex (vectorised).

    Parameters
    ----------
    group_a : ndarray, shape (n_a, N) or (n_a, N, T)
        Descriptor values for group A (subjects x vertices [x scales]).
        If 3D, tests are run on the mean across the last axis.
    group_b : ndarray, shape (n_b, N)
        Descriptor values for group B.
    correction : str
        ``"fdr"`` -- Benjamini-Hochberg; ``"bonferroni"``; ``"none"``.
    alpha : float
    equal_var : bool
        If ``False`` (default), Welch's t-test (does not assume equal
        variances) -- the safer default for groups with unequal size or
        spread. If ``True``, Student's pooled-variance t-test.

    Returns
    -------
    VertexWiseResult
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if b.ndim == 3:
        b = b.mean(axis=-1)

    # Vectorised across vertices (axis=0 = subjects).
    t_stat, p_vals = sp_stats.ttest_ind(a, b, axis=0, equal_var=equal_var)
    # Constant-across-subjects vertices yield NaN -- treat as null.
    t_stat = np.nan_to_num(np.asarray(t_stat), nan=0.0)
    p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)

    p_corr = _correct_pvalues(p_vals, method=correction)
    d = _cohens_d_arrays(a, b)

    return VertexWiseResult(
        statistic=t_stat,
        p_values=p_vals,
        p_corrected=p_corr,
        correction=correction,
        significant=p_corr < alpha,
        alpha=alpha,
        effect_size=d,
    )


def vertexwise_mannwhitney(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    correction: Literal["fdr", "bonferroni", "none"] = "fdr",
    alpha: float = 0.05,
) -> VertexWiseResult:
    """Non-parametric Mann-Whitney U test at each vertex.

    Parameters
    ----------
    group_a, group_b : ndarray, shape (n, N)
    correction, alpha : as above.

    Returns
    -------
    VertexWiseResult
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if b.ndim == 3:
        b = b.mean(axis=-1)

    # Vectorised across vertices (scipy >= 1.7 supports axis=).
    u_stat, p_vals = sp_stats.mannwhitneyu(a, b, alternative="two-sided", axis=0)
    u_stat = np.nan_to_num(np.asarray(u_stat), nan=0.0)
    p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)

    p_corr = _correct_pvalues(p_vals, method=correction)

    return VertexWiseResult(
        statistic=u_stat,
        p_values=p_vals,
        p_corrected=p_corr,
        correction=correction,
        significant=p_corr < alpha,
        alpha=alpha,
    )


def vertexwise_permutation(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    n_permutations: int = 5000,
    stat_func: Literal["t", "mean_diff"] = "t",
    correction: Literal["max", "fdr", "none"] = "max",
    seed: int | None = None,
    alpha: float = 0.05,
) -> VertexWiseResult:
    """Permutation test at each vertex with multiple-comparison control.

    The label permutation builds an exact (non-parametric) null. Crucially,
    the per-vertex permutation p-value is **not** corrected for multiple
    comparisons on its own. This function offers proper correction:

    - ``correction="max"`` (default): family-wise error rate (FWER) control
      via the **maximum-statistic** null distribution (Westfall & Young;
      Nichols & Holmes 2002). On each permutation the maximum of
      ``|statistic|`` across all vertices is recorded; a vertex is
      significant if its observed statistic exceeds the
      ``(1-alpha)`` quantile of that null. This is the standard rigorous
      correction for vertex/voxel-wise permutation testing.
    - ``correction="fdr"``: per-vertex permutation p-values, then
      Benjamini-Hochberg.
    - ``correction="none"``: raw per-vertex permutation p-values.

    Parameters
    ----------
    group_a, group_b : ndarray, shape (n, N)
    n_permutations : int
    stat_func : ``"t"`` or ``"mean_diff"``
    correction : ``"max"``, ``"fdr"``, or ``"none"``
    seed : int, optional
    alpha : float

    Returns
    -------
    VertexWiseResult
        ``p_values`` are always the raw per-vertex permutation p-values;
        ``p_corrected`` reflects the chosen correction. For ``"max"``,
        ``p_corrected`` is the FWER-adjusted p-value (fraction of the
        max-null at or above each observed statistic).

    References
    ----------
    Nichols TE, Holmes AP. Nonparametric permutation tests for functional
    neuroimaging. *Hum Brain Mapp* 15(1):1-25, 2002.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if b.ndim == 3:
        b = b.mean(axis=-1)

    rng = np.random.default_rng(seed)
    combined = np.vstack([a, b])
    n_a, N = a.shape
    n_total = combined.shape[0]

    def _stat(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        if stat_func == "t":
            t, _ = sp_stats.ttest_ind(xa, xb, axis=0, equal_var=False)
            return np.nan_to_num(np.asarray(t), nan=0.0)
        return xa.mean(axis=0) - xb.mean(axis=0)

    obs = _stat(a, b)
    abs_obs = np.abs(obs)

    count_extreme = np.zeros(N, dtype=np.int64)  # per-vertex (uncorrected)
    count_max = np.zeros(N, dtype=np.int64)  # FWER via max-statistic
    with progress_simple("Permutation test", total=n_permutations) as tick:
        for _ in range(n_permutations):
            perm = rng.permutation(n_total)
            perm_stat = _stat(combined[perm[:n_a]], combined[perm[n_a:]])
            abs_perm = np.abs(perm_stat)
            count_extreme += (abs_perm >= abs_obs).astype(np.int64)
            count_max += (abs_perm.max() >= abs_obs).astype(np.int64)
            tick(1)

    # Raw per-vertex permutation p-values (add-one for validity).
    p_vals = (count_extreme + 1) / (n_permutations + 1)

    if correction == "max":
        p_corr = (count_max + 1) / (n_permutations + 1)
        corr_label = "permutation-max (FWER)"
    elif correction == "fdr":
        p_corr = _fdr_bh(p_vals)
        corr_label = "permutation+fdr"
    else:
        p_corr = p_vals.copy()
        corr_label = "permutation (uncorrected)"

    return VertexWiseResult(
        statistic=obs,
        p_values=p_vals,
        p_corrected=p_corr,
        correction=corr_label,
        significant=p_corr < alpha,
        alpha=alpha,
    )


def tfce(
    statistic_map: np.ndarray,
    adjacency: Any,
    *,
    E: float = 0.5,
    H: float = 2.0,
    n_steps: int = 100,
) -> np.ndarray:
    """Threshold-Free Cluster Enhancement (Smith & Nichols 2009).

    Enhances a statistical map by integrating cluster extent and
    height across all thresholds.

    TFCE(v) = int_0^h(v) e(h)^E * h^H dh

    Parameters
    ----------
    statistic_map : ndarray, shape (N,)
        Vertex-wise test statistic (e.g. t-values).
    adjacency : sparse matrix, shape (N, N)
        Vertex adjacency (from mesh or kNN graph).
    E : float
        Cluster extent exponent (default 0.5).
    H : float
        Height exponent (default 2.0).
    n_steps : int
        Number of threshold steps for numerical integration.

    Returns
    -------
    ndarray, shape (N,)
        Signed TFCE-enhanced statistic map: positive and negative tails are
        enhanced separately (never merged) and the sign of the input is kept.

    References
    ----------
    Smith SM, Nichols TE. Threshold-free cluster enhancement.
    *NeuroImage* 44(1):83-98, 2009.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    adj = sp.csr_matrix(adjacency)
    stat_full = np.asarray(statistic_map, dtype=np.float64)

    def _tfce_one_sign(stat: np.ndarray) -> np.ndarray:
        """TFCE of a non-negative map (one tail)."""
        out = np.zeros_like(stat, dtype=np.float64)
        max_stat = stat.max() if stat.size else 0.0
        if max_stat < 1e-10:
            return out
        thresholds = np.linspace(0, max_stat, n_steps + 1)[1:]
        dh = thresholds[1] - thresholds[0] if len(thresholds) > 1 else max_stat
        for h in thresholds:
            mask = stat >= h
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            sub_adj = adj[idx][:, idx]
            n_comp, comp_labels = connected_components(sub_adj, directed=False)
            extents = np.bincount(comp_labels, minlength=n_comp)
            out[idx] += (extents[comp_labels] ** E) * (h**H) * dh
        return out

    # Positive and negative tails are enhanced separately so that adjacent
    # vertices of opposite sign are never merged into one cluster.
    pos = _tfce_one_sign(np.clip(stat_full, 0.0, None))
    neg = _tfce_one_sign(np.clip(-stat_full, 0.0, None))
    return pos - neg


def _correct_pvalues(
    p_values: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply multiple comparison correction (NaN p-values are left as NaN
    and excluded from the family size)."""
    p_values = np.asarray(p_values, dtype=np.float64)
    if method == "none":
        return p_values.copy()
    elif method == "bonferroni":
        m = int(np.isfinite(p_values).sum())
        return np.minimum(p_values * max(m, 1), 1.0)
    elif method == "fdr":
        return _fdr_bh(p_values)
    raise ValueError(f"Unknown correction: {method!r}")


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Non-finite p-values are ignored (returned as NaN) instead of poisoning
    the step-up cumulative minimum for every other test.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    result = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    pv = p_values[finite]
    n = pv.size
    if n == 0:
        return result
    order = np.argsort(pv)
    ranked_p = pv[order]
    adjusted = ranked_p * n / (np.arange(1, n + 1))
    # Enforce monotonicity.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(adjusted, 1.0)
    result[finite] = out
    return result


# ======================================================================
# S2  EFFECT SIZES
# ======================================================================


def _cohens_d_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cohen's d per column (vertex)."""
    na, nb = a.shape[0], b.shape[0]
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (ma - mb) / (pooled + 1e-30)


def cohens_d_map(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> ScalarMap:
    """Vertex-wise Cohen's d effect-size map.

    Parameters
    ----------
    group_a, group_b : ndarray, shape (n, N)

    Returns
    -------
    ndarray, shape (N,)
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if b.ndim == 3:
        b = b.mean(axis=-1)
    return _cohens_d_arrays(a, b)


def hedges_g_map(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> ScalarMap:
    """Vertex-wise Hedges' g (bias-corrected Cohen's d).

    Parameters
    ----------
    group_a, group_b : ndarray, shape (n, N)

    Returns
    -------
    ndarray, shape (N,)
    """
    d = cohens_d_map(group_a, group_b)
    n = group_a.shape[0] + group_b.shape[0]
    # Correction factor J.
    J = 1 - 3 / (4 * (n - 2) - 1)
    return d * J


# ======================================================================
# S3  VERTEX-WISE CORRELATION
# ======================================================================


def vertexwise_correlation(
    descriptors: np.ndarray,
    scores: np.ndarray,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
    correction: str = "fdr",
    alpha: float = 0.05,
    covariates: np.ndarray | None = None,
) -> VertexWiseResult:
    """Correlate a per-vertex descriptor with a clinical score.

    Parameters
    ----------
    descriptors : ndarray, shape (S, N) or (S, N, T)
        Per-subject descriptor values.  If 3D, averaged over T.
    scores : ndarray, shape (S,)
        Clinical score per subject.
    method : str
    correction : str
    alpha : float
    covariates : ndarray, shape (S, C), optional
        If given, partial correlation controlling for covariates.

    Returns
    -------
    VertexWiseResult
    """
    desc = np.asarray(descriptors, dtype=np.float64)
    if desc.ndim == 3:
        desc = desc.mean(axis=-1)
    scores = np.asarray(scores, dtype=np.float64).ravel()

    cov = None
    n_cov = 0
    if covariates is not None:
        cov = np.asarray(covariates, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov[:, None]
        n_cov = cov.shape[1]

    # Subjects with a missing score or covariate are dropped (explicitly).
    row_ok = np.isfinite(scores)
    if cov is not None:
        row_ok &= np.all(np.isfinite(cov), axis=1)
    if not row_ok.all():
        warnings.warn(
            f"vertexwise_correlation: dropping {int((~row_ok).sum())} subject(s) "
            "with non-finite score/covariates.",
            RuntimeWarning,
            stacklevel=2,
        )
        desc = desc[row_ok]
        scores = scores[row_ok]
        if cov is not None:
            cov = cov[row_ok]

    N = desc.shape[1]
    r_vals = np.zeros(N, dtype=np.float64)
    p_vals = np.ones(N, dtype=np.float64)

    col_complete = np.all(np.isfinite(desc), axis=0)
    if col_complete.any():
        r, p = _corr_block(desc[:, col_complete], scores, cov, method)
        r_vals[col_complete] = r
        p_vals[col_complete] = p
    incomplete = np.where(~col_complete)[0]
    if incomplete.size:
        warnings.warn(
            f"vertexwise_correlation: {incomplete.size} vertex column(s) contain "
            "non-finite values; they are tested on their available subjects only.",
            RuntimeWarning,
            stacklevel=2,
        )
        for v in incomplete:
            ok = np.isfinite(desc[:, v])
            if ok.sum() < 3 + n_cov:
                r_vals[v], p_vals[v] = np.nan, np.nan
                continue
            r, p = _corr_block(
                desc[ok, v : v + 1], scores[ok], cov[ok] if cov is not None else None, method
            )
            r_vals[v], p_vals[v] = r[0], p[0]

    p_corr = _correct_pvalues(p_vals, method=correction)

    return VertexWiseResult(
        statistic=r_vals,
        p_values=p_vals,
        p_corrected=p_corr,
        correction=correction,
        significant=np.nan_to_num(p_corr, nan=1.0) < alpha,
        alpha=alpha,
    )


def _corr_block(
    desc: np.ndarray, scores: np.ndarray, cov: np.ndarray | None, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson/Spearman (partial) correlation of each column with ``scores``.

    For Spearman with covariates the data are rank-transformed first and the
    ranks are then residualised (partial Spearman).
    """
    S = desc.shape[0]
    n_cov = 0 if cov is None else cov.shape[1]
    if method == "spearman":
        x = _rank_columns(desc)
        y = sp_stats.rankdata(scores)
    else:
        x = desc
        y = scores
    if cov is not None:
        y = _residualise(y, cov)
        x = _residualise_columns(x, cov)

    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r_vals = np.where(denom > 1e-30, (xc * yc[:, None]).sum(axis=0) / denom, 0.0)
    r_vals = np.clip(r_vals, -1.0, 1.0)

    df = S - 2 - n_cov
    if df <= 0:
        raise ValueError(
            f"Not enough samples for {n_cov} covariates: df = {df} (need S > {2 + n_cov})."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = r_vals * np.sqrt(df / (1.0 - r_vals**2))
    t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=np.inf, neginf=-np.inf)
    p_vals = 2.0 * sp_stats.t.sf(np.abs(t_stat), df)
    return r_vals, p_vals


def _residualise_columns(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residualise every column of ``Y`` (S, N) on design ``X`` (S, C)."""
    Xd = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    return Y - Xd @ beta


def _rank_columns(Y: np.ndarray) -> np.ndarray:
    """Column-wise rank transform (for Spearman)."""
    return sp_stats.rankdata(Y, axis=0)


def _residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals: y - X @ (X^+ @ y)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


# ======================================================================
# S4  SURPRISE / ANOMALY MAPS
# ======================================================================


def surprise_map(
    subject_descriptor: np.ndarray,
    normative_mean: np.ndarray,
    normative_std: np.ndarray,
) -> ScalarMap:
    """Z-score anomaly map against a normative distribution.

    Parameters
    ----------
    subject_descriptor : ndarray, shape (N,) or (N, T)
    normative_mean : ndarray, same shape
    normative_std : ndarray, same shape

    Returns
    -------
    ndarray, same shape
        Z-scores: positive = above normative, negative = below.
    """
    z = (subject_descriptor - normative_mean) / (normative_std + 1e-30)
    return z


def surprise_map_percentile(
    subject_descriptor: np.ndarray,
    normative_distribution: np.ndarray,
) -> ScalarMap:
    """Percentile-based anomaly map.

    Parameters
    ----------
    subject_descriptor : ndarray, shape (N,)
    normative_distribution : ndarray, shape (S, N)
        Normative values from S reference subjects.

    Returns
    -------
    ndarray, shape (N,)
        Percentile rank (0-100) of subject relative to normative.
    """
    subj = np.asarray(subject_descriptor)
    norm = np.asarray(normative_distribution)
    N = subj.shape[0]
    pctile = np.zeros(N)
    for v in range(N):
        pctile[v] = sp_stats.percentileofscore(norm[:, v], subj[v])
    return pctile


# ======================================================================
# S5  CLASSIFICATION
# ======================================================================


@dataclass
class ClassificationResult:
    """Output of a classification analysis."""

    accuracy: float
    accuracy_std: float
    auc: float
    auc_std: float
    feature_importance: np.ndarray | None
    confusion_matrix: np.ndarray | None
    model_name: str

    def __repr__(self) -> str:
        """Return a compact summary string."""
        return (
            f"Classification({self.model_name}: "
            f"AUC={self.auc:.3f}+/-{self.auc_std:.3f}, "
            f"Acc={self.accuracy:.3f}+/-{self.accuracy_std:.3f})"
        )


def classify(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model: Literal["svm", "logistic", "random_forest"] = "svm",
    n_folds: int = 5,
    seed: int | None = 42,
) -> ClassificationResult:
    """Cross-validated classification with feature importance.

    Parameters
    ----------
    features : ndarray, shape (S, d)
    labels : ndarray, shape (S,)
    model : str
    n_folds : int
    seed : int

    Returns
    -------
    ClassificationResult
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "svm":
        from sklearn.svm import SVC

        clf = SVC(kernel="linear", probability=True, random_state=seed)
    elif model == "logistic":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    auc_scores = cross_val_score(pipe, features, labels, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(pipe, features, labels, cv=cv, scoring="balanced_accuracy")

    # Fit on full data for feature importance.
    pipe.fit(features, labels)
    importance = None
    if model in ("svm", "logistic"):
        importance = np.abs(pipe.named_steps["clf"].coef_).ravel()
    elif model == "random_forest":
        importance = pipe.named_steps["clf"].feature_importances_

    return ClassificationResult(
        accuracy=float(acc_scores.mean()),
        accuracy_std=float(acc_scores.std()),
        auc=float(auc_scores.mean()),
        auc_std=float(auc_scores.std()),
        feature_importance=importance,
        confusion_matrix=None,
        model_name=model,
    )


# ======================================================================
# S6  DIMENSION COLLAPSING -- from per-vertex to per-shape
# ======================================================================


def fisher_vector(
    descriptor: DescriptorMatrix,
    gmm_means: np.ndarray,
    gmm_covs: np.ndarray,
    gmm_weights: np.ndarray,
) -> GlobalDescriptor:
    """Fisher vector encoding of per-vertex descriptors.

    Projects a per-vertex descriptor distribution onto the gradient
    of a Gaussian Mixture Model, producing a fixed-length global
    vector regardless of the number of vertices.

    Parameters
    ----------
    descriptor : ndarray, shape (N, T)
        Per-vertex descriptor matrix.
    gmm_means : ndarray, shape (K, T)
        GMM component means.
    gmm_covs : ndarray, shape (K, T)
        GMM diagonal covariances.
    gmm_weights : ndarray, shape (K,)
        GMM component weights (sum to 1).

    Returns
    -------
    ndarray, shape (2*K*T,)
        Fisher vector (concatenation of first and second order
        gradient statistics).

    References
    ----------
    Perronnin F, Dance C. Fisher kernels on visual vocabularies for
    image categorization. *CVPR 2007*.
    Sanchez J, Perronnin F, Mensink T, Verbeek J. Image classification
    with the Fisher vector. *IJCV* 105(3):222-245, 2013.
    """
    N, _T = descriptor.shape
    K = gmm_means.shape[0]

    # Responsibilities: gamma(n, k) = P(k|x_n)
    log_resp = np.zeros((N, K))
    for k in range(K):
        diff = descriptor - gmm_means[k]
        log_resp[:, k] = (
            np.log(gmm_weights[k] + 1e-30)
            - 0.5 * np.sum(diff**2 / (gmm_covs[k] + 1e-30), axis=1)
            - 0.5 * np.sum(np.log(gmm_covs[k] + 1e-30))
        )
    # Normalise responsibilities.
    log_resp -= log_resp.max(axis=1, keepdims=True)
    resp = np.exp(log_resp)
    resp /= resp.sum(axis=1, keepdims=True)

    fv_parts = []
    for k in range(K):
        gamma_k = resp[:, k]  # (N,)
        sqrt_w = np.sqrt(gmm_weights[k] + 1e-30)
        diff = descriptor - gmm_means[k]  # (N, T)
        sigma = gmm_covs[k]  # (T,)

        # First-order gradient (mean).
        g_mean = (1 / (N * sqrt_w)) * np.sum(
            gamma_k[:, None] * diff / (np.sqrt(sigma) + 1e-30), axis=0
        )
        # Second-order gradient (variance).
        g_var = (1 / (N * sqrt_w * np.sqrt(2))) * np.sum(
            gamma_k[:, None] * (diff**2 / (sigma + 1e-30) - 1), axis=0
        )
        fv_parts.extend([g_mean, g_var])

    fv = np.concatenate(fv_parts)

    # L2 normalisation + power normalisation.
    fv = np.sign(fv) * np.sqrt(np.abs(fv))  # power norm
    norm = np.linalg.norm(fv)
    if norm > 1e-10:
        fv /= norm

    return fv


def fit_gmm_codebook(
    all_descriptors: np.ndarray,
    n_components: int = 32,
    *,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a GMM codebook on pooled descriptors from a population.

    Parameters
    ----------
    all_descriptors : ndarray, shape (N_total, T)
        Pooled descriptors from all subjects.
    n_components : int
        Number of GMM components.
    seed : int

    Returns
    -------
    means : ndarray, shape (K, T)
    covariances : ndarray, shape (K, T)
        Diagonal covariances.
    weights : ndarray, shape (K,)
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=seed,
        max_iter=200,
    )
    gmm.fit(all_descriptors)
    return gmm.means_, gmm.covariances_, gmm.weights_


def bag_of_spectral_words(
    descriptor: DescriptorMatrix,
    codebook: np.ndarray,
    *,
    soft: bool = True,
    sigma: float | None = None,
) -> GlobalDescriptor:
    """Bag-of-Words encoding of per-vertex descriptors.

    Parameters
    ----------
    descriptor : ndarray, shape (N, T)
    codebook : ndarray, shape (K, T)
        Cluster centres (from k-means on pooled descriptors).
    soft : bool
        Soft assignment (Gaussian weighted) vs hard assignment.
    sigma : float, optional
        Bandwidth for soft assignment.  ``None`` = auto.

    Returns
    -------
    ndarray, shape (K,)
        Normalised histogram over codebook words.
    """
    from scipy.spatial.distance import cdist

    dists = cdist(descriptor, codebook, metric="euclidean")  # (N, K)

    if soft:
        if sigma is None:
            sigma = float(np.median(dists))
        weights = np.exp(-(dists**2) / (2 * sigma**2))
        weights /= weights.sum(axis=1, keepdims=True)
        histogram = weights.sum(axis=0)
    else:
        assignments = np.argmin(dists, axis=1)
        histogram = np.bincount(assignments, minlength=codebook.shape[0]).astype(np.float64)

    # L1 normalisation.
    histogram /= histogram.sum() + 1e-30
    return histogram


def kernel_mean_embedding(
    descriptor: DescriptorMatrix,
    *,
    kernel: Literal["rbf", "linear"] = "rbf",
    sigma: float | None = None,
    n_landmarks: int = 100,
    seed: int | None = None,
) -> GlobalDescriptor:
    """Kernel mean embedding of a descriptor distribution.

    Embeds the empirical distribution of per-vertex descriptors into
    an RKHS, approximated by random Fourier features (Rahimi &
    Recht 2007) for scalability.

    Parameters
    ----------
    descriptor : ndarray, shape (N, T)
    kernel : str
    sigma : float, optional
    n_landmarks : int
        Number of random Fourier features.
    seed : int

    Returns
    -------
    ndarray, shape (n_landmarks,)
        Approximate kernel mean embedding.
    """
    rng = np.random.default_rng(seed)
    N, T = descriptor.shape

    if sigma is None:
        from scipy.spatial.distance import pdist

        sample = descriptor[rng.choice(N, min(200, N), replace=False)]
        sigma = float(np.median(pdist(sample))) if len(sample) > 1 else 1.0
        sigma = max(sigma, 1e-6)

    if kernel == "rbf":
        # Random Fourier features.
        W = rng.normal(0, 1 / sigma, (T, n_landmarks))
        b = rng.uniform(0, 2 * np.pi, n_landmarks)
        Z = np.sqrt(2 / n_landmarks) * np.cos(descriptor @ W + b)
        return Z.mean(axis=0)
    elif kernel == "linear":
        return descriptor.mean(axis=0)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")


# ======================================================================
# S7  DISSIMILARITY MEASURES
# ======================================================================


def emd_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Earth Mover's Distance (1D Wasserstein) between distributions.

    For multi-dimensional descriptors, averages across columns.

    Parameters
    ----------
    a, b : ndarray, shape (N,) or (N, T)

    Returns
    -------
    float
    """
    from scipy.stats import wasserstein_distance

    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        return float(wasserstein_distance(a, b))

    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    T = min(a.shape[1], b.shape[1])
    return float(np.mean([wasserstein_distance(a[:, t], b[:, t]) for t in range(T)]))


def kl_divergence(a: np.ndarray, b: np.ndarray, *, bins: int = 50) -> float:
    """KL divergence estimated via histogram binning.

    Parameters
    ----------
    a, b : ndarray, shape (N,)
    bins : int

    Returns
    -------
    float
        D_KL(a || b).
    """
    a, b = np.ravel(a), np.ravel(b)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(a, bins=edges, density=True)[0] + 1e-10
    q = np.histogram(b, bins=edges, density=True)[0] + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(a: np.ndarray, b: np.ndarray, **kwargs) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2]).

    Parameters
    ----------
    a, b : ndarray

    Returns
    -------
    float
    """
    bins = kwargs.pop("bins", 50)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
    a, b = np.ravel(a), np.ravel(b)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(a, bins=edges)[0].astype(np.float64)
    q = np.histogram(b, bins=edges)[0].astype(np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        nz = x > 0
        return float(np.sum(x[nz] * np.log(x[nz] / y[nz])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance between two multivariate samples.

    Parameters
    ----------
    a : ndarray, shape (N_a, d)
    b : ndarray, shape (N_b, d)

    Returns
    -------
    float
    """
    from scipy.spatial.distance import cdist

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    ab = cdist(a, b).mean()
    aa = cdist(a, a).mean()
    bb = cdist(b, b).mean()
    return float(2 * ab - aa - bb)


# ======================================================================
# S8  RSA -- Representational Similarity Analysis
# ======================================================================


def rdm(
    features: np.ndarray,
    *,
    metric: Literal["correlation", "euclidean", "cosine"] = "correlation",
) -> DistanceMatrix:
    """Representational Dissimilarity Matrix.

    Parameters
    ----------
    features : ndarray, shape (S, d)
        S items x d features.
    metric : str

    Returns
    -------
    ndarray, shape (S, S)
    """
    from scipy.spatial.distance import pdist, squareform

    if metric == "correlation":
        D = pdist(features, metric="correlation")
    elif metric == "euclidean":
        D = pdist(features, metric="euclidean")
    elif metric == "cosine":
        D = pdist(features, metric="cosine")
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    return squareform(D)


def rsa_compare(
    rdm_a: DistanceMatrix,
    rdm_b: DistanceMatrix,
    *,
    method: Literal["spearman", "pearson", "kendall"] = "spearman",
    permutations: int = 0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compare two RDMs via Representational Similarity Analysis.

    Parameters
    ----------
    rdm_a, rdm_b : ndarray, shape (S, S)
        Representational Dissimilarity Matrices.
    method : str
        Correlation method.
    permutations : int
        If > 0, compute p-value via permutation test.
    seed : int

    Returns
    -------
    r : float
        Correlation between upper triangles.
    p_value : float
        p-value (parametric if permutations=0, permutation otherwise).
    """
    a_upper = rdm_a[np.triu_indices_from(rdm_a, k=1)]
    b_upper = rdm_b[np.triu_indices_from(rdm_b, k=1)]

    if method == "spearman":
        r_obs, p_param = sp_stats.spearmanr(a_upper, b_upper)
    elif method == "pearson":
        r_obs, p_param = sp_stats.pearsonr(a_upper, b_upper)
    elif method == "kendall":
        r_obs, p_param = sp_stats.kendalltau(a_upper, b_upper)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if permutations <= 0:
        return float(r_obs), float(p_param)

    # Permutation test (row/column shuffle of one RDM).
    rng = np.random.default_rng(seed)
    count = 0
    n = rdm_a.shape[0]
    for _ in range(permutations):
        perm = rng.permutation(n)
        rdm_perm = rdm_a[np.ix_(perm, perm)]
        perm_upper = rdm_perm[np.triu_indices(n, k=1)]
        if method == "spearman":
            r_perm, _ = sp_stats.spearmanr(perm_upper, b_upper)
        elif method == "pearson":
            r_perm, _ = sp_stats.pearsonr(perm_upper, b_upper)
        else:
            r_perm, _ = sp_stats.kendalltau(perm_upper, b_upper)
        if abs(r_perm) >= abs(r_obs):
            count += 1

    p_perm = (count + 1) / (permutations + 1)
    return float(r_obs), float(p_perm)


def mantel_test(
    matrix_a: DistanceMatrix,
    matrix_b: DistanceMatrix,
    *,
    n_permutations: int = 5000,
    method: Literal["pearson", "spearman", "kendall"] = "spearman",
    seed: int | None = None,
) -> tuple[float, float]:
    """Mantel test -- correlation between two distance matrices.

    Tests whether two distance matrices are correlated by comparing
    the observed correlation to a null distribution generated by
    row/column permutation.

    Parameters
    ----------
    matrix_a, matrix_b : ndarray, shape (N, N)
    n_permutations : int
    method : str
    seed : int

    Returns
    -------
    r : float
    p_value : float
    """
    return rsa_compare(
        matrix_a,
        matrix_b,
        method=method,
        permutations=n_permutations,
        seed=seed,
    )


# ======================================================================
# S9  CONNECTOME & NETWORK ANALYSIS
# ======================================================================


def modularity(
    connectome: ConnectomeMatrix,
    community_labels: np.ndarray,
    *,
    gamma: float = 1.0,
) -> float:
    """Newman's modularity Q for a given community partition.

    Q = (1/2m) Sigma_{ij} [A_{ij} - gamma*k_i*k_j/(2m)] * delta(c_i, c_j)

    Parameters
    ----------
    connectome : ndarray, shape (R, R)
        Similarity matrix (higher = more similar).
    community_labels : ndarray, shape (R,)
        Community assignment per node.
    gamma : float
        Resolution parameter.

    Returns
    -------
    float
        Modularity Q.
    """
    A = np.array(connectome, dtype=np.float64, copy=True)  # never mutate caller
    np.fill_diagonal(A, 0)
    m2 = A.sum()
    if m2 < 1e-10:
        return 0.0

    k = A.sum(axis=1)
    Q = 0.0
    for i in range(len(A)):
        for j in range(len(A)):
            if community_labels[i] == community_labels[j]:
                Q += A[i, j] - gamma * k[i] * k[j] / m2

    return float(Q / m2)


def participation_coefficient(
    connectome: ConnectomeMatrix,
    community_labels: np.ndarray,
) -> np.ndarray:
    """Participation coefficient per node.

    PC_i = 1 - Sigma_k (s_{ik} / s_i)^2

    High PC -> hub connected to multiple communities.
    Low PC -> provincial node within one community.

    Parameters
    ----------
    connectome : ndarray, shape (R, R)
    community_labels : ndarray, shape (R,)

    Returns
    -------
    ndarray, shape (R,)
    """
    A = np.array(connectome, dtype=np.float64, copy=True)  # never mutate caller
    np.fill_diagonal(A, 0)
    labels = np.asarray(community_labels)
    communities = np.unique(labels)

    s_total = A.sum(axis=1)  # (R,)
    PC = np.ones(len(A))

    for c in communities:
        mask = labels == c
        s_c = A[:, mask].sum(axis=1)  # (R,)
        ratio = s_c / (s_total + 1e-30)
        PC -= ratio**2

    return PC


def intra_inter_ratio(
    connectome: ConnectomeMatrix,
    community_labels: np.ndarray,
) -> dict[str, float]:
    """Intra- vs inter-community connectivity ratio.

    Parameters
    ----------
    connectome : ndarray, shape (R, R)
    community_labels : ndarray, shape (R,)

    Returns
    -------
    dict
        ``"intra_mean"``, ``"inter_mean"``, ``"ratio"``.
    """
    A = np.asarray(connectome, dtype=np.float64)
    labels = np.asarray(community_labels)

    same = np.equal.outer(labels, labels)
    np.fill_diagonal(same, False)

    intra_vals = A[same]
    inter_vals = A[~same & ~np.eye(len(A), dtype=bool)]

    intra_mean = float(intra_vals.mean()) if len(intra_vals) > 0 else 0.0
    inter_mean = float(inter_vals.mean()) if len(inter_vals) > 0 else 0.0
    ratio = intra_mean / (inter_mean + 1e-30)

    return {"intra_mean": intra_mean, "inter_mean": inter_mean, "ratio": ratio}


# ======================================================================
# S10  ASYMMETRY ANALYSIS
# ======================================================================


def lateralisation_index(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Lateralisation Index: LI = (L - R) / (|L| + |R|).

    Parameters
    ----------
    left, right : ndarray
        Matching descriptor values for L and R hemispheres.

    Returns
    -------
    ndarray
        LI in [-1, +1].  Positive = left > right.
    """
    L = np.asarray(left, dtype=np.float64)
    R = np.asarray(right, dtype=np.float64)
    return (L - R) / (np.abs(L) + np.abs(R) + 1e-30)


def asymmetry_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    test: Literal["paired_t", "wilcoxon"] = "wilcoxon",
) -> tuple[float, float]:
    """Test whether L and R descriptors differ significantly.

    Parameters
    ----------
    left, right : ndarray, shape (S,) -- per-subject global descriptors
    test : str

    Returns
    -------
    statistic, p_value : float
    """
    L = np.asarray(left, dtype=np.float64).ravel()
    R = np.asarray(right, dtype=np.float64).ravel()
    if L.shape != R.shape:
        raise ValueError(
            f"left and right must be paired (same length); got {L.shape} and {R.shape}."
        )

    if test == "paired_t":
        return tuple(float(x) for x in sp_stats.ttest_rel(L, R))
    elif test == "wilcoxon":
        return tuple(float(x) for x in sp_stats.wilcoxon(L, R))
    raise ValueError(f"Unknown test: {test!r}")


# ======================================================================
# S11  DIMENSIONALITY REDUCTION
# ======================================================================


def spectral_pca(
    features: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA on spectral features.

    Parameters
    ----------
    features : ndarray, shape (S, d)
    n_components : int

    Returns
    -------
    scores : ndarray, shape (S, n_components)
    loadings : ndarray, shape (n_components, d)
    explained_variance_ratio : ndarray, shape (n_components,)
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(features)
    return scores, pca.components_, pca.explained_variance_ratio_


def spectral_mds(
    distance_matrix: DistanceMatrix,
    n_components: int = 2,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Classical MDS embedding from a distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray, shape (S, S)
    n_components : int
    seed : int

    Returns
    -------
    ndarray, shape (S, n_components)
    """
    from sklearn.manifold import MDS

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
    )
    return mds.fit_transform(distance_matrix)


def spectral_umap(
    features: np.ndarray,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """UMAP embedding of spectral features.

    Parameters
    ----------
    features : ndarray, shape (S, d)
    n_components : int
    n_neighbors : int
    min_dist : float
    seed : int

    Returns
    -------
    ndarray, shape (S, n_components)
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required for UMAP.\n  pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(features)


# ======================================================================
# S12-13  EXPLORATORY DATA ANALYSIS, QC AND DESCRIPTOR RECOMMENDATION
# ======================================================================


# ======================================================================
# S1  SPECTRAL QC
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
    recommended_k: int | None = None
    warnings: list[str] = field(default_factory=list)
    passed: bool = True

    def __repr__(self) -> str:
        """Return a human-readable summary of the QC report."""
        status = "PASSED" if self.passed else "ISSUES FOUND"
        return (
            f"SpectralQC({status}, N={self.n_vertices}, "
            f"k={self.n_eigenvalues}, lambda_0={self.lambda_0:.2e}, "
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
        Tolerance for lambda_0 ~ 0.
    ortho_tol : float
        Tolerance for M-orthonormality of eigenvectors.
    row_sum_tol : float
        Tolerance for Laplacian row-sum ~ 0.
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

    # Check lambda_0 ~ 0.
    if abs(evals[0]) > lambda_0_tol:
        rpt.lambda_0_ok = False
        rpt.warnings.append(f"lambda_0 = {evals[0]:.2e} (expected ~ 0, tol={lambda_0_tol:.0e})")

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

    # Check M-orthonormality: Phi^T M Phi ~ I.
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
                f"Eigenvectors not M-orthonormal (max error={rpt.orthonormality_error:.2e})"
            )

    # Check Laplacian row sum.
    if decomp.stiffness is not None:
        row_sums = np.abs(np.asarray(decomp.stiffness.sum(axis=1)).ravel())
        rpt.laplacian_row_sum_max = float(row_sums.max())
        if rpt.laplacian_row_sum_max > row_sum_tol:
            rpt.laplacian_row_sum_ok = False
            rpt.warnings.append(
                f"Laplacian row-sum max={rpt.laplacian_row_sum_max:.2e} (should be ~ 0)"
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

    rpt.passed = rpt.lambda_0_ok and rpt.eigenvalues_nonneg and rpt.orthonormality_ok

    return rpt


# ======================================================================
# S2  OPTIMAL k SELECTION
# ======================================================================


@dataclass
class OptimalKResult:
    """Recommended number of eigenpairs by multiple criteria."""

    k_elbow: int = 0
    k_energy_95: int = 0
    k_energy_99: int = 0
    k_gap: int = 0
    k_recommended: int = 0
    eigenvalues: np.ndarray | None = None
    cumulative_energy: np.ndarray | None = None

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
    energy_thresholds: tuple[float, float] = (0.95, 0.99),
) -> OptimalKResult:
    """Determine optimal number of eigenpairs.

    Three criteria:
    1. **Elbow** -- maximum curvature of log(lambda) vs index.
    2. **Energy** -- Sigma_i lambda_i / Sigma lambda > threshold.
    3. **Max gap** -- largest relative gap between consecutive lambda.

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

    # Elbow: maximum second derivative of log(lambda).
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
# S3  DESCRIPTOR PROFILING
# ======================================================================


def descriptor_profile(
    descriptors: dict[str, np.ndarray],
    *,
    normality_samples: int = 500,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Summary statistics for each descriptor.

    Parameters
    ----------
    descriptors : dict of {name: ndarray}
        Descriptor arrays (any shape -- handles ScalarMap, DescriptorMatrix,
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
    from scipy.stats import kurtosis, shapiro, skew

    rng = np.random.default_rng(seed)
    profiles: dict[str, dict[str, Any]] = {}

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

        # Outlier count (beyond 3sigma).
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
        except ValueError as exc:  # e.g. fewer than 3 values
            warnings.warn(
                f"descriptor_profile: Shapiro-Wilk failed for '{name}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            p["shapiro_p"] = None
            p["normally_distributed"] = None

        profiles[name] = p

    return profiles


def descriptor_correlation(
    descriptors: dict[str, np.ndarray],
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> tuple[np.ndarray, list[str]]:
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

    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise ValueError(
            f"All descriptors must have the same length (got {sorted(lengths)}); "
            "truncating would pair unrelated elements."
        )
    mat = np.column_stack(vectors)

    if method == "pearson":
        corr = np.corrcoef(mat, rowvar=False)
    else:
        corr, _ = spearmanr(mat)
        if corr.ndim == 0:
            corr = np.array([[1.0]])

    return corr, names


# ======================================================================
# S4  TEST-RETEST RELIABILITY
# ======================================================================


def compute_icc(
    test: np.ndarray,
    retest: np.ndarray,
    *,
    icc_type: Literal["ICC2,1", "ICC3,1"] = "ICC3,1",
    reduce: Literal["mean", "none"] = "mean",
) -> float | np.ndarray:
    """Intraclass Correlation Coefficient for test-retest.

    Parameters
    ----------
    test : ndarray, shape (N,) or (N, T)
        Descriptor values at time 1.
    retest : ndarray, shape (N,) or (N, T)
        Descriptor values at time 2.
    icc_type : str
        ``"ICC2,1"`` -- two-way random, single measures.
        ``"ICC3,1"`` -- two-way mixed, single measures
        (recommended for neuroimaging).
    reduce : ``"mean"`` or ``"none"``
        For 2-D input the ICC is computed **per column** (subjects are the
        rows; scales are never pooled as pseudo-subjects). ``"mean"``
        returns the mean column ICC, ``"none"`` the per-column array.

    Returns
    -------
    float or ndarray
        ICC value in [-1, 1].  >0.75 = excellent, 0.60-0.75 = good,
        0.40-0.60 = fair, <0.40 = poor.
    """
    test = np.asarray(test, dtype=np.float64)
    retest = np.asarray(retest, dtype=np.float64)
    if test.shape != retest.shape:
        raise ValueError(
            f"test and retest must have the same shape; got {test.shape} and {retest.shape}."
        )
    if test.ndim == 2:
        vals = np.array([_icc_1d(test[:, j], retest[:, j], icc_type) for j in range(test.shape[1])])
        return vals if reduce == "none" else float(np.nanmean(vals))
    return _icc_1d(test.ravel(), retest.ravel(), icc_type)


def _icc_1d(test: np.ndarray, retest: np.ndarray, icc_type: str) -> float:
    """ICC for one measurement (subjects x 2 sessions)."""
    n = len(test)
    # Two-way ANOVA decomposition.
    k = 2  # two measurements
    grand_mean = (test.mean() + retest.mean()) / 2

    # Between-subjects SS.
    subject_means = (test + retest) / 2
    SS_between = k * np.sum((subject_means - grand_mean) ** 2)

    # Within-subjects SS.
    SS_within = np.sum((test - subject_means) ** 2) + np.sum((retest - subject_means) ** 2)

    # Between-measures SS.
    measure_means = np.array([test.mean(), retest.mean()])
    SS_measures = n * np.sum((measure_means - grand_mean) ** 2)

    # Error SS.
    SS_error = SS_within - SS_measures

    # Mean squares.
    MS_between = SS_between / (n - 1)
    MS_measures = SS_measures / (k - 1) if k > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-10

    if icc_type == "ICC3,1":
        # ICC(3,1) = (MS_between - MS_error) / (MS_between + (k-1)*MS_error)
        icc = (MS_between - MS_error) / (MS_between + (k - 1) * MS_error)
    elif icc_type == "ICC2,1":
        icc = (MS_between - MS_error) / (
            MS_between + (k - 1) * MS_error + k * (MS_measures - MS_error) / n
        )
    else:
        raise ValueError(f"Unknown ICC type: {icc_type!r}")

    return float(np.clip(icc, -1.0, 1.0))


def batch_effect_scan(
    descriptors: dict[str, np.ndarray],
    site_labels: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
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
    results: dict[str, dict[str, Any]] = {}

    for name, arr in descriptors.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        groups = [arr[site_labels == s] for s in unique_sites]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            warnings.warn(
                f"batch_effect_scan: '{name}' has fewer than two sites with >1 "
                "subject; the batch effect is untestable.",
                RuntimeWarning,
                stacklevel=2,
            )
            results[name] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "has_batch_effect": None,
                "effect_size_eta2": float("nan"),
                "error": "fewer than two testable sites",
            }
            continue

        try:
            stat, p_val = kruskal(*groups)
            if not np.isfinite(p_val):
                # SciPy >= 1.11 returns NaN instead of raising for constant data.
                raise ValueError("statistic undefined (all values identical)")
            # Effect size: eta^2 = H / (N - 1)
            N = sum(len(g) for g in groups)
            eta_sq = float(stat / (N - 1)) if N > 1 else 0.0
            results[name] = {
                "statistic": float(stat),
                "p_value": float(p_val),
                "has_batch_effect": p_val < alpha,
                "effect_size_eta2": eta_sq,
            }
        except ValueError as exc:  # e.g. all values identical
            warnings.warn(
                f"batch_effect_scan: Kruskal-Wallis failed for '{name}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            results[name] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "has_batch_effect": None,
                "effect_size_eta2": float("nan"),
                "error": str(exc),
            }

    return results


def eigenvalue_stability(
    decomps: list[SpectralDecomposition],
    *,
    n_eigenvalues: int | None = None,
) -> dict[str, np.ndarray]:
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
        variation), ``"eigenvalue_matrix"`` (subjects x k).
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
# S5  RECOMMEND_DESCRIPTOR -- surrogate-based selection
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
    ranking: list[dict[str, Any]]
    surrogate_details: dict[str, Any]

    def __repr__(self) -> str:
        """Return a summary showing the top-5 ranked descriptors."""
        top5 = ", ".join(f"{r['descriptor']}({r['score']:.3f})" for r in self.ranking[:5])
        return f"Recommendation('{self.recommended}' for {self.objective}) -- top 5: [{top5}]"


def _generate_surrogates(
    points: np.ndarray,
    objective: str,
    *,
    n_surrogates: int = 30,
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate synthetic deformations for descriptor evaluation.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Reference vertex coordinates (faces are kept by the caller).
    objective : str
    n_surrogates : int
    seed : int

    Returns
    -------
    surrogate_points : list of ndarray
        Deformed vertex sets (same vertex order as ``points``).
    labels : ndarray, shape (n_surrogates,)
        0 = control (undeformed), 1 = deformed.
    """
    rng = np.random.default_rng(seed)
    n_half = n_surrogates // 2
    surrogates: list[np.ndarray] = []
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
    descriptor_values: list[np.ndarray],
    labels: np.ndarray,
    *,
    n_splits: int = 5,
) -> dict[str, float]:
    """Evaluate a descriptor's discriminative power on surrogates.

    Returns AUC, balanced accuracy, and Cohen's d (largest per-feature |d|).
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

    X = np.array(features)  # (n_surr, d)
    y = labels

    # Remove zero-variance columns (a label-free filter, so no leakage).
    var_mask = X.std(axis=0) > 1e-10
    if not var_mask.any():
        return {"auc": 0.5, "auc_std": 0.0, "accuracy": 0.5, "cohens_d": 0.0}
    X = X[:, var_mask]

    n_splits_actual = min(n_splits, min(np.bincount(y)))
    n_splits_actual = max(2, n_splits_actual)

    from sklearn.pipeline import make_pipeline

    # Scaling lives inside the pipeline so it is re-fitted in every fold.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    error = None
    try:
        auc_scores = cross_val_score(clf, X, y, cv=n_splits_actual, scoring="roc_auc")
        acc_scores = cross_val_score(clf, X, y, cv=n_splits_actual, scoring="balanced_accuracy")
    except ValueError as exc:
        warnings.warn(
            f"Descriptor evaluation: cross-validation failed ({exc}); scores set to chance.",
            RuntimeWarning,
            stacklevel=2,
        )
        error = str(exc)
        auc_scores = np.array([0.5])
        acc_scores = np.array([0.5])

    # Cohen's d per feature (absolute, so opposite-signed effects do not
    # cancel), summarised by the strongest feature.
    X_scaled = StandardScaler().fit_transform(X)
    X0 = X_scaled[y == 0]
    X1 = X_scaled[y == 1]
    pooled_std = np.sqrt(
        ((len(X0) - 1) * X0.var(axis=0, ddof=1) + (len(X1) - 1) * X1.var(axis=0, ddof=1))
        / (len(X0) + len(X1) - 2)
    )
    d_per_feature = np.abs(X0.mean(axis=0) - X1.mean(axis=0)) / (pooled_std + 1e-10)
    cohens_d = float(np.max(d_per_feature))

    out = {
        "auc": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "accuracy": float(np.mean(acc_scores)),
        "cohens_d": cohens_d,
    }
    if error is not None:
        out["error"] = error
    return out


def recommend_descriptor(
    mesh: Any,
    labels: np.ndarray | None = None,
    objective: str | AnalysisObjective = "group_discrimination",
    *,
    n_surrogates: int = 30,
    k_eigenpairs: int = 30,
    n_jobs: int = 1,
    seed: int | None = 42,
) -> DescriptorRecommendation:
    """Recommend the best spectral descriptor for an analysis objective.

    Generates synthetic surrogates with controlled deformations,
    computes all eligible descriptors, evaluates each descriptor's
    discriminative power, and ranks by consensus.

    Parameters
    ----------
    mesh : BrainMesh or (vertices, faces)
        Representative triangle mesh (e.g. a template or one subject's
        surface). Surrogates displace its vertices and keep its faces, so
        every surrogate is decomposed with the same cotangent LBO used
        everywhere else in the library.
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
    This function is computationally heavy (30 surrogates x k
    eigenpairs x all descriptors by default).  For large meshes,
    consider using ``n_jobs=-1`` to parallelise the surrogate
    decomposition across CPU cores.

    Examples
    --------
    >>> rec = sb.statistics.recommend_descriptor(
    ...     mesh,
    ...     objective="group_discrimination",
    ...     n_jobs=-1,
    ... )
    >>> print(rec.recommended)
    'wks'
    >>> print(rec.ranking[:3])
    """
    from spectralbrain.spectral.lbo.descriptors import (
        compute_bates_signatures,
        compute_bks,
        compute_gps,
        compute_hks,
        compute_shapedna,
        compute_si_hks,
        compute_wks,
    )

    if labels is not None:
        warnings.warn(
            "recommend_descriptor: `labels` is reserved and currently ignored "
            "(surrogates generate their own labels).",
            UserWarning,
            stacklevel=2,
        )
    if isinstance(objective, AnalysisObjective):
        obj_str = objective.value
    else:
        obj_str = objective

    eligible = DESCRIPTOR_ELIGIBILITY.get(obj_str, [])
    if not eligible:
        raise ValueError(
            f"Unknown objective: {obj_str!r}. Available: {list(DESCRIPTOR_ELIGIBILITY.keys())}"
        )

    # Map descriptor names to compute functions.
    compute_fns: dict[str, Callable] = {
        "shapedna": lambda d: compute_shapedna(d, normalize="area"),
        "hks": lambda d: compute_hks(d, n_times=20),
        "si_hks": lambda d: compute_si_hks(d, n_frequencies=6),
        "wks": lambda d: compute_wks(d, n_energies=20),
        "gps": lambda d: compute_gps(d),
        "bates_sp": lambda d: compute_bates_signatures(d, order=2, n_times=5),
        "bks": lambda d: compute_bks(d),
    }

    # Filter to eligible + available.
    active_descs = {name: fn for name, fn in compute_fns.items() if name in eligible}

    logger.info(
        "recommend_descriptor: objective='%s', %d eligible, %d surrogates, k=%d",
        obj_str,
        len(active_descs),
        n_surrogates,
        k_eigenpairs,
    )

    # Resolve the reference mesh (vertices + faces).
    from spectralbrain.core.meshes import BrainMesh

    if isinstance(mesh, BrainMesh):
        points, faces = mesh.vertices, mesh.faces
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        points = np.asarray(mesh[0], dtype=np.float64)
        faces = np.asarray(mesh[1], dtype=np.intp)
    else:
        raise TypeError(
            "recommend_descriptor needs a triangle mesh: pass a BrainMesh or a "
            "(vertices, faces) tuple. Bare coordinate arrays are no longer "
            "accepted (SpectralBrain is mesh-only; see the 'pointsbrain' library)."
        )

    # Generate surrogates.
    surrogates, surr_labels = _generate_surrogates(
        points,
        obj_str,
        n_surrogates=n_surrogates,
        seed=seed,
    )

    # Decompose all surrogates.
    def _decompose_single(pts: np.ndarray) -> SpectralDecomposition:
        """Decompose a single surrogate mesh (displaced vertices, same faces)."""
        return BrainMesh(pts, faces).decompose(k=k_eigenpairs)

    decomps: list[SpectralDecomposition] = []
    if n_jobs == 1:
        with progress_simple("Decomposing surrogates", total=n_surrogates) as tick:
            for pts in surrogates:
                decomps.append(_decompose_single(pts))
                tick(1)
    else:
        from spectralbrain.runtime import parallel_map

        decomps = parallel_map(
            _decompose_single,
            surrogates,
            n_jobs=n_jobs,
            description="Decomposing surrogates (parallel)",
        )

    # Compute each descriptor on all surrogates and evaluate.
    scores: list[dict[str, Any]] = []

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
                scores.append(
                    {
                        "descriptor": desc_name,
                        "score": score,
                        **metrics,
                    }
                )
            except (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError) as exc:
                warnings.warn(
                    f"Descriptor '{desc_name}' failed and is ranked last: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                scores.append(
                    {
                        "descriptor": desc_name,
                        "score": 0.0,
                        "auc": 0.0,
                        "accuracy": 0.0,
                        "cohens_d": 0.0,
                        "error": str(exc),
                    }
                )
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


__all__: list[str] = [
    "ClassificationResult",
    "DescriptorRecommendation",
    "OptimalKResult",
    "SpectralQCReport",
    "VertexWiseResult",
    "asymmetry_test",
    "bag_of_spectral_words",
    "batch_effect_scan",
    "classify",
    "cohens_d_map",
    "compute_icc",
    "descriptor_correlation",
    "descriptor_profile",
    "eigenvalue_stability",
    "emd_distance",
    "energy_distance",
    "fisher_vector",
    "fit_gmm_codebook",
    "hedges_g_map",
    "intra_inter_ratio",
    "js_divergence",
    "kernel_mean_embedding",
    "kl_divergence",
    "lateralisation_index",
    "mantel_test",
    "modularity",
    "optimal_k",
    "participation_coefficient",
    "rdm",
    "recommend_descriptor",
    "rsa_compare",
    "spectral_mds",
    "spectral_pca",
    "spectral_qc",
    "spectral_umap",
    "surprise_map",
    "surprise_map_percentile",
    "tfce",
    "vertexwise_correlation",
    "vertexwise_mannwhitney",
    "vertexwise_permutation",
    "vertexwise_ttest",
]
