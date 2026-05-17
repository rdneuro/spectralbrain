"""Statistical analysis toolkit for spectral morphometry.

Covers the full analytical pipeline from vertex-wise group comparison
to connectome-level network analysis, including dimension-collapsing
methods for converting per-vertex descriptors into per-shape global
vectors.

Sections
--------
§1  Vertex-wise group comparison (t-test, Mann-Whitney, TFCE, FDR, permutation)
§2  Effect sizes (Cohen's d, Hedges' g — vertex-wise maps)
§3  Vertex-wise correlation with clinical scores
§4  Surprise / anomaly maps (z-score against normative)
§5  Classification (SVM, LogReg + CV)
§6  Dimension collapsing (Fisher vectors, Bag-of-Spectral-Words, kernel mean embedding)
§7  Dissimilarity measures (EMD, KL, JS divergence, energy distance)
§8  RSA — Representational Similarity Analysis
§9  Connectome & network analysis (modularity, participation, NBS, Mantel)
§10 Asymmetry analysis (lateralisation indices)
§11 Dimensionality reduction (PCA, MDS, UMAP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

from spectralbrain.runtime import (
    ConnectomeMatrix,
    DescriptorMatrix,
    DistanceMatrix,
    GlobalDescriptor,
    NetworkMatrix,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  VERTEX-WISE GROUP COMPARISON
# ======================================================================

@dataclass
class VertexWiseResult:
    """Results of a vertex-wise statistical test."""

    statistic: np.ndarray       # (N,) test statistic per vertex
    p_values: np.ndarray        # (N,) uncorrected p-values
    p_corrected: np.ndarray     # (N,) corrected p-values
    correction: str             # method used
    significant: np.ndarray     # (N,) bool mask at alpha
    alpha: float
    effect_size: Optional[np.ndarray] = None  # (N,) Cohen's d

    @property
    def n_significant(self) -> int:
        """Return the count of significant results after correction."""
        return int(self.significant.sum())

    def __repr__(self) -> str:
        """Return a compact summary string."""
        return (
            f"VertexWiseResult({self.n_significant} significant / "
            f"{len(self.statistic)} vertices, "
            f"correction='{self.correction}', α={self.alpha})"
        )


def vertexwise_ttest(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    correction: Literal["fdr", "bonferroni", "none"] = "fdr",
    alpha: float = 0.05,
) -> VertexWiseResult:
    """Independent two-sample t-test at each vertex.

    Parameters
    ----------
    group_a : ndarray, shape (n_a, N) or (n_a, N, T)
        Descriptor values for group A (subjects × vertices [× scales]).
        If 3D, tests are run on the mean across the last axis.
    group_b : ndarray, shape (n_b, N)
        Descriptor values for group B.
    correction : str
        ``"fdr"`` — Benjamini-Hochberg.
        ``"bonferroni"`` — Bonferroni.
        ``"none"`` — no correction.
    alpha : float

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

    N = a.shape[1]
    t_stat = np.zeros(N)
    p_vals = np.ones(N)

    for v in range(N):
        t_stat[v], p_vals[v] = sp_stats.ttest_ind(a[:, v], b[:, v])

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

    N = a.shape[1]
    u_stat = np.zeros(N)
    p_vals = np.ones(N)

    for v in range(N):
        u_stat[v], p_vals[v] = sp_stats.mannwhitneyu(
            a[:, v], b[:, v], alternative="two-sided",
        )

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
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> VertexWiseResult:
    """Permutation test at each vertex (non-parametric, exact).

    Parameters
    ----------
    group_a, group_b : ndarray, shape (n, N)
    n_permutations : int
    stat_func : str
    seed : int, optional
    alpha : float

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

    rng = np.random.default_rng(seed)
    combined = np.vstack([a, b])
    n_a, N = a.shape
    n_total = combined.shape[0]

    # Observed statistic.
    if stat_func == "t":
        obs = np.array([
            sp_stats.ttest_ind(a[:, v], b[:, v])[0] for v in range(N)
        ])
    else:
        obs = a.mean(axis=0) - b.mean(axis=0)

    # Permutation distribution.
    count_extreme = np.zeros(N, dtype=np.int64)
    with progress_simple("Permutation test", total=n_permutations) as tick:
        for _ in range(n_permutations):
            perm = rng.permutation(n_total)
            a_perm = combined[perm[:n_a]]
            b_perm = combined[perm[n_a:]]
            if stat_func == "t":
                perm_stat = np.array([
                    sp_stats.ttest_ind(a_perm[:, v], b_perm[:, v])[0]
                    for v in range(N)
                ])
            else:
                perm_stat = a_perm.mean(axis=0) - b_perm.mean(axis=0)
            count_extreme += (np.abs(perm_stat) >= np.abs(obs)).astype(np.int64)
            tick(1)

    p_vals = (count_extreme + 1) / (n_permutations + 1)

    return VertexWiseResult(
        statistic=obs,
        p_values=p_vals,
        p_corrected=p_vals,  # permutation p-values are already corrected
        correction="permutation",
        significant=p_vals < alpha,
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

    TFCE(v) = ∫₀^h(v) e(h)^E · h^H dh

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
        TFCE-enhanced statistic map.

    References
    ----------
    Smith SM, Nichols TE. Threshold-free cluster enhancement.
    *NeuroImage* 44(1):83–98, 2009.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    adj = sp.csr_matrix(adjacency)
    stat = np.abs(statistic_map)
    max_stat = stat.max()
    if max_stat < 1e-10:
        return np.zeros_like(stat)

    thresholds = np.linspace(0, max_stat, n_steps + 1)[1:]
    dh = thresholds[1] - thresholds[0] if len(thresholds) > 1 else max_stat
    tfce_map = np.zeros_like(stat, dtype=np.float64)

    for h in thresholds:
        # Supra-threshold mask.
        mask = stat >= h
        if not mask.any():
            continue

        # Find connected components in supra-threshold subgraph.
        sub_adj = adj[mask][:, mask]
        n_comp, comp_labels = connected_components(sub_adj, directed=False)

        # Cluster extent for each vertex.
        for c in range(n_comp):
            c_mask = comp_labels == c
            extent = c_mask.sum()
            # Add contribution: e^E · h^H · dh.
            vertices_in_cluster = np.where(mask)[0][c_mask]
            tfce_map[vertices_in_cluster] += (extent ** E) * (h ** H) * dh

    return tfce_map


def _correct_pvalues(
    p_values: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply multiple comparison correction."""
    if method == "none":
        return p_values.copy()
    elif method == "bonferroni":
        return np.minimum(p_values * len(p_values), 1.0)
    elif method == "fdr":
        return _fdr_bh(p_values)
    raise ValueError(f"Unknown correction: {method!r}")


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    adjusted = ranked_p * n / (np.arange(1, n + 1))
    # Enforce monotonicity.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(p_values)
    result[order] = np.minimum(adjusted, 1.0)
    return result


# ======================================================================
# §2  EFFECT SIZES
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
# §3  VERTEX-WISE CORRELATION
# ======================================================================

def vertexwise_correlation(
    descriptors: np.ndarray,
    scores: np.ndarray,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
    correction: str = "fdr",
    alpha: float = 0.05,
    covariates: Optional[np.ndarray] = None,
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
    scores = np.asarray(scores, dtype=np.float64)
    S, N = desc.shape

    r_vals = np.zeros(N)
    p_vals = np.ones(N)

    for v in range(N):
        x = desc[:, v]
        y = scores

        if covariates is not None:
            # Partial correlation: residualise both x and y.
            cov = np.asarray(covariates, dtype=np.float64)
            x = _residualise(x, cov)
            y = _residualise(y, cov)

        if method == "pearson":
            r_vals[v], p_vals[v] = sp_stats.pearsonr(x, y)
        else:
            r_vals[v], p_vals[v] = sp_stats.spearmanr(x, y)

    p_corr = _correct_pvalues(p_vals, method=correction)

    return VertexWiseResult(
        statistic=r_vals,
        p_values=p_vals,
        p_corrected=p_corr,
        correction=correction,
        significant=p_corr < alpha,
        alpha=alpha,
    )


def _residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals: y - X @ (X^+ @ y)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


# ======================================================================
# §4  SURPRISE / ANOMALY MAPS
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
        Percentile rank (0–100) of subject relative to normative.
    """
    subj = np.asarray(subject_descriptor)
    norm = np.asarray(normative_distribution)
    N = subj.shape[0]
    pctile = np.zeros(N)
    for v in range(N):
        pctile[v] = sp_stats.percentileofscore(norm[:, v], subj[v])
    return pctile


# ======================================================================
# §5  CLASSIFICATION
# ======================================================================

@dataclass
class ClassificationResult:
    """Output of a classification analysis."""
    accuracy: float
    accuracy_std: float
    auc: float
    auc_std: float
    feature_importance: Optional[np.ndarray]
    confusion_matrix: Optional[np.ndarray]
    model_name: str

    def __repr__(self) -> str:
        """Return a compact summary string."""
        return (
            f"Classification({self.model_name}: "
            f"AUC={self.auc:.3f}±{self.auc_std:.3f}, "
            f"Acc={self.accuracy:.3f}±{self.accuracy_std:.3f})"
        )


def classify(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model: Literal["svm", "logistic", "random_forest"] = "svm",
    n_folds: int = 5,
    seed: Optional[int] = 42,
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

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
# §6  DIMENSION COLLAPSING — from per-vertex to per-shape
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
    ndarray, shape (2·K·T,)
        Fisher vector (concatenation of first and second order
        gradient statistics).

    References
    ----------
    Perronnin F, Dance C. Fisher kernels on visual vocabularies for
    image categorization. *CVPR 2007*.
    Sánchez J, Perronnin F, Mensink T, Verbeek J. Image classification
    with the Fisher vector. *IJCV* 105(3):222–245, 2013.
    """
    N, T = descriptor.shape
    K = gmm_means.shape[0]

    # Responsibilities: γ(n, k) = P(k|x_n)
    log_resp = np.zeros((N, K))
    for k in range(K):
        diff = descriptor - gmm_means[k]
        log_resp[:, k] = (
            np.log(gmm_weights[k] + 1e-30)
            - 0.5 * np.sum(diff ** 2 / (gmm_covs[k] + 1e-30), axis=1)
            - 0.5 * np.sum(np.log(gmm_covs[k] + 1e-30))
        )
    # Normalise responsibilities.
    log_resp -= log_resp.max(axis=1, keepdims=True)
    resp = np.exp(log_resp)
    resp /= resp.sum(axis=1, keepdims=True)

    fv_parts = []
    for k in range(K):
        gamma_k = resp[:, k]                                # (N,)
        sqrt_w = np.sqrt(gmm_weights[k] + 1e-30)
        diff = descriptor - gmm_means[k]                    # (N, T)
        sigma = gmm_covs[k]                                  # (T,)

        # First-order gradient (mean).
        g_mean = (1 / (N * sqrt_w)) * np.sum(
            gamma_k[:, None] * diff / (np.sqrt(sigma) + 1e-30), axis=0
        )
        # Second-order gradient (variance).
        g_var = (1 / (N * sqrt_w * np.sqrt(2))) * np.sum(
            gamma_k[:, None] * (diff ** 2 / (sigma + 1e-30) - 1), axis=0
        )
        fv_parts.extend([g_mean, g_var])

    fv = np.concatenate(fv_parts)

    # L2 normalisation + power normalisation.
    fv = np.sign(fv) * np.sqrt(np.abs(fv))                  # power norm
    norm = np.linalg.norm(fv)
    if norm > 1e-10:
        fv /= norm

    return fv


def fit_gmm_codebook(
    all_descriptors: np.ndarray,
    n_components: int = 32,
    *,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    sigma: Optional[float] = None,
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
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
        weights /= weights.sum(axis=1, keepdims=True)
        histogram = weights.sum(axis=0)
    else:
        assignments = np.argmin(dists, axis=1)
        histogram = np.bincount(assignments, minlength=codebook.shape[0]).astype(
            np.float64
        )

    # L1 normalisation.
    histogram /= histogram.sum() + 1e-30
    return histogram


def kernel_mean_embedding(
    descriptor: DescriptorMatrix,
    *,
    kernel: Literal["rbf", "linear"] = "rbf",
    sigma: Optional[float] = None,
    n_landmarks: int = 100,
    seed: Optional[int] = None,
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
# §7  DISSIMILARITY MEASURES
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
    return 0.5 * kl_divergence(a, b, **kwargs) + 0.5 * kl_divergence(b, a, **kwargs)


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
# §8  RSA — Representational Similarity Analysis
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
        S items × d features.
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
    seed: Optional[int] = None,
) -> Tuple[float, float]:
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
    method: Literal["pearson", "spearman"] = "spearman",
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Mantel test — correlation between two distance matrices.

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
        matrix_a, matrix_b,
        method=method,
        permutations=n_permutations,
        seed=seed,
    )


# ======================================================================
# §9  CONNECTOME & NETWORK ANALYSIS
# ======================================================================

def modularity(
    connectome: ConnectomeMatrix,
    community_labels: np.ndarray,
    *,
    gamma: float = 1.0,
) -> float:
    """Newman's modularity Q for a given community partition.

    Q = (1/2m) Σ_{ij} [A_{ij} - γ·k_i·k_j/(2m)] · δ(c_i, c_j)

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
    A = np.asarray(connectome, dtype=np.float64)
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

    PC_i = 1 - Σ_k (s_{ik} / s_i)²

    High PC → hub connected to multiple communities.
    Low PC → provincial node within one community.

    Parameters
    ----------
    connectome : ndarray, shape (R, R)
    community_labels : ndarray, shape (R,)

    Returns
    -------
    ndarray, shape (R,)
    """
    A = np.asarray(connectome, dtype=np.float64)
    np.fill_diagonal(A, 0)
    labels = np.asarray(community_labels)
    communities = np.unique(labels)

    s_total = A.sum(axis=1)                                 # (R,)
    PC = np.ones(len(A))

    for c in communities:
        mask = labels == c
        s_c = A[:, mask].sum(axis=1)                        # (R,)
        ratio = s_c / (s_total + 1e-30)
        PC -= ratio ** 2

    return PC


def intra_inter_ratio(
    connectome: ConnectomeMatrix,
    community_labels: np.ndarray,
) -> Dict[str, float]:
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
# §10  ASYMMETRY ANALYSIS
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
        LI ∈ [-1, +1].  Positive = left > right.
    """
    L = np.asarray(left, dtype=np.float64)
    R = np.asarray(right, dtype=np.float64)
    return (L - R) / (np.abs(L) + np.abs(R) + 1e-30)


def asymmetry_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    test: Literal["paired_t", "wilcoxon"] = "wilcoxon",
) -> Tuple[float, float]:
    """Test whether L and R descriptors differ significantly.

    Parameters
    ----------
    left, right : ndarray, shape (S,) — per-subject global descriptors
    test : str

    Returns
    -------
    statistic, p_value : float
    """
    L = np.asarray(left).ravel()
    R = np.asarray(right).ravel()
    n = min(len(L), len(R))
    L, R = L[:n], R[:n]

    if test == "paired_t":
        return tuple(float(x) for x in sp_stats.ttest_rel(L, R))
    elif test == "wilcoxon":
        return tuple(float(x) for x in sp_stats.wilcoxon(L, R))
    raise ValueError(f"Unknown test: {test!r}")


# ======================================================================
# §11  DIMENSIONALITY REDUCTION
# ======================================================================

def spectral_pca(
    features: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    seed: Optional[int] = None,
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
    seed: Optional[int] = None,
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
        raise ImportError(
            "umap-learn is required for UMAP.\n"
            "  pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(features)


# ======================================================================

__all__: List[str] = [
    # Vertex-wise tests
    "VertexWiseResult",
    "vertexwise_ttest",
    "vertexwise_mannwhitney",
    "vertexwise_permutation",
    "tfce",
    # Effect sizes
    "cohens_d_map",
    "hedges_g_map",
    # Correlation
    "vertexwise_correlation",
    # Surprise maps
    "surprise_map",
    "surprise_map_percentile",
    # Classification
    "ClassificationResult",
    "classify",
    # Dimension collapsing
    "fisher_vector",
    "fit_gmm_codebook",
    "bag_of_spectral_words",
    "kernel_mean_embedding",
    # Dissimilarity
    "emd_distance",
    "kl_divergence",
    "js_divergence",
    "energy_distance",
    # RSA
    "rdm",
    "rsa_compare",
    "mantel_test",
    # Connectome
    "modularity",
    "participation_coefficient",
    "intra_inter_ratio",
    # Asymmetry
    "lateralisation_index",
    "asymmetry_test",
    # Dimensionality reduction
    "spectral_pca",
    "spectral_mds",
    "spectral_umap",
]
