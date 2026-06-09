"""Regression tests for the statistics audit corrections.

These lock in the fixes from the frequentist/Bayesian audit:
FWER control in vertex-wise permutation, partial-correlation degrees of
freedom, Welch vs Student t, vectorisation equivalence, the analytic
DeLong test, the non-inferiority paired/unpaired branches, and
reproducible parallel bootstrap.
"""

import numpy as np
import pytest
from scipy import stats as sp

import spectralbrain.statistics.analysis as A
import spectralbrain.statistics.normative as Nm
import spectralbrain.statistics.surrogates as Su


# ----------------------------------------------------------------------
# Vertex-wise tests: vectorisation equivalence + Welch default
# ----------------------------------------------------------------------
def test_ttest_vectorized_matches_loop_welch_and_student():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (20, 40))
    b = rng.normal(0.3, 1.5, (25, 40))

    welch = A.vertexwise_ttest(a, b, correction="none")  # default Welch
    ref_w = np.array([sp.ttest_ind(a[:, v], b[:, v], equal_var=False)[0] for v in range(40)])
    assert np.allclose(welch.statistic, ref_w)

    student = A.vertexwise_ttest(a, b, correction="none", equal_var=True)
    ref_s = np.array([sp.ttest_ind(a[:, v], b[:, v])[0] for v in range(40)])
    assert np.allclose(student.statistic, ref_s)


def test_mannwhitney_vectorized_matches_loop():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (15, 30))
    b = rng.normal(0.5, 1, (18, 30))
    res = A.vertexwise_mannwhitney(a, b, correction="none")
    ref = np.array(
        [sp.mannwhitneyu(a[:, v], b[:, v], alternative="two-sided")[1] for v in range(30)]
    )
    assert np.allclose(res.p_values, ref)


# ----------------------------------------------------------------------
# Permutation FWER (max-statistic) — the headline correctness fix
# ----------------------------------------------------------------------
def test_permutation_max_dominates_pervertex():
    """FWER max-stat p-value is >= the per-vertex p at every vertex.

    This is the deterministic signature of a real max-statistic
    correction (the old code falsely returned the per-vertex p as
    'corrected').
    """
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (16, 50))
    b = rng.normal(0, 1, (16, 50))
    b[:, :8] += 1.5
    res = A.vertexwise_permutation(a, b, n_permutations=500, correction="max", seed=0)
    assert np.all(res.p_corrected >= res.p_values - 1e-12)


def test_permutation_max_detects_effect_controls_null():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (18, 40))
    b = rng.normal(0, 1, (18, 40))
    b[:, :8] += 2.0
    res = A.vertexwise_permutation(a, b, n_permutations=1000, correction="max", seed=0)
    assert res.significant[:8].sum() >= 5  # detects signal
    assert res.significant[8:].sum() == 0  # no false positives in null region


# ----------------------------------------------------------------------
# Partial correlation: degrees of freedom + confound removal
# ----------------------------------------------------------------------
def test_partial_correlation_removes_confound():
    rng = np.random.default_rng(4)
    n = 40
    conf = rng.normal(0, 1, n)
    score = 2 * conf + rng.normal(0, 0.5, n)
    desc = 2 * conf[:, None] + rng.normal(0, 0.5, (n, 6))

    spurious = A.vertexwise_correlation(desc, score, correction="none")
    controlled = A.vertexwise_correlation(desc, score, covariates=conf, correction="none")
    assert spurious.statistic.mean() > 0.8  # strong spurious correlation
    assert abs(controlled.statistic.mean()) < 0.3  # removed after control


def test_partial_correlation_uses_adjusted_df():
    rng = np.random.default_rng(5)
    n = 30
    cov = rng.normal(0, 1, (n, 2))
    score = rng.normal(0, 1, n)
    desc = rng.normal(0, 1, (n, 4))
    res = A.vertexwise_correlation(desc, score, covariates=cov, correction="none")
    r0 = res.statistic[0]
    df = n - 2 - 2  # S - 2 - C
    t = r0 * np.sqrt(df / (1 - r0**2))
    p_manual = 2 * sp.t.sf(abs(t), df)
    assert np.isclose(res.p_values[0], p_manual)


# ----------------------------------------------------------------------
# DeLong AUC comparison (analytic)
# ----------------------------------------------------------------------
def test_delong_auc_matches_sklearn_and_is_deterministic():
    roc = pytest.importorskip("sklearn.metrics").roc_auc_score
    rng = np.random.default_rng(6)
    n = 150
    y = (rng.random(n) < 0.4).astype(int)
    s1 = y + rng.normal(0, 1.0, n)
    s2 = y + rng.normal(0, 2.0, n)
    auc1, auc2, p = Nm.auc_comparison_delong(y, s1, s2)
    assert np.isclose(auc1, roc(y, s1))
    assert np.isclose(auc2, roc(y, s2))
    assert Nm.auc_comparison_delong(y, s1, s2) == (auc1, auc2, p)  # deterministic


def test_delong_identical_models_p_one_and_symmetric():
    rng = np.random.default_rng(7)
    y = (rng.random(120) < 0.5).astype(int)
    s = y + rng.normal(0, 1, 120)
    _, _, p_id = Nm.auc_comparison_delong(y, s, s.copy())
    assert np.isclose(p_id, 1.0)
    # Swapping the two models gives the same p-value.
    s2 = y + rng.normal(0, 1.5, 120)
    _, _, p_ab = Nm.auc_comparison_delong(y, s, s2)
    _, _, p_ba = Nm.auc_comparison_delong(y, s2, s)
    assert np.isclose(p_ab, p_ba)


# ----------------------------------------------------------------------
# Non-inferiority: paired and unpaired branches both work and differ
# ----------------------------------------------------------------------
def test_non_inferiority_paired_and_unpaired():
    rng = np.random.default_rng(8)
    new = rng.normal(0.80, 0.05, 40)
    ref = rng.normal(0.79, 0.05, 40)
    paired = Nm.non_inferiority_test(new, ref, margin=0.05, paired=True)
    unpaired = Nm.non_inferiority_test(new, ref, margin=0.05, paired=False)
    # Both reach a decision; SE/df differ between schemes so p-values differ.
    assert paired.p_value != unpaired.p_value
    assert isinstance(unpaired.is_non_inferior, (bool, np.bool_))


# ----------------------------------------------------------------------
# Bootstrap: parallel reproducibility + BCa validity
# ----------------------------------------------------------------------
def test_bootstrap_njobs_invariant():
    data = np.random.default_rng(9).normal(5, 2, 60)
    _, l1, h1 = Su.bootstrap_ci(data, np.mean, n_bootstrap=1500, seed=3, n_jobs=1)
    _, l2, h2 = Su.bootstrap_ci(data, np.mean, n_bootstrap=1500, seed=3, n_jobs=2)
    assert np.isclose(l1, l2) and np.isclose(h1, h2)


def test_bootstrap_bca_brackets_estimate():
    data = np.random.default_rng(10).normal(5, 2, 80)
    est, lo, hi = Su.bootstrap_ci(data, np.mean, n_bootstrap=2000, seed=1, method="bca")
    assert np.isclose(est, data.mean())
    assert lo < est < hi


def test_compute_icc_recovers_high_reliability():
    """ICC ≈ 1 when retest closely tracks test, ≈ 0 when independent."""
    import spectralbrain.statistics.eda as E

    rng = np.random.default_rng(11)
    subj = rng.normal(0, 1, 50)
    high = E.compute_icc(subj + rng.normal(0, 0.05, 50), subj + rng.normal(0, 0.05, 50))
    low = E.compute_icc(rng.normal(0, 1, 50), rng.normal(0, 1, 50))
    assert high > 0.9
    assert low < 0.4


# ----------------------------------------------------------------------
# Bayesian fit dispatch (audit: no silent error swallowing)
# ----------------------------------------------------------------------
def test_bayesian_invalid_sampler_raises():
    pytest.importorskip("pymc")
    from spectralbrain.statistics.bayesian import HorseshoeRegression

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 3))
    y = X[:, 0] + rng.normal(0, 0.3, 30)
    with pytest.raises(ValueError, match="Unknown sampler"):
        HorseshoeRegression().fit(X, y, sampler="not_a_sampler", draws=50, tune=50, chains=1)


def test_horseshoe_recovers_sparse_signal():
    pytest.importorskip("pymc")
    from spectralbrain.statistics.bayesian import HorseshoeRegression

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 6))
    beta = np.array([2.0, 0, 0, -1.5, 0, 0])
    y = X @ beta + rng.normal(0, 0.3, 80)
    m = HorseshoeRegression(tau_prior=0.5)
    m.fit(X, y, sampler="nuts", draws=300, tune=300, chains=2, cores=1, seed=0)
    top2 = set(np.argsort(m.feature_importance())[::-1][:2])
    assert top2 == {0, 3}  # the two truly nonzero coefficients


def test_best_rope_probabilities_sum_to_one():
    pytest.importorskip("pymc")
    from spectralbrain.statistics.bayesian import BayesianGroupComparison

    rng = np.random.default_rng(1)
    a = rng.normal(0.6, 1, 30)
    b = rng.normal(0, 1, 30)
    gc = BayesianGroupComparison(rope=(-0.1, 0.1)).fit(
        a, b, sampler="nuts", draws=300, tune=300, chains=2, cores=1, seed=0
    )
    rope = gc.rope_probability()
    assert np.isclose(sum(rope.values()), 1.0)
    assert rope["p_above"] > rope["p_below"]  # group a shifted up
