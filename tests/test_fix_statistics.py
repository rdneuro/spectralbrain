"""Regression tests for the statistics bug-fix pass.

Each test locks in one confirmed bug from the audit of
``spectralbrain.statistics`` (TFCE sign merging, eigenstrapping copies,
non-degree-preserving rewiring, caller-array mutation, NaN poisoning, ...).
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import spectralbrain.statistics.analysis as A
import spectralbrain.statistics.analysis as E
import spectralbrain.statistics.clustering.methods as C
import spectralbrain.statistics.normative as Nm
import spectralbrain.statistics.surrogates as Su
from spectralbrain.statistics.clustering._core import consensus, nulls


def _path_adjacency(n: int) -> sp.csr_matrix:
    return sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")


def _path_laplacian(n: int) -> sp.csr_matrix:
    A = _path_adjacency(n)
    return (sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A).tocsr()


# ----------------------------------------------------------------------
# Surrogates / nulls
# ----------------------------------------------------------------------
def _orthobasis(V=200, K=30, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(V, K)))
    return Q


def test_eigenstrapping_surrogates_are_not_copies():
    Q = _orthobasis()
    evals = np.linspace(0, 10, Q.shape[1])  # no degeneracy at all
    data = np.random.default_rng(1).normal(size=Q.shape[0])
    surr = nulls.eigenstrapping_surrogates(
        data, evals, Q, np.ones(Q.shape[0]), n_surrogates=5, residual="none"
    )
    assert np.abs(surr - surr[0]).max() > 1e-3
    # Power within each Koussis block is preserved; mode 0 is fixed.
    c = Q.T @ data
    for s in surr:
        cs = Q.T @ s
        assert np.isclose(cs[0], c[0])
        for g in nulls._harmonic_groups(Q.shape[1]):
            assert np.isclose((cs[g] ** 2).sum(), (c[g] ** 2).sum())


def test_eigenstrapping_residual_permuted_keeps_variance_scale():
    Q = _orthobasis()
    data = np.random.default_rng(2).normal(size=Q.shape[0])
    surr = Su.null_eigenstrapping(data, np.linspace(0, 10, 30), Q, np.ones(200), n_surrogates=3)
    assert np.isclose(np.sum(surr[0] ** 2), np.sum(data**2), rtol=0.5)


def test_eigenstrapping_degenerate_grouping_warns():
    Q = _orthobasis()
    with pytest.warns(RuntimeWarning, match="No near-degenerate"):
        nulls.eigenstrapping_surrogates(
            np.ones(200),
            np.linspace(0, 10, 30),
            Q,
            np.ones(200),
            n_surrogates=2,
            grouping="degenerate",
        )


def test_edge_rewiring_preserves_degree_without_loops_or_multiedges():
    rng = np.random.default_rng(0)
    Cm = (rng.random((20, 20)) < 0.3).astype(float)
    Cm = np.triu(Cm, 1)
    Cm = Cm + Cm.T
    for M in Su.null_edge_rewiring(Cm, n_surrogates=3, seed=0):
        assert np.array_equal((M > 0).sum(0), (Cm > 0).sum(0))
        assert np.allclose(np.diag(M), 0)
        assert M.max() == 1.0
        assert not np.array_equal(M, Cm)


def test_phase_randomisation_preserves_power_and_mean():
    Q = _orthobasis(V=50, K=50)
    Q[:, 0] = 1 / np.sqrt(50)
    Q, _ = np.linalg.qr(Q)
    x = np.random.default_rng(3).normal(size=50) + 4
    for s in Su.null_phase_randomisation(x, Q, n_surrogates=4, seed=1):
        assert np.isclose(np.sum(s**2), np.sum(x**2))
        assert np.isclose(s.mean(), x.mean())


def test_paired_label_permutation_add_one():
    diff = np.full(10, 5.0)
    _, p, _ = nulls.paired_label_permutation(diff, np.zeros(10), n_perm=99)
    assert p >= 1 / 100


def test_brainsmash_seed_kwarg():
    pytest.importorskip("brainsmash")
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(60, 3))
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    out = Su.null_brainsmash(rng.normal(size=60), D, n_surrogates=2, seed=1)
    assert len(out) == 2


# ----------------------------------------------------------------------
# analysis.py
# ----------------------------------------------------------------------
def test_tfce_keeps_tails_separate():
    adj = _path_adjacency(3)
    both = A.tfce(np.array([3.0, -3.0, 0.0]), adj)
    single = A.tfce(np.array([3.0, 0.0, 0.0]), adj)
    assert both[0] > 0 and both[1] < 0
    assert np.isclose(both[0], single[0])
    assert np.isclose(both[1], -single[0])


def test_network_metrics_do_not_mutate_input():
    M = np.ones((4, 4))
    A.modularity(M, np.array([0, 0, 1, 1]))
    A.participation_coefficient(M, np.array([0, 0, 1, 1]))
    assert np.allclose(np.diag(M), 1.0)


def test_vertexwise_correlation_nan_robust():
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    X = y[:, None] * np.ones((40, 5)) + 0.1 * rng.normal(size=(40, 5))
    y[3] = np.nan
    X[5, 2] = np.nan
    with pytest.warns(RuntimeWarning):
        res = A.vertexwise_correlation(X, y)
    assert np.all(res.statistic > 0.9)
    assert res.significant.all()


def test_perfect_correlation_is_significant():
    y = np.arange(10.0)
    res = A.vertexwise_correlation(np.column_stack([y, y]), y, correction="none")
    assert np.all(res.p_values < 1e-6)


def test_fdr_ignores_nan():
    out = A._fdr_bh(np.array([0.001, 0.01, np.nan, 0.02]))
    assert np.isnan(out[2])
    assert np.allclose(out[[0, 1, 3]], [0.003, 0.015, 0.02])


def test_js_divergence_is_bounded():
    rng = np.random.default_rng(0)
    js = A.js_divergence(rng.normal(0, 1, 1000), rng.normal(10, 1, 1000))
    assert 0 <= js <= np.log(2) + 1e-12
    assert A.js_divergence(np.arange(100.0), np.arange(100.0)) == pytest.approx(0.0)


def test_asymmetry_test_rejects_unpaired():
    with pytest.raises(ValueError):
        A.asymmetry_test(np.ones(5), np.ones(4))


# ----------------------------------------------------------------------
# normative.py
# ----------------------------------------------------------------------
def _site_data(seed=0, n=30, F=6):
    rng = np.random.default_rng(seed)
    ages = rng.uniform(20, 80, 2 * n)
    sex = rng.integers(0, 2, 2 * n).astype(float)
    sites = np.array(["A"] * n + ["B"] * n)
    data = 0.02 * ages[:, None] + rng.normal(size=(2 * n, F))
    data[n:] = data[n:] * 1.5 + 2.0
    return data, ages, sex, sites


def test_combat_apply_reproduces_training_harmonisation():
    data, ages, sex, sites = _site_data()
    cov = np.column_stack([ages, sex])
    res = Nm.harmonize_combat(data, sites, covariates=cov)
    again = Nm.combat_apply(data, sites, res.estimates, covariates=cov)
    assert np.allclose(again, res.data_harmonized)


def test_combat_mean_only_and_nonparametric_are_honoured():
    data, _, _, sites = _site_data()
    mo = Nm.harmonize_combat(data, sites, mean_only=True)
    assert np.allclose(mo.estimates["delta_star"], 1.0)
    npar = Nm.harmonize_combat(data, sites, parametric=False)
    par = Nm.harmonize_combat(data, sites, parametric=True)
    assert not np.allclose(npar.estimates["gamma_star"], npar.estimates["gamma_hat"])
    assert not np.allclose(npar.data_harmonized, par.data_harmonized)


def test_normative_scoring_applies_harmonisation():
    data, ages, sex, sites = _site_data()
    nm = Nm.NormativeModel("gaussian").fit(
        data, ages=ages, sex=sex, sites=sites, harmonize_method="combat"
    )
    with pytest.raises(ValueError, match="site"):
        nm.score(data[40], age=ages[40], sex=sex[40])
    z_raw_space = Nm.NormativeModel("gaussian").fit(data, ages=ages, sex=sex)
    z_h = nm.score(data[40], age=ages[40], sex=sex[40], site="B")
    # A site-B subject scored through the harmonised model has the same
    # z-score as its harmonised training row.
    harm = nm._reference_data[40]
    expected = (harm - nm._intercept - nm._age_coef * ages[40] - nm._sex_coef * sex[40]) / (
        nm._residual_std
    )
    assert np.allclose(z_h, expected)
    assert z_raw_space is not None
    zb = nm.score_batch(data[:3], ages=ages[:3], sex=sex[:3], sites=sites[:3])
    assert zb.shape == (3, data.shape[1])


def test_normative_requires_sex_when_fitted_with_sex():
    data, ages, sex, _ = _site_data()
    nm = Nm.NormativeModel("gaussian").fit(data, ages=ages, sex=sex)
    with pytest.raises(ValueError, match="sex"):
        nm.score(data[0], age=40.0)


def test_centile_scoring_is_age_matched():
    rng = np.random.default_rng(0)
    ages = np.linspace(20, 80, 200)
    data = (ages / 10)[:, None] + 0.1 * rng.normal(size=(200, 1))
    nm = Nm.NormativeModel("centile", centile_neighbours=30).fit(data, ages=ages)
    # 3.0 is the median for 30-year-olds but extreme for 70-year-olds.
    assert nm.score(np.array([3.0]), age=30.0)[0] == pytest.approx(50, abs=20)
    assert nm.score(np.array([3.0]), age=70.0)[0] < 5


def test_centile_curves_include_oldest_subject():
    ages = np.array([20.0, 30, 40, 50, 60])
    d = np.array([1.0, 2, 3, 4, 100])
    out = Nm.centile_curves(d, ages, percentiles=(100,), n_age_bins=2, smooth=False)
    assert out["centiles"][100][-1] == 100


def test_paired_tests_reject_length_mismatch_and_tost_ci():
    with pytest.raises(ValueError):
        Nm.non_inferiority_test(np.ones(5), np.ones(4))
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=30), rng.normal(size=30)
    r = Nm.equivalence_test_tost(a, b, margin=0.5, alpha=0.05)
    from scipy import stats

    diff = a - b
    se = diff.std(ddof=1) / np.sqrt(30)
    assert np.isclose(r.ci_upper - r.difference, stats.t.ppf(0.95, 29) * se)


def test_nadeau_bengio_correction_widens_ci():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.8, 0.02, 10), rng.normal(0.79, 0.02, 10)
    plain = Nm.non_inferiority_test(a, b)
    corr = Nm.non_inferiority_test(a, b, cv_test_train_ratio=1 / 9)
    assert (corr.ci_upper - corr.ci_lower) > (plain.ci_upper - plain.ci_lower)


# ----------------------------------------------------------------------
# analysis.py -- EDA / QC section
# ----------------------------------------------------------------------
def test_icc_is_per_column():
    rng = np.random.default_rng(0)
    # Columns on wildly different scales but pure noise between sessions.
    scale = np.array([1, 10, 100, 1000.0])
    t = rng.normal(size=(30, 4)) * scale
    r = rng.normal(size=(30, 4)) * scale
    assert E.compute_icc(t, r) < 0.5
    per = E.compute_icc(t, r, reduce="none")
    assert per.shape == (4,)


def test_evaluate_descriptor_cohens_d_does_not_cancel():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 10 + [1] * 10)
    vals = []
    for lab in labels:
        base = rng.normal(size=(50, 2)) * 0.01
        base[:, 0] += 1.0 * lab
        base[:, 1] -= 1.0 * lab
        vals.append(base)
    out = E._evaluate_descriptor(vals, labels)
    assert out["cohens_d"] > 2


def test_batch_effect_scan_flags_untestable():
    with pytest.warns(RuntimeWarning):
        out = E.batch_effect_scan({"x": np.ones(6)}, np.array([0, 0, 0, 1, 1, 1]))
    assert out["x"]["has_batch_effect"] is None


# ----------------------------------------------------------------------
# clustering.py
# ----------------------------------------------------------------------
def test_denoise_identity_filter_is_identity():
    n = 40
    H = np.ones((n, 8)) * 5
    out = C.denoise_joint_timevertex(
        H, _path_laplacian(n), alpha_graph=0, beta_time=0, n_eigenvectors=10
    )
    assert np.allclose(out, 5.0)


def test_joint_spectral_uses_data():
    n = 60
    rng = np.random.default_rng(0)
    L = _path_laplacian(n)
    r1 = C.cluster_joint_spectral(rng.random((n, 5)), L, n_clusters=3, n_eigenvectors=20)
    r2 = C.cluster_joint_spectral(rng.random((n, 5)) ** 4, L, n_clusters=3, n_eigenvectors=20)
    assert not np.allclose(r1.metadata["spectral_energy"], r2.metadata["spectral_energy"])


def test_graph_laplacian_helper():
    A_ = _path_adjacency(5)
    L = C._as_graph_laplacian(A_)
    assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0)
    L2 = C._as_graph_laplacian(_path_laplacian(5))
    assert np.allclose(L2.toarray(), _path_laplacian(5).toarray())


def test_coclustering_smoothing_uses_laplacian():
    pytest.importorskip("sklearn")
    n = 40
    rng = np.random.default_rng(0)
    H = np.vstack(
        [
            rng.random((20, 6)) + np.array([5, 5, 5, 0, 0, 0]),
            rng.random((20, 6)) + np.array([0, 0, 0, 5, 5, 5]),
        ]
    )
    r = C.cluster_spectral_coclustering(H, n_clusters=2, adjacency=_path_adjacency(n))
    # Smoothing on a path keeps two contiguous blocks.
    assert len(np.unique(r.labels[:20])) == 1 and len(np.unique(r.labels[20:])) == 1


def test_spatiotemporal_gnmf_stays_finite_nonnegative():
    rng = np.random.default_rng(0)
    H = rng.random((30, 10))
    r = C.cluster_spatiotemporal_gnmf(
        H, _path_adjacency(30), n_components=3, lam_temporal=50.0, n_iter=50, backend="cpu"
    )
    for k in ("W", "F"):
        assert np.all(np.isfinite(r.metadata[k])) and np.all(r.metadata[k] >= 0)


def test_fuse_concatenate_does_not_mutate():
    hks = np.random.default_rng(0).random((10, 4))
    wks = np.random.default_rng(1).random((10, 3))
    h0, w0 = hks.copy(), wks.copy()
    C.fuse_concatenate(hks, wks, log_transform=False)
    assert np.array_equal(hks, h0) and np.array_equal(wks, w0)


def test_persistence_fallback_clusters_maxima():
    # Two bumps on a path: clusters are the basins of the maxima of H.
    x = np.linspace(0, 2 * np.pi, 80)
    f = np.sin(x) ** 2
    try:
        import gudhi  # noqa: F401
    except ImportError:
        r = C.cluster_persistence(f, _path_adjacency(80), n_clusters=2)
        assert r.n_clusters == 2
        assert r.labels[20] != r.labels[60]


def test_vineyards_record_representative_vertex():
    rng = np.random.default_rng(0)
    H = rng.random((50, 4))
    r = C.cluster_vineyards(H, _path_adjacency(50), min_persistence_frac=0.0, min_life_frac=0.0)
    assert r.salient_features
    assert all(f["representative_vertex"] >= 0 for f in r.salient_features)


def test_dpmm_mrf_not_silently_ignored():
    with pytest.raises(NotImplementedError):
        C.cluster_dpmm(
            np.random.default_rng(0).random((30, 4)),
            adjacency=_path_adjacency(30),
            mrf_beta=1.0,
            dim_reduction=None,
        )


# ----------------------------------------------------------------------
# clustering._core
# ----------------------------------------------------------------------
def test_mantel_kendall_null_matches_statistic(monkeypatch):
    # The permutation null must use the same statistic as the observed value.
    calls = {"kendall": 0, "spearman": 0, "pearson": 0}
    orig = {
        k: getattr(A.sp_stats, f)
        for k, f in [("kendall", "kendalltau"), ("spearman", "spearmanr"), ("pearson", "pearsonr")]
    }
    for k, f in [("kendall", "kendalltau"), ("spearman", "spearmanr"), ("pearson", "pearsonr")]:

        def spy(*a, _k=k, **kw):
            calls[_k] += 1
            return orig[_k](*a, **kw)

        monkeypatch.setattr(A.sp_stats, f, spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    r, p = A.mantel_test(D, D, n_permutations=20, method="kendall", seed=0)
    assert calls == {"kendall": 21, "spearman": 0, "pearson": 0}
    assert 0 < p <= 1 and r == pytest.approx(1.0)


def test_co_association_ignores_noise():
    co = consensus.co_association_matrix([np.array([-1, -1, 0, 0])])
    assert co[0, 1] == 0 and co[2, 3] == 1 and co[0, 0] == 1
    stab = consensus.stability_per_vertex(co, np.array([-1, -1, 0, 0]))
    assert np.isnan(stab[0]) and stab[2] == 1


# ----------------------------------------------------------------------
# Bayesian (PyMC optional)
# ----------------------------------------------------------------------
def test_bayesian_connectome_subject_level_likelihood():
    pytest.importorskip("pymc")
    from spectralbrain.statistics.bayesian import BayesianConnectome

    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 4, 4))
    b = rng.normal(size=(5, 4, 4))
    m = BayesianConnectome()
    m._conn_a, m._conn_b = a, b
    iu = np.triu_indices(4, 1)
    m._a_edges = np.array([c[iu] for c in a])
    m._b_edges = np.array([c[iu] for c in b])
    model = m._build_model(np.zeros((11, 1)), np.zeros(11))
    assert model["obs_a"].eval().shape == (6, 6)


def test_check_sampling_warns_on_divergences():
    pytest.importorskip("arviz")
    import arviz as az

    idata = az.from_dict(
        posterior={"x": np.random.default_rng(0).normal(size=(2, 50))},
        sample_stats={"diverging": np.r_[np.ones(3), np.zeros(97)].reshape(2, 50).astype(bool)},
    )
    from spectralbrain.statistics.bayesian import check_sampling

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(RuntimeWarning, match="divergent"):
            out = check_sampling(idata)
    assert out["n_divergences"] == 3
