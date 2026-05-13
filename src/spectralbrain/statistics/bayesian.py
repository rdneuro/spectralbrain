"""Bayesian statistical models for spectral morphometry.

Six models with a scikit-learn-like API: ``.fit()``, ``.predict()``,
``.score()``, ``.summary()``.  All models delegate MCMC sampling
to the backends (PyMC NUTS, nutpie, or NumPyro).

Models
------
1. **HorseshoeRegression** — sparse regression for feature selection.
2. **BayesianGroupComparison** — BEST (Kruschke 2013) with HDI + ROPE.
3. **HierarchicalLinearModel** — multi-site random effects.
4. **GaussianProcessNormative** — GP age-trajectory normative.
5. **BayesianSpatialModel** — GMRF vertex-wise spatial prior.
6. **BayesianConnectome** — hierarchical connectome comparison.

Examples
--------
>>> model = HorseshoeRegression()
>>> model.fit(descriptors, clinical_scores)
>>> model.summary()
>>> predictions = model.predict(new_descriptors)
>>> model.score()  # LOO-CV

Dependencies
------------
PyMC, ArviZ (optional, lazy-imported).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from spectralbrain.runtime import (
    ConnectomeMatrix,
    PathLike,
    get_logger,
)

logger = get_logger(__name__)


# ======================================================================
# Lazy imports
# ======================================================================

def _require_pymc():
    try:
        import pymc as pm
        return pm
    except ImportError as exc:
        raise ImportError(
            "PyMC is required for Bayesian models.\n"
            "  pip install pymc"
        ) from exc


def _require_arviz():
    try:
        import arviz as az
        return az
    except ImportError as exc:
        raise ImportError(
            "ArviZ is required for Bayesian diagnostics.\n"
            "  pip install arviz"
        ) from exc


# ======================================================================
# §0  BASE CLASS
# ======================================================================

class BayesianModel(abc.ABC):
    """Abstract base for all SpectralBrain Bayesian models.

    Subclasses implement :meth:`_build_model` to construct a PyMC
    model, and optionally override :meth:`_build_posterior_predictive`
    for custom prediction logic.

    Attributes
    ----------
    trace_ : arviz.InferenceData or None
        Posterior samples (populated after ``.fit()``).
    model_ : pymc.Model or None
        The PyMC model object.
    """

    def __init__(self) -> None:
        self.trace_: Any = None
        self.model_: Any = None
        self._is_fitted: bool = False

    @abc.abstractmethod
    def _build_model(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any:
        """Construct and return a PyMC model."""
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sampler: Literal["auto", "nuts", "nutpie", "numpyro"] = "auto",
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        target_accept: float = 0.95,
        seed: Optional[int] = 42,
        **kwargs: Any,
    ) -> "BayesianModel":
        """Fit the model via MCMC sampling.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Feature matrix.
        y : ndarray, shape (n,)
            Target variable.
        sampler : str
            ``"auto"`` tries nutpie → numpyro → nuts.
        draws, tune, chains, cores : int
            MCMC configuration.
        target_accept : float
        seed : int
        **kwargs
            Extra arguments passed to the sampler.

        Returns
        -------
        self
        """
        pm = _require_pymc()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.model_ = self._build_model(X, y, **kwargs)

        with self.model_:
            if sampler in ("auto", "nuts"):
                try:
                    # Try nutpie first.
                    import nutpie
                    compiled = nutpie.compile_pymc_model(self.model_)
                    self.trace_ = nutpie.sample(
                        compiled, draws=draws, tune=tune,
                        chains=chains, seed=seed,
                    )
                    logger.info("Fitted with nutpie (%d draws × %d chains).", draws, chains)
                    self._is_fitted = True
                    return self
                except (ImportError, Exception):
                    pass

                # Fallback to PyMC native.
                self.trace_ = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    cores=cores, target_accept=target_accept,
                    random_seed=seed, return_inferencedata=True,
                    progressbar=True, **kwargs,
                )
                logger.info("Fitted with PyMC NUTS (%d draws × %d chains).", draws, chains)

            elif sampler == "nutpie":
                import nutpie
                compiled = nutpie.compile_pymc_model(self.model_)
                self.trace_ = nutpie.sample(
                    compiled, draws=draws, tune=tune,
                    chains=chains, seed=seed,
                )
                logger.info("Fitted with nutpie (%d draws × %d chains).", draws, chains)

            elif sampler == "numpyro":
                import pymc.sampling.jax as pmjax
                self.trace_ = pmjax.sample_numpyro_nuts(
                    draws=draws, tune=tune, chains=chains,
                    target_accept=target_accept,
                    random_seed=seed,
                    progress_bar=True, **kwargs,
                )
                logger.info("Fitted with NumPyro (%d draws × %d chains).", draws, chains)

        self._is_fitted = True
        return self

    def predict(
        self,
        X_new: np.ndarray,
        *,
        n_samples: int = 500,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate posterior predictive samples.

        Parameters
        ----------
        X_new : ndarray, shape (m, d)
            New feature values.
        n_samples : int
            Number of posterior predictive draws.
        seed : int

        Returns
        -------
        ndarray, shape (n_samples, m)
            Posterior predictive samples.
        """
        self._check_fitted()
        pm = _require_pymc()

        with self.model_:
            pm.set_data({"X": X_new})
            ppc = pm.sample_posterior_predictive(
                self.trace_, random_seed=seed,
                predictions=True,
            )

        # Extract prediction array.
        pred_vars = list(ppc.predictions.data_vars)
        return ppc.predictions[pred_vars[0]].values.reshape(-1, X_new.shape[0])

    def score(
        self,
        method: Literal["loo", "waic"] = "loo",
    ) -> float:
        """Model comparison score via LOO-CV or WAIC.

        Parameters
        ----------
        method : str
            ``"loo"`` — Leave-One-Out via PSIS.
            ``"waic"`` — Widely Applicable Information Criterion.

        Returns
        -------
        float
            Expected log pointwise predictive density (elpd).
        """
        self._check_fitted()
        az = _require_arviz()

        if method == "loo":
            result = az.loo(self.trace_, pointwise=False)
            return float(result.elpd_loo)
        elif method == "waic":
            result = az.waic(self.trace_, pointwise=False)
            return float(result.elpd_waic)
        raise ValueError(f"Unknown method: {method!r}")

    def summary(
        self,
        var_names: Optional[List[str]] = None,
        hdi_prob: float = 0.94,
    ) -> Any:
        """ArviZ summary table.

        Parameters
        ----------
        var_names : list of str, optional
        hdi_prob : float

        Returns
        -------
        pandas.DataFrame
        """
        self._check_fitted()
        az = _require_arviz()
        return az.summary(self.trace_, var_names=var_names, hdi_prob=hdi_prob)

    def save(self, path: PathLike) -> Path:
        """Save trace to NetCDF."""
        self._check_fitted()
        az = _require_arviz()
        out = Path(path)
        self.trace_.to_netcdf(str(out))
        logger.info("Trace saved → %s", out)
        return out

    @classmethod
    def load_trace(cls, path: PathLike) -> Any:
        """Load a saved trace."""
        az = _require_arviz()
        return az.from_netcdf(str(path))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted yet. Call .fit(X, y) first."
            )


# ======================================================================
# §1  HORSESHOE REGRESSION
# ======================================================================

class HorseshoeRegression(BayesianModel):
    """Sparse Bayesian regression with horseshoe prior.

    The horseshoe prior (Carvalho, Polson & Scott 2009) provides
    aggressive shrinkage of irrelevant coefficients toward zero
    while leaving large effects unshrunk — ideal for selecting
    which of 20+ spectral descriptors predict a clinical outcome.

    Parameters
    ----------
    tau_prior : float
        Global shrinkage scale (smaller = more sparse).
        Rule of thumb: p_eff / (d - p_eff) / sqrt(n), where
        p_eff is expected number of relevant features.

    Examples
    --------
    >>> model = HorseshoeRegression(tau_prior=0.1)
    >>> model.fit(descriptors, clinical_scores)
    >>> model.summary()
    >>> model.predict(new_descriptors)
    """

    def __init__(self, tau_prior: float = 0.1) -> None:
        super().__init__()
        self.tau_prior = tau_prior

    def _build_model(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> Any:
        pm = _require_pymc()
        n, d = X.shape

        with pm.Model() as model:
            X_data = pm.Data("X", X)
            y_data = pm.Data("y_obs", y)

            # Global shrinkage.
            tau = pm.HalfCauchy("tau", beta=self.tau_prior)

            # Local shrinkage per coefficient.
            lam = pm.HalfCauchy("lambda", beta=1.0, shape=d)

            # Coefficients.
            beta = pm.Normal("beta", mu=0, sigma=tau * lam, shape=d)

            # Intercept.
            alpha = pm.Normal("alpha", mu=y.mean(), sigma=y.std() * 2)

            # Noise.
            sigma = pm.HalfNormal("sigma", sigma=y.std())

            # Likelihood.
            mu = alpha + pm.math.dot(X_data, beta)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

        return model

    def feature_importance(self) -> np.ndarray:
        """Posterior mean of |β| — higher = more important.

        Returns
        -------
        ndarray, shape (d,)
        """
        self._check_fitted()
        az = _require_arviz()
        beta = self.trace_.posterior["beta"].values
        return np.abs(beta).mean(axis=(0, 1))


# ======================================================================
# §2  BAYESIAN GROUP COMPARISON (BEST — Kruschke 2013)
# ======================================================================

class BayesianGroupComparison(BayesianModel):
    """Bayesian Estimation Supersedes the T-test (BEST).

    Estimates the full posterior distribution of group means,
    standard deviations, effect size, and their differences.
    Reports HDI (Highest Density Interval) and probability of
    the difference exceeding a ROPE.

    Parameters
    ----------
    rope : tuple of float
        Region Of Practical Equivalence.  Default (-0.1, 0.1)
        in units of the pooled standard deviation.

    Examples
    --------
    >>> model = BayesianGroupComparison(rope=(-0.1, 0.1))
    >>> model.fit(group_a_values, group_b_values)
    >>> model.summary()
    >>> model.effect_size_posterior()
    """

    def __init__(self, rope: Tuple[float, float] = (-0.1, 0.1)) -> None:
        super().__init__()
        self.rope = rope

    def fit(self, group_a: np.ndarray, group_b: np.ndarray, **kwargs) -> "BayesianGroupComparison":
        """Fit BEST model.

        Parameters
        ----------
        group_a, group_b : ndarray, shape (n_a,) and (n_b,)
        """
        a = np.asarray(group_a, dtype=np.float64).ravel()
        b = np.asarray(group_b, dtype=np.float64).ravel()
        # Store for predict.
        self._a = a
        self._b = b
        # Pack into X, y format for parent .fit().
        X = np.zeros((len(a) + len(b), 1))
        y = np.concatenate([a, b])
        return super().fit(X, y, **kwargs)

    def _build_model(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> Any:
        pm = _require_pymc()
        a, b = self._a, self._b
        pooled = np.concatenate([a, b])

        with pm.Model() as model:
            # Group means.
            mu_a = pm.Normal("mu_a", mu=pooled.mean(), sigma=pooled.std() * 2)
            mu_b = pm.Normal("mu_b", mu=pooled.mean(), sigma=pooled.std() * 2)

            # Group standard deviations.
            sigma_a = pm.HalfNormal("sigma_a", sigma=pooled.std() * 2)
            sigma_b = pm.HalfNormal("sigma_b", sigma=pooled.std() * 2)

            # Normality parameter (Student-t df).
            nu = pm.Exponential("nu_minus1", lam=1 / 29.0) + 1

            # Likelihoods.
            pm.StudentT("obs_a", nu=nu, mu=mu_a, sigma=sigma_a, observed=a)
            pm.StudentT("obs_b", nu=nu, mu=mu_b, sigma=sigma_b, observed=b)

            # Derived quantities.
            pm.Deterministic("diff_means", mu_a - mu_b)
            pm.Deterministic("diff_stds", sigma_a - sigma_b)
            pooled_sd = pm.math.sqrt(
                (sigma_a ** 2 + sigma_b ** 2) / 2
            )
            pm.Deterministic("effect_size", (mu_a - mu_b) / pooled_sd)

        return model

    def predict(self, X_new=None, **kwargs):
        raise NotImplementedError(
            "BayesianGroupComparison does not support predict(). "
            "Use .effect_size_posterior() or .summary() instead."
        )

    def effect_size_posterior(self) -> np.ndarray:
        """Posterior samples of Cohen's d.

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        self._check_fitted()
        return self.trace_.posterior["effect_size"].values.ravel()

    def rope_probability(self) -> Dict[str, float]:
        """Probability of effect size in, below, and above ROPE.

        Returns
        -------
        dict
            Keys: ``"p_rope"`` (inside), ``"p_below"`` (below),
            ``"p_above"`` (above ROPE).
        """
        d = self.effect_size_posterior()
        lo, hi = self.rope
        return {
            "p_below": float((d < lo).mean()),
            "p_rope": float(((d >= lo) & (d <= hi)).mean()),
            "p_above": float((d > hi).mean()),
        }


# ======================================================================
# §3  HIERARCHICAL LINEAR MODEL
# ======================================================================

class HierarchicalLinearModel(BayesianModel):
    """Multi-site hierarchical linear model with random effects.

    Models spectral descriptors with fixed effects (group, age, sex)
    and random intercepts/slopes per site, handling batch effects
    within the model rather than post-hoc harmonisation.

    y ~ α + β·X + u_site + ε

    Parameters
    ----------
    random_effects : str
        ``"intercept"`` — random intercept per site.
        ``"slope"`` — random intercept + slope per site.

    Examples
    --------
    >>> model = HierarchicalLinearModel(random_effects="intercept")
    >>> model.fit(X, y, site_labels=sites)
    """

    def __init__(
        self,
        random_effects: Literal["intercept", "slope"] = "intercept",
    ) -> None:
        super().__init__()
        self.random_effects = random_effects

    def fit(self, X, y, *, site_labels: np.ndarray, **kwargs):
        """Fit with site labels for random effects."""
        self._site_labels = np.asarray(site_labels)
        self._unique_sites = np.unique(self._site_labels)
        self._site_idx = np.searchsorted(
            self._unique_sites, self._site_labels,
        )
        return super().fit(X, y, **kwargs)

    def _build_model(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> Any:
        pm = _require_pymc()
        n, d = X.shape
        n_sites = len(self._unique_sites)

        with pm.Model() as model:
            X_data = pm.Data("X", X)
            y_data = pm.Data("y_obs", y)
            site_idx = pm.Data("site_idx", self._site_idx)

            # Fixed effects.
            alpha = pm.Normal("alpha", mu=y.mean(), sigma=y.std() * 2)
            beta = pm.Normal("beta", mu=0, sigma=1, shape=d)

            # Random effects by site.
            sigma_site = pm.HalfNormal("sigma_site", sigma=y.std())
            u_site = pm.Normal("u_site", mu=0, sigma=sigma_site, shape=n_sites)

            # Linear predictor.
            mu = alpha + pm.math.dot(X_data, beta) + u_site[site_idx]

            if self.random_effects == "slope" and d > 0:
                sigma_slope = pm.HalfNormal("sigma_slope", sigma=1.0)
                beta_site = pm.Normal(
                    "beta_site", mu=0, sigma=sigma_slope,
                    shape=(n_sites, d),
                )
                mu = mu + (X_data * beta_site[site_idx]).sum(axis=1)

            # Noise.
            sigma = pm.HalfNormal("sigma", sigma=y.std())
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

        return model

    def site_effects(self) -> np.ndarray:
        """Posterior mean of random intercepts per site.

        Returns
        -------
        ndarray, shape (n_sites,)
        """
        self._check_fitted()
        return self.trace_.posterior["u_site"].values.mean(axis=(0, 1))


# ======================================================================
# §4  GAUSSIAN PROCESS NORMATIVE
# ======================================================================

class GaussianProcessNormative(BayesianModel):
    """GP-based normative model for age trajectories.

    Fits a Gaussian Process over age (or any continuous covariate)
    to model the normative distribution of a spectral descriptor.
    Individual deviations are computed as z-scores from the posterior
    predictive.

    Parameters
    ----------
    kernel : str
        ``"matern32"`` or ``"matern52"`` or ``"rbf"``.
    lengthscale_prior : float
        Prior mean for the GP lengthscale (in years for age).

    Examples
    --------
    >>> gp = GaussianProcessNormative(kernel="matern52")
    >>> gp.fit(ages_controls[:, None], descriptor_controls)
    >>> z_patient = gp.deviation(age_patient, descriptor_patient)
    """

    def __init__(
        self,
        kernel: Literal["matern32", "matern52", "rbf"] = "matern52",
        lengthscale_prior: float = 10.0,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.lengthscale_prior = lengthscale_prior
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def _build_model(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> Any:
        pm = _require_pymc()
        import pymc.gp as gp

        self._X_train = X.copy()
        self._y_train = y.copy()

        with pm.Model() as model:
            # GP hyperpriors.
            ls = pm.InverseGamma(
                "lengthscale",
                alpha=5,
                beta=5 * self.lengthscale_prior,
            )
            eta = pm.HalfNormal("eta", sigma=y.std())
            sigma = pm.HalfNormal("sigma", sigma=y.std() * 0.5)

            # Kernel.
            if self.kernel == "matern32":
                cov = eta ** 2 * gp.cov.Matern32(input_dim=X.shape[1], ls=ls)
            elif self.kernel == "matern52":
                cov = eta ** 2 * gp.cov.Matern52(input_dim=X.shape[1], ls=ls)
            elif self.kernel == "rbf":
                cov = eta ** 2 * gp.cov.ExpQuad(input_dim=X.shape[1], ls=ls)
            else:
                raise ValueError(f"Unknown kernel: {self.kernel!r}")

            # Marginal GP.
            self._gp = gp.Marginal(cov_func=cov)
            self._gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)

        return model

    def predict(self, X_new: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior predictive mean and std at new points.

        Parameters
        ----------
        X_new : ndarray, shape (m, d)

        Returns
        -------
        mean : ndarray, shape (m,)
        std : ndarray, shape (m,)
        """
        self._check_fitted()
        pm = _require_pymc()

        with self.model_:
            pred = self._gp.conditional("f_pred", X_new)
            ppc = pm.sample_posterior_predictive(
                self.trace_, var_names=["f_pred"],
                random_seed=kwargs.get("seed"),
            )

        samples = ppc.posterior_predictive["f_pred"].values
        samples = samples.reshape(-1, X_new.shape[0])
        return samples.mean(axis=0), samples.std(axis=0)

    def deviation(
        self,
        age: float,
        observed_value: float,
    ) -> float:
        """Z-score deviation of an individual from the normative.

        Parameters
        ----------
        age : float
        observed_value : float

        Returns
        -------
        float
            Z-score (positive = above normative).
        """
        X_new = np.array([[age]])
        mean, std = self.predict(X_new)
        return float((observed_value - mean[0]) / (std[0] + 1e-30))


# ======================================================================
# §5  BAYESIAN SPATIAL MODEL
# ======================================================================

class BayesianSpatialModel(BayesianModel):
    """Vertex-wise Bayesian model with spatial GMRF prior.

    Places a Gaussian Markov Random Field prior on the vertex-level
    effects, so neighbouring vertices share information.  This is
    Bayesian spatial smoothing — more principled than Gaussian
    kernel pre-smoothing.

    Parameters
    ----------
    spatial_strength : float
        Precision of the GMRF prior (higher = more spatial
        smoothing).

    Examples
    --------
    >>> model = BayesianSpatialModel(spatial_strength=10.0)
    >>> model.fit(group_labels, vertex_descriptors,
    ...           adjacency=mesh_adjacency)
    """

    def __init__(self, spatial_strength: float = 10.0) -> None:
        super().__init__()
        self.spatial_strength = spatial_strength

    def fit(self, group_labels, vertex_data, *, adjacency, **kwargs):
        """Fit spatial model.

        Parameters
        ----------
        group_labels : ndarray, shape (S,)
            Group assignment (0/1) per subject.
        vertex_data : ndarray, shape (S, N)
            Per-subject vertex-wise descriptor values.
        adjacency : sparse matrix, shape (N, N)
            Mesh or kNN adjacency.
        """
        self._adjacency = adjacency
        self._group_labels = np.asarray(group_labels)
        self._vertex_data = np.asarray(vertex_data, dtype=np.float64)

        # Build as X (group) → y (mean vertex descriptor)
        X = group_labels.reshape(-1, 1).astype(np.float64)
        y = vertex_data.mean(axis=1)  # collapse vertices for base .fit()
        return super().fit(X, y, **kwargs)

    def _build_model(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> Any:
        pm = _require_pymc()
        import scipy.sparse as sp

        N = self._vertex_data.shape[1]
        S = len(self._group_labels)

        # Build GMRF precision from adjacency (graph Laplacian + diagonal).
        adj = sp.csr_matrix(self._adjacency)
        degree = np.asarray(adj.sum(axis=1)).ravel()
        Q = sp.diags(degree) - adj + sp.eye(N) * self.spatial_strength

        with pm.Model() as model:
            # Global intercept and group effect.
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta_group = pm.Normal("beta_group", mu=0, sigma=5)

            # Vertex-level group effect with spatial prior.
            # Simplified: model group difference per vertex as
            # spatially smooth via CAR-like prior.
            sigma_spatial = pm.HalfNormal("sigma_spatial", sigma=1)
            tau_spatial = 1 / (sigma_spatial ** 2)

            # Vertex-level effects (simplified as Normal with spatial std).
            vertex_effect = pm.Normal(
                "vertex_effect", mu=0,
                sigma=sigma_spatial, shape=N,
            )

            # Spatial penalty as potential (soft GMRF).
            Q_dense = Q.toarray()
            spatial_penalty = -0.5 * tau_spatial * pm.math.dot(
                vertex_effect, pm.math.dot(Q_dense, vertex_effect)
            )
            pm.Potential("spatial_prior", spatial_penalty)

            # Likelihood: per-subject, per-vertex.
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=self._vertex_data.std())
            group_float = self._group_labels.astype(np.float64)

            for s in range(S):
                mu_s = alpha + beta_group * group_float[s] + vertex_effect
                pm.Normal(
                    f"y_{s}", mu=mu_s, sigma=sigma_obs,
                    observed=self._vertex_data[s],
                )

        return model

    def vertex_effect_map(self) -> np.ndarray:
        """Posterior mean of vertex-level group effect.

        Returns
        -------
        ndarray, shape (N,)
        """
        self._check_fitted()
        return self.trace_.posterior["vertex_effect"].values.mean(axis=(0, 1))


# ======================================================================
# §6  BAYESIAN CONNECTOME COMPARISON
# ======================================================================

class BayesianConnectome(BayesianModel):
    """Hierarchical Bayesian model for connectome comparison.

    Models each entry of the geometric connectome matrix with a
    hierarchical prior, testing whether connection strengths differ
    between groups while sharing information across edges.

    Parameters
    ----------
    shrinkage : float
        Hierarchical shrinkage strength.

    Examples
    --------
    >>> model = BayesianConnectome()
    >>> model.fit(connectomes_patients, connectomes_controls)
    >>> diff = model.edge_difference_posterior()
    """

    def __init__(self, shrinkage: float = 1.0) -> None:
        super().__init__()
        self.shrinkage = shrinkage

    def fit(
        self,
        group_a_connectomes: np.ndarray,
        group_b_connectomes: np.ndarray,
        **kwargs,
    ) -> "BayesianConnectome":
        """Fit connectome comparison model.

        Parameters
        ----------
        group_a_connectomes : ndarray, shape (n_a, R, R)
        group_b_connectomes : ndarray, shape (n_b, R, R)
        """
        a = np.asarray(group_a_connectomes, dtype=np.float64)
        b = np.asarray(group_b_connectomes, dtype=np.float64)
        self._conn_a = a
        self._conn_b = b
        self._R = a.shape[1]

        # Extract upper triangle for modeling.
        triu_idx = np.triu_indices(self._R, k=1)
        self._triu_idx = triu_idx
        n_edges = len(triu_idx[0])

        # Stack into X (group indicator), y (edge values).
        a_edges = np.array([c[triu_idx] for c in a])       # (n_a, n_edges)
        b_edges = np.array([c[triu_idx] for c in b])       # (n_b, n_edges)
        self._a_edges = a_edges
        self._b_edges = b_edges

        X = np.zeros((len(a) + len(b), 1))
        y = np.concatenate([a_edges.mean(axis=1), b_edges.mean(axis=1)])
        return super().fit(X, y, **kwargs)

    def _build_model(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> Any:
        pm = _require_pymc()
        n_edges = self._a_edges.shape[1]
        a_mean = self._a_edges.mean(axis=0)
        b_mean = self._b_edges.mean(axis=0)
        all_edges = np.concatenate([self._a_edges, self._b_edges])

        with pm.Model() as model:
            # Hierarchical prior on edge-level differences.
            mu_diff = pm.Normal("mu_diff", mu=0, sigma=self.shrinkage)
            sigma_diff = pm.HalfNormal("sigma_diff", sigma=self.shrinkage)

            # Per-edge difference.
            edge_diff = pm.Normal(
                "edge_diff", mu=mu_diff, sigma=sigma_diff,
                shape=n_edges,
            )

            # Group means.
            grand_mean = pm.Normal(
                "grand_mean", mu=all_edges.mean(), sigma=all_edges.std(),
                shape=n_edges,
            )

            sigma_obs = pm.HalfNormal("sigma_obs", sigma=all_edges.std())

            # Likelihoods.
            pm.Normal(
                "obs_a", mu=grand_mean + edge_diff / 2,
                sigma=sigma_obs, observed=a_mean,
            )
            pm.Normal(
                "obs_b", mu=grand_mean - edge_diff / 2,
                sigma=sigma_obs, observed=b_mean,
            )

        return model

    def predict(self, X_new=None, **kwargs):
        raise NotImplementedError(
            "BayesianConnectome does not support predict(). "
            "Use .edge_difference_posterior() instead."
        )

    def edge_difference_posterior(self) -> np.ndarray:
        """Posterior mean of per-edge group difference.

        Returns
        -------
        ndarray, shape (n_edges,)
        """
        self._check_fitted()
        return self.trace_.posterior["edge_diff"].values.mean(axis=(0, 1))

    def edge_difference_matrix(self) -> ConnectomeMatrix:
        """Reconstruct the difference as a symmetric matrix.

        Returns
        -------
        ndarray, shape (R, R)
        """
        diff = self.edge_difference_posterior()
        mat = np.zeros((self._R, self._R))
        mat[self._triu_idx] = diff
        mat += mat.T
        return mat


# ======================================================================

__all__: List[str] = [
    "BayesianModel",
    "HorseshoeRegression",
    "BayesianGroupComparison",
    "HierarchicalLinearModel",
    "GaussianProcessNormative",
    "BayesianSpatialModel",
    "BayesianConnectome",
]
