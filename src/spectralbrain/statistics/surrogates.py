"""Surrogate data, null models, and synthetic data generators.

Three pillars:

1. **Bootstrap** — resampling with CI (percentile + BCa).
2. **Null models** — 6 hypothesis-specific null generators.
3. **Synthetic generators** — descriptors, meshes, point clouds
   from real data or 22 parametric distributions.

Null models
-----------
1. Eigenvalue permutation — tests spectral ordering.
2. Phase randomisation — tests vertex-specific structure.
3. Spin permutation — tests beyond spatial autocorrelation.
4. Subject permutation — tests group difference.
5. Edge rewiring — tests network topology.
6. Parametric — tests beyond marginal distribution.

Distributions
-------------
normal, beta, gamma, cauchy, dirichlet, exponential, halfcauchy,
halfnormal, poisson, inversegamma, laplace, kumaraswamy, studentt,
negativebinomial, binomial, logistic, mixture, pareto, uniform,
wald, vonmises, weibull.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Any,
    Literal,
)

import numpy as np
from scipy import stats as sp_stats

from spectralbrain.runtime import (
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)

DistributionName = Literal[
    "normal",
    "beta",
    "gamma",
    "cauchy",
    "dirichlet",
    "exponential",
    "halfcauchy",
    "halfnormal",
    "poisson",
    "inversegamma",
    "laplace",
    "kumaraswamy",
    "studentt",
    "negativebinomial",
    "binomial",
    "logistic",
    "mixture",
    "pareto",
    "uniform",
    "wald",
    "vonmises",
    "weibull",
]


# ==== DISTRIBUTION SAMPLER ====


def _sample_distribution(
    distribution: DistributionName,
    size: tuple[int, ...],
    *,
    rng: np.random.Generator,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw samples from one of 22 parametric distributions.

    Parameters
    ----------
    distribution : str
        Distribution name (see :data:`DistributionName`).
    size : tuple of int
        Output shape.
    rng : numpy.random.Generator
        Random number generator instance.
    params : dict, optional
        Distribution-specific parameters.

    Returns
    -------
    ndarray
        Samples with shape *size*.

    Raises
    ------
    ValueError
        If *distribution* is not recognised.
    """
    p = params or {}
    if distribution == "normal":
        return rng.normal(p.get("loc", 0), p.get("scale", 1), size)
    elif distribution == "beta":
        return rng.beta(p.get("a", 2), p.get("b", 5), size)
    elif distribution == "gamma":
        return rng.gamma(p.get("shape", 2), p.get("scale", 1), size)
    elif distribution == "cauchy":
        return sp_stats.cauchy.rvs(
            loc=p.get("loc", 0), scale=p.get("scale", 1), size=size, random_state=rng
        )
    elif distribution == "dirichlet":
        alpha = p.get("alpha", np.ones(size[-1]) if len(size) > 1 else np.ones(3))
        n = int(np.prod(size[:-1])) if len(size) > 1 else size[0]
        return rng.dirichlet(alpha, n).reshape(size) if len(size) > 1 else rng.dirichlet(alpha, n)
    elif distribution == "exponential":
        return rng.exponential(p.get("scale", 1), size)
    elif distribution == "halfcauchy":
        return np.abs(
            sp_stats.cauchy.rvs(loc=0, scale=p.get("scale", 1), size=size, random_state=rng)
        )
    elif distribution == "halfnormal":
        return np.abs(rng.normal(0, p.get("scale", 1), size))
    elif distribution == "poisson":
        return rng.poisson(p.get("lam", 5), size).astype(np.float64)
    elif distribution == "inversegamma":
        return sp_stats.invgamma.rvs(p.get("a", 3), size=size, random_state=rng)
    elif distribution == "laplace":
        return rng.laplace(p.get("loc", 0), p.get("scale", 1), size)
    elif distribution == "kumaraswamy":
        a, b = p.get("a", 2), p.get("b", 5)
        u = rng.uniform(0, 1, size)
        return (1 - (1 - u) ** (1 / b)) ** (1 / a)
    elif distribution == "studentt":
        return sp_stats.t.rvs(
            p.get("df", 5),
            loc=p.get("loc", 0),
            scale=p.get("scale", 1),
            size=size,
            random_state=rng,
        )
    elif distribution == "negativebinomial":
        return rng.negative_binomial(p.get("n", 5), p.get("p", 0.5), size).astype(np.float64)
    elif distribution == "binomial":
        return rng.binomial(p.get("n", 10), p.get("p", 0.5), size).astype(np.float64)
    elif distribution == "logistic":
        return rng.logistic(p.get("loc", 0), p.get("scale", 1), size)
    elif distribution == "mixture":
        k = p.get("n_components", 2)
        weights = p.get("weights", np.ones(k) / k)
        locs = p.get("locs", np.linspace(-2, 2, k))
        scales = p.get("scales", np.ones(k))
        comp = rng.choice(k, size=size, p=weights)
        return np.array([rng.normal(locs[c], scales[c]) for c in comp.ravel()]).reshape(size)
    elif distribution == "pareto":
        return (rng.pareto(p.get("a", 3), size) + 1) * p.get("scale", 1)
    elif distribution == "uniform":
        return rng.uniform(p.get("low", 0), p.get("high", 1), size)
    elif distribution == "wald":
        return rng.wald(p.get("mean", 1), p.get("scale", 1), size)
    elif distribution == "vonmises":
        return rng.vonmises(p.get("mu", 0), p.get("kappa", 2), size)
    elif distribution == "weibull":
        return rng.weibull(p.get("a", 2), size) * p.get("scale", 1)
    raise ValueError(f"Unknown distribution: {distribution!r}")


def _match_distribution(
    reference: np.ndarray, size: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Generate samples matching an empirical reference distribution.

    Uses bootstrap resampling with added jitter (5% of reference std)
    to produce new samples that follow the same marginal distribution
    as *reference*.

    Parameters
    ----------
    reference : ndarray
        Empirical data to match.
    size : tuple of int
        Output shape.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray
        Samples with shape *size*.
    """
    flat = reference.ravel()
    idx = rng.choice(len(flat), size=int(np.prod(size)), replace=True)
    noise_scale = flat.std() * 0.05
    return (flat[idx] + rng.normal(0, noise_scale, int(np.prod(size)))).reshape(size)


# ==== §1 BOOTSTRAP ====


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    method: Literal["percentile", "bca"] = "percentile",
    seed: int | None = None,
    n_jobs: int = 1,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a scalar statistic.

    Parameters
    ----------
    data : ndarray
        Input data array.
    statistic : callable
        Function that maps ``data`` to a scalar.
    n_bootstrap : int
        Number of bootstrap replicates.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    method : ``"percentile"`` or ``"bca"``
        CI method.  BCa (bias-corrected and accelerated) is more
        accurate but slower.
    seed : int, optional
        RNG seed.
    n_jobs : int
        Parallel workers for bootstrap replicates (``1`` = sequential,
        ``-1`` = all cores).  Requires ``joblib``.

    Returns
    -------
    estimate : float
        The statistic computed on the original data.
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    obs = statistic(data)

    # Per-replicate child seeds → bootstrap is identical for any n_jobs.
    child = rng.bit_generator._seed_seq.spawn(n_bootstrap)

    def _one(ss: np.random.SeedSequence) -> float:
        r = np.random.default_rng(ss)
        return statistic(data[r.choice(n, n, replace=True)])

    if n_jobs == 1:
        boot = np.array([_one(ss) for ss in child])
    else:
        from spectralbrain.backends.cpu import parallel_map

        boot = np.array(parallel_map(_one, child, n_jobs=n_jobs, description="Bootstrap"))

    alpha = (1 - ci) / 2
    if method == "percentile":
        return (
            obs,
            float(np.percentile(boot, 100 * alpha)),
            float(np.percentile(boot, 100 * (1 - alpha))),
        )
    elif method == "bca":
        z0 = sp_stats.norm.ppf(np.mean(boot < obs))
        jk = np.array([statistic(np.delete(data, i)) for i in range(n)])
        jm = jk.mean()
        a = np.sum((jm - jk) ** 3) / (6 * np.sum((jm - jk) ** 2) ** 1.5 + 1e-30)
        zl, zh = sp_stats.norm.ppf(alpha), sp_stats.norm.ppf(1 - alpha)
        pl = sp_stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
        ph = sp_stats.norm.cdf(z0 + (z0 + zh) / (1 - a * (z0 + zh)))
        return obs, float(np.percentile(boot, 100 * pl)), float(np.percentile(boot, 100 * ph))
    raise ValueError(f"Unknown method: {method!r}")


def bootstrap_paired_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for paired mean difference.

    Computes the CI of ``mean(a - b)`` via bootstrap resampling
    on the paired differences.

    Parameters
    ----------
    a, b : ndarray
        Paired observations (same length).
    n_bootstrap : int
        Number of bootstrap replicates.
    ci : float
        Confidence level.
    seed : int, optional
        RNG seed.

    Returns
    -------
    estimate : float
        Mean paired difference.
    ci_low, ci_high : float
        Confidence interval bounds.
    """
    return bootstrap_ci(
        np.asarray(a) - np.asarray(b), np.mean, n_bootstrap=n_bootstrap, ci=ci, seed=seed
    )


# ==== §2 NULL MODELS ====


def null_eigenvalue_permutation(eigenvalues, eigenvectors, *, n_surrogates=1000, seed=None):
    """Null 1: permute eigenvalues, keep eigenvectors. Tests spectral ordering."""
    rng = np.random.default_rng(seed)
    return [
        (eigenvalues[rng.permutation(len(eigenvalues))], eigenvectors.copy())
        for _ in range(n_surrogates)
    ]


def null_phase_randomisation(descriptor, eigenvectors, *, n_surrogates=1000, seed=None):
    """Null 2: randomise phases in spectral domain. Tests vertex-level structure."""
    rng = np.random.default_rng(seed)
    desc = np.atleast_2d(descriptor.T).T  # ensure (N, T)
    coeffs = eigenvectors.T @ desc
    amps = np.abs(coeffs)
    surrogates = []
    for _ in range(n_surrogates):
        phases = rng.uniform(0, 2 * np.pi, coeffs.shape)
        surrogates.append((eigenvectors @ (amps * np.cos(phases))).squeeze())
    return surrogates


def null_spin_permutation(descriptor, sphere_coords, *, n_surrogates=1000, seed=None, n_jobs=1):
    """Null 3: spin permutation (Alexander-Bloch 2018). Tests beyond spatial autocorrelation.

    Each spin is keyed to an independent, reproducible child RNG, so results
    are identical for ``n_jobs=1`` (sequential) and ``n_jobs=-1`` (all cores).
    The per-spin nearest-neighbour query over all vertices is the cost that
    parallelism amortises.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    tree = cKDTree(sphere_coords)
    desc = np.asarray(descriptor)
    child_seeds = np.random.SeedSequence(seed).spawn(n_surrogates)

    def _one_spin(child_seed: np.random.SeedSequence) -> np.ndarray:
        rng = np.random.default_rng(child_seed)
        rot = Rotation.random(random_state=rng)
        _, nearest = tree.query(rot.apply(sphere_coords))
        return desc[nearest]

    if n_jobs == 1:
        surrogates = []
        with progress_simple("Spin permutation", total=n_surrogates) as tick:
            for cs in child_seeds:
                surrogates.append(_one_spin(cs))
                tick(1)
        return surrogates

    from spectralbrain.backends.cpu import parallel_map

    return parallel_map(_one_spin, child_seeds, n_jobs=n_jobs, description="Spin permutation")


def null_subject_permutation(descriptors, labels, *, n_surrogates=5000, seed=None):
    """Null 4: permute group labels. Classic permutation test."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(labels) for _ in range(n_surrogates)]


def null_edge_rewiring(connectome, *, n_surrogates=1000, n_swaps_per_edge=10, seed=None, n_jobs=1):
    """Null 5: degree-preserving edge rewiring (Maslov & Sneppen 2002). Tests network topology.

    Each surrogate is keyed to an independent, reproducible child RNG via
    :class:`numpy.random.SeedSequence`, so results are **identical whether
    run sequentially (``n_jobs=1``) or in parallel** (``n_jobs=-1`` uses all
    cores). Rewiring is the expensive null (an inner swap loop per
    surrogate), so parallelism gives a near-linear speedup.
    """
    C = np.asarray(connectome, dtype=np.float64)
    rows, cols = np.triu_indices(C.shape[0], k=1)
    weights = C[rows, cols]
    nz = weights > 0
    er, ec, ew = rows[nz].copy(), cols[nz].copy(), weights[nz].copy()
    ne = len(er)
    child_seeds = np.random.SeedSequence(seed).spawn(n_surrogates)

    def _one_surrogate(child_seed: np.random.SeedSequence) -> np.ndarray:
        rng = np.random.default_rng(child_seed)
        r, c, w = er.copy(), ec.copy(), ew.copy()
        for _ in range(ne * n_swaps_per_edge):
            if ne < 2:
                break
            e1, e2 = rng.choice(ne, 2, replace=False)
            if rng.random() < 0.5:
                r[e1], r[e2] = r[e2], r[e1]
            else:
                c[e1], c[e2] = c[e2], c[e1]
        M = np.zeros_like(C)
        M[r, c] = w
        M += M.T
        return M

    if n_jobs == 1:
        surrogates = []
        with progress_simple("Edge rewiring", total=n_surrogates) as tick:
            for cs in child_seeds:
                surrogates.append(_one_surrogate(cs))
                tick(1)
        return surrogates

    from spectralbrain.backends.cpu import parallel_map

    return parallel_map(_one_surrogate, child_seeds, n_jobs=n_jobs, description="Edge rewiring")


def null_parametric(descriptor, *, n_surrogates=1000, seed=None):
    """Null 6: matched Gaussian surrogate. Tests beyond marginal distribution."""
    rng = np.random.default_rng(seed)
    desc = np.asarray(descriptor, dtype=np.float64)
    m, s = desc.mean(axis=0), desc.std(axis=0, ddof=1)
    return [rng.normal(m, s + 1e-30, desc.shape) for _ in range(n_surrogates)]


# ==== §3 SYNTHETIC DATA GENERATORS ====


class SyntheticDescriptors:
    """Generate synthetic spectral descriptor matrices.

    Parameters
    ----------
    reference : ndarray, optional
        Real data to match.
    distribution : str, optional
        Parametric family (default ``"normal"``).
    dist_params : dict, optional
    seed : int

    Examples
    --------
    >>> gen = SyntheticDescriptors(distribution="gamma", dist_params={"shape": 2})
    >>> data = gen.generate(n_subjects=50, n_vertices=1000, n_scales=20)
    """

    def __init__(self, reference=None, distribution=None, dist_params=None, seed=None):
        """Initialise the descriptor generator.

        Parameters
        ----------
        reference : ndarray, optional
            Real data to match via bootstrap resampling.
        distribution : str, optional
            Parametric distribution name (default ``"normal"``).
        dist_params : dict, optional
            Distribution-specific parameters.
        seed : int, optional
            RNG seed.
        """
        self.reference = reference
        self.distribution = distribution or "normal"
        self.dist_params = dist_params or {}
        self.rng = np.random.default_rng(seed)

    def generate(self, n_subjects=20, n_vertices=500, n_scales=10):
        """Returns shape (n_subjects, n_vertices, n_scales)."""
        shape = (n_subjects, n_vertices, n_scales)
        if self.reference is not None:
            return _match_distribution(self.reference, shape, self.rng)
        return _sample_distribution(self.distribution, shape, rng=self.rng, params=self.dist_params)

    def generate_global(self, n_subjects=20, n_features=50):
        """Returns shape (n_subjects, n_features)."""
        shape = (n_subjects, n_features)
        if self.reference is not None:
            return _match_distribution(self.reference, shape, self.rng)
        return _sample_distribution(self.distribution, shape, rng=self.rng, params=self.dist_params)


class SyntheticMesh:
    """Generate synthetic triangle meshes (sphere, ellipsoid, torus).

    Parameters
    ----------
    reference_vertices : ndarray, optional
    distribution, dist_params, seed : as above.
    """

    def __init__(self, reference_vertices=None, distribution=None, dist_params=None, seed=None):
        """Initialise the mesh generator.

        Parameters
        ----------
        reference_vertices : ndarray, optional
            Reference vertices for noise distribution matching.
        distribution : str, optional
            Noise distribution (default ``"normal"``).
        dist_params : dict, optional
            Distribution-specific parameters.
        seed : int, optional
            RNG seed.
        """
        self.reference = reference_vertices
        self.distribution = distribution or "normal"
        self.dist_params = dist_params or {}
        self.rng = np.random.default_rng(seed)

    def _noise(self, verts, scale):
        """Apply noise to vertices using the configured distribution."""
        if self.reference is not None:
            ref_std = self.reference.std(axis=0).mean()
            return verts + _match_distribution(
                self.reference - self.reference.mean(axis=0), verts.shape, self.rng
            ) * (scale * ref_std)
        return verts + _sample_distribution(
            self.distribution,
            verts.shape,
            rng=self.rng,
            params={**self.dist_params, "scale": scale},
        )

    def sphere(self, n_lat=30, n_lon=60, radius=50.0, noise=0.0):
        """Generate a UV-sphere mesh.

        Parameters
        ----------
        n_lat, n_lon : int
            Latitude and longitude subdivisions.
        radius : float
            Sphere radius.
        noise : float
            Noise scale (0 = no noise).

        Returns
        -------
        vertices : ndarray, shape (V, 3)
        faces : ndarray, shape (F, 3)
        """
        v, f = _uv_sphere(n_lat, n_lon, radius)
        return (self._noise(v, noise), f) if noise > 0 else (v, f)

    def ellipsoid(self, radii=(50, 30, 20), n_lat=30, n_lon=60, noise=0.0):
        """Generate an ellipsoid mesh.

        Parameters
        ----------
        radii : tuple of float
            Semi-axis lengths (a, b, c).
        n_lat, n_lon : int
            Latitude and longitude subdivisions.
        noise : float
            Noise scale.

        Returns
        -------
        vertices : ndarray, shape (V, 3)
        faces : ndarray, shape (F, 3)
        """
        v, f = _uv_sphere(n_lat, n_lon, 1.0)
        v *= np.array(radii)
        return (self._noise(v, noise), f) if noise > 0 else (v, f)

    def torus(self, R=40.0, r=15.0, n_major=40, n_minor=20, noise=0.0):
        """Generate a torus mesh.

        Parameters
        ----------
        R : float
            Major radius (center of tube to center of torus).
        r : float
            Minor radius (tube radius).
        n_major, n_minor : int
            Subdivisions along major and minor circles.
        noise : float
            Noise scale.

        Returns
        -------
        vertices : ndarray, shape (V, 3)
        faces : ndarray, shape (F, 3)
        """
        v, f = _torus(R, r, n_major, n_minor)
        return (self._noise(v, noise), f) if noise > 0 else (v, f)


class SyntheticPointCloud:
    """Generate synthetic point clouds (sphere, ellipsoid, blob, multi-cluster).

    Parameters
    ----------
    reference : ndarray, optional
    distribution, dist_params, seed : as above.
    """

    def __init__(self, reference=None, distribution=None, dist_params=None, seed=None):
        """Initialise the point cloud generator.

        Parameters
        ----------
        reference : ndarray, optional
            Reference point cloud for distribution matching.
        distribution : str, optional
            Noise distribution (default ``"normal"``).
        dist_params : dict, optional
            Distribution-specific parameters.
        seed : int, optional
            RNG seed.
        """
        self.reference = reference
        self.distribution = distribution or "normal"
        self.dist_params = dist_params or {}
        self.rng = np.random.default_rng(seed)

    def sphere(self, n_points=1000, radius=50.0, noise=0.5):
        """Generate random points on a sphere surface.

        Parameters
        ----------
        n_points : int
        radius : float
        noise : float
            Gaussian jitter magnitude.

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        pts = _random_sphere_points(n_points, radius, self.rng)
        if noise > 0:
            pts += self._noise_3d(n_points) * noise
        return pts

    def ellipsoid(self, n_points=1000, radii=(50, 30, 20), noise=0.5):
        """Generate random points on an ellipsoid surface.

        Parameters
        ----------
        n_points : int
        radii : tuple of float
            Semi-axis lengths (a, b, c).
        noise : float

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        pts = _random_sphere_points(n_points, 1.0, self.rng) * np.array(radii)
        if noise > 0:
            pts += self._noise_3d(n_points) * noise
        return pts

    def blob(self, n_points=1000, center=(0, 0, 0), scale=(10, 10, 10)):
        """Generate a Gaussian blob point cloud.

        Parameters
        ----------
        n_points : int
        center : tuple of float
        scale : tuple of float

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        return self.rng.normal(center, scale, (n_points, 3))

    def multi_cluster(self, n_points=1000, n_clusters=5, spread=30.0, cluster_std=5.0):
        """Generate a multi-cluster point cloud.

        Parameters
        ----------
        n_points : int
        n_clusters : int
        spread : float
            Spatial extent for cluster centres.
        cluster_std : float
            Standard deviation within each cluster.

        Returns
        -------
        ndarray, shape (~n_points, 3)
        """
        per = n_points // n_clusters
        centers = self.rng.uniform(-spread, spread, (n_clusters, 3))
        return np.vstack([self.rng.normal(c, cluster_std, (per, 3)) for c in centers])

    def from_reference(self, n_points=None):
        """Generate points matching the reference distribution.

        Parameters
        ----------
        n_points : int, optional
            Number of points (default: same as reference).

        Returns
        -------
        ndarray, shape (n_points, 3)
        """
        if self.reference is None:
            raise ValueError("No reference data.")
        return _match_distribution(
            self.reference, (n_points or self.reference.shape[0], 3), self.rng
        )

    def _noise_3d(self, n):
        """Generate 3D noise samples using the configured distribution."""
        if self.reference is not None:
            return _match_distribution(
                self.reference - self.reference.mean(axis=0), (n, 3), self.rng
            )
        return _sample_distribution(
            self.distribution, (n, 3), rng=self.rng, params={**self.dist_params, "scale": 1}
        )


# ==== Geometry helpers ====


def _uv_sphere(n_lat, n_lon, radius):
    """Build a UV-sphere mesh.

    Parameters
    ----------
    n_lat, n_lon : int
        Number of latitude / longitude divisions.
    radius : float
        Sphere radius.

    Returns
    -------
    vertices : ndarray, shape (V, 3)
    faces : ndarray, shape (F, 3)
    """
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                ]
            )
    verts = np.array(verts, dtype=np.float64)
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * n_lon + j
            v1 = i * n_lon + (j + 1) % n_lon
            v2 = (i + 1) * n_lon + j
            v3 = (i + 1) * n_lon + (j + 1) % n_lon
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return verts, np.array(faces, dtype=np.int64)


def _torus(R, r, n_major, n_minor):
    """Build a torus mesh.

    Parameters
    ----------
    R : float
        Major radius.
    r : float
        Minor (tube) radius.
    n_major, n_minor : int
        Subdivisions along the major and minor circles.

    Returns
    -------
    vertices : ndarray, shape (V, 3)
    faces : ndarray, shape (F, 3)
    """
    verts = []
    for i in range(n_major):
        theta = 2 * np.pi * i / n_major
        for j in range(n_minor):
            phi = 2 * np.pi * j / n_minor
            verts.append(
                [
                    (R + r * np.cos(phi)) * np.cos(theta),
                    (R + r * np.cos(phi)) * np.sin(theta),
                    r * np.sin(phi),
                ]
            )
    verts = np.array(verts, dtype=np.float64)
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            v0 = i * n_minor + j
            v1 = i * n_minor + (j + 1) % n_minor
            v2 = ((i + 1) % n_major) * n_minor + j
            v3 = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return verts, np.array(faces, dtype=np.int64)


def _random_sphere_points(n, radius, rng):
    """Sample *n* uniformly distributed points on a sphere surface.

    Uses the Marsaglia method (uniform z + uniform azimuth).

    Parameters
    ----------
    n : int
        Number of points.
    radius : float
        Sphere radius.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r_xy = np.sqrt(1 - z**2)
    return radius * np.column_stack([r_xy * np.cos(phi), r_xy * np.sin(phi), z])


__all__ = [
    "SyntheticDescriptors",
    "SyntheticMesh",
    "SyntheticPointCloud",
    "bootstrap_ci",
    "bootstrap_paired_difference",
    "null_edge_rewiring",
    "null_eigenvalue_permutation",
    "null_parametric",
    "null_phase_randomisation",
    "null_spin_permutation",
    "null_subject_permutation",
]
