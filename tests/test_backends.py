"""Tests for the compute and Bayesian backends.

The GPU-oriented backends (CuPy, JAX, Torch) and the JAX-based samplers
(NumPyro, BlackJAX) all run on CPU when no accelerator is present, so the
ones whose dependencies are installable on CPU are exercised here. Tests
``importorskip`` their dependency, so a minimal install skips them rather
than failing.

The reference for every general backend is :class:`NumpyBackend`: on the
analytic unit sphere the Laplace–Beltrami eigenvalues are ``l*(l+1)``
(``0, 2, 2, 2, 6, 6, ...``), so any backend's ``eigsh`` must reproduce the
NumPy result to near machine precision.
"""

import numpy as np
import pytest

import spectralbrain as sb
from spectralbrain.backends import NumpyBackend


def _icosphere(subdivisions: int = 3):
    """Return (vertices, faces) of a unit icosphere."""
    phi = (1.0 + 5.0**0.5) / 2.0
    v = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    f = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    for _ in range(subdivisions):
        v, f = _subdivide(v, f)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v, f


def _subdivide(v, f):
    """One 1-to-4 triangle subdivision pass."""
    midpoint: dict[tuple[int, int], int] = {}
    verts = list(v)

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoint:
            verts.append((v[a] + v[b]) / 2.0)
            midpoint[key] = len(verts) - 1
        return midpoint[key]

    new_faces = []
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)


# ----------------------------------------------------------------------
# General compute backends
# ----------------------------------------------------------------------
def test_numpy_backend_recovers_sphere_spectrum():
    """NumpyBackend reproduces the analytic sphere eigenvalues."""
    v, f = _icosphere(3)
    decomp = sb.BrainMesh(v, f).decompose(k=10, backend=NumpyBackend())
    ev = np.sort(decomp.eigenvalues)
    assert abs(ev[0]) < 1e-6
    # First non-trivial triplet ≈ 2 (l=1 → l(l+1)=2).
    assert np.allclose(ev[1:4], 2.0, atol=0.1)


def test_torch_backend_matches_numpy():
    """TorchBackend.eigsh matches NumpyBackend to near machine precision."""
    pytest.importorskip("torch")
    from spectralbrain.backends import TorchBackend

    v, f = _icosphere(3)
    ref = sb.BrainMesh(v, f).decompose(k=20, backend=NumpyBackend()).eigenvalues
    got = sb.BrainMesh(v, f).decompose(k=20, backend=TorchBackend()).eigenvalues
    assert np.max(np.abs(np.sort(np.asarray(got)) - np.sort(ref))) < 1e-6


def test_get_gpu_backend_factory():
    """The factory returns the requested backend type (CPU fallback is fine)."""
    pytest.importorskip("torch")
    from spectralbrain.backends import TorchBackend, get_gpu_backend

    assert isinstance(get_gpu_backend("torch"), TorchBackend)
    with pytest.raises(ValueError, match="Unknown GPU backend"):
        get_gpu_backend("does-not-exist")


# ----------------------------------------------------------------------
# Bayesian samplers
# ----------------------------------------------------------------------
def test_blackjax_sampler_recovers_mean():
    """BlackjaxSampler recovers the mean of a Gaussian target."""
    pytest.importorskip("blackjax")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from spectralbrain.backends import BlackjaxSampler

    rng = np.random.default_rng(0)
    data = jnp.asarray(rng.normal(3.0, 1.0, size=60))

    def logdensity(theta):
        mu = theta["mu"]
        return (
            jax.scipy.stats.norm.logpdf(mu, 0.0, 10.0)
            + jax.scipy.stats.norm.logpdf(data, mu, 1.0).sum()
        )

    sampler = BlackjaxSampler(num_warmup=200, num_samples=300, num_chains=1, seed=0)
    samples = sampler.sample(logdensity, {"mu": jnp.array(0.0)})
    mu_hat = float(np.asarray(samples["mu"]).mean())
    assert abs(mu_hat - 3.0) < 0.3
    assert np.asarray(samples["mu"]).shape == (300,)


def test_blackjax_sampler_multichain_shape():
    """Multi-chain sampling adds a leading chain axis."""
    pytest.importorskip("blackjax")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from spectralbrain.backends import BlackjaxSampler

    def logdensity(theta):
        return jax.scipy.stats.norm.logpdf(theta["x"], 0.0, 1.0)

    sampler = BlackjaxSampler(num_warmup=100, num_samples=150, num_chains=4, seed=1)
    samples = sampler.sample(logdensity, {"x": jnp.array(0.0)})
    assert np.asarray(samples["x"]).shape == (4, 150)


def test_get_gpu_bayesian_sampler_factory():
    """The Bayesian factory dispatches by backend name."""
    pytest.importorskip("blackjax")
    from spectralbrain.backends import BlackjaxSampler, get_gpu_bayesian_sampler

    assert isinstance(get_gpu_bayesian_sampler("blackjax"), BlackjaxSampler)
    with pytest.raises(ValueError, match="Unknown GPU Bayesian backend"):
        get_gpu_bayesian_sampler("does-not-exist")


# ----------------------------------------------------------------------
# CPU parallelism (joblib)
# ----------------------------------------------------------------------
def test_parallel_map_with_progress_is_picklable():
    """parallel_map must not pickle its progress object into workers.

    Regression guard: an earlier version captured the Rich progress bar
    (which holds a thread lock) in the worker closure, breaking the
    process-based ``loky`` backend with a PicklingError.
    """
    from spectralbrain.backends import parallel_map

    out = parallel_map(abs, list(range(-5, 5)), n_jobs=2, progress=True, description="abs")
    assert out == [abs(i) for i in range(-5, 5)]


def test_null_edge_rewiring_njobs_invariant():
    """Edge-rewiring surrogates are identical sequentially and in parallel."""
    import numpy as np

    from spectralbrain.statistics.surrogates import null_edge_rewiring

    rng = np.random.default_rng(0)
    c = rng.random((30, 30))
    c = (c + c.T) / 2
    np.fill_diagonal(c, 0)
    c[c < 0.7] = 0

    seq = null_edge_rewiring(c, n_surrogates=12, n_swaps_per_edge=4, seed=42, n_jobs=1)
    par = null_edge_rewiring(c, n_surrogates=12, n_swaps_per_edge=4, seed=42, n_jobs=2)
    assert all(np.allclose(a, b) for a, b in zip(seq, par))


def test_null_spin_permutation_njobs_invariant():
    """Spin-permutation surrogates are identical sequentially and in parallel."""
    import numpy as np

    from spectralbrain.statistics.surrogates import null_spin_permutation

    rng = np.random.default_rng(1)
    coords = rng.standard_normal((100, 3))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    desc = rng.standard_normal(100)

    seq = null_spin_permutation(desc, coords, n_surrogates=10, seed=3, n_jobs=1)
    par = null_spin_permutation(desc, coords, n_surrogates=10, seed=3, n_jobs=2)
    assert all(np.allclose(a, b) for a, b in zip(seq, par))
