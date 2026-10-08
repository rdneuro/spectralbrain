"""Regression tests for the spectral (lbo + operators) bug-fix pass.

Each test pins one previously verified bug (numbering follows the audit
report) so that it cannot silently come back.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.core.meshes import BrainMesh, _cotangent_laplacian, _vertex_normals

# ----------------------------------------------------------------------
# Synthetic geometry
# ----------------------------------------------------------------------


def _icosphere(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    V = [np.asarray(v, float) / np.linalg.norm(v) for v in verts]
    for _ in range(n):
        cache: dict[tuple[int, int], int] = {}
        new_faces = []

        def mid(a: int, b: int, cache: dict[tuple[int, int], int] = cache) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.array(V), np.array(faces, dtype=np.int64)


def _bumped(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """An asymmetric closed surface (no intrinsic symmetries)."""
    V, F = _icosphere(n)
    x, y, z = V.T
    r = (
        1
        + 0.25 * np.exp(-((x - 0.6) ** 2 + (y - 0.5) ** 2 + (z - 0.6) ** 2) / 0.1)
        + 0.15 * np.exp(-((x + 0.3) ** 2 + (y + 0.8) ** 2 + (z - 0.2) ** 2) / 0.05)
        + 0.1 * x * y
    )
    return V * r[:, None] * np.array([1.3, 1.0, 0.8]), F


def _grid(n: int = 15, lx: float = 1.0, ly: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.linspace(0, lx, n), np.linspace(0, ly, n))
    P = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)]
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            F += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return P, np.array(F, dtype=np.int64)


@pytest.fixture(scope="module")
def ellipsoid_decomp() -> SpectralDecomposition:
    V, F = _icosphere(3)
    return BrainMesh(V * np.array([1.2, 1.0, 0.9]), F).decompose(k=20)


def _flip_sign(decomp: SpectralDecomposition, col: int) -> SpectralDecomposition:
    E = decomp.eigenvectors.copy()
    E[:, col] *= -1
    return SpectralDecomposition(
        decomp.eigenvalues,
        E,
        stiffness=decomp.stiffness,
        mass=decomp.mass,
        surface_area=decomp.surface_area,
    )


# ----------------------------------------------------------------------
# spectral/lbo/anisotropic.py  (#1, #2, #3)
# ----------------------------------------------------------------------


def test_anisotropic_laplacian_symmetric_zero_rowsum() -> None:
    from spectralbrain.spectral.lbo.anisotropic import anisotropic_laplacian

    V, F = _icosphere(3)
    L, _ = anisotropic_laplacian(V, F, anisotropy=0.5)
    assert abs(L - L.T).max() < 1e-12
    assert np.abs(np.asarray(L.sum(axis=1))).max() < 1e-10
    offdiag = np.asarray(L.sum(axis=1)).ravel() - L.diagonal()
    np.testing.assert_allclose(L.diagonal(), -offdiag, atol=1e-10)
    # PSD (smallest eigenvalue ~ 0)
    w = np.linalg.eigvalsh(L.toarray())
    assert w.min() > -1e-9


def test_min_curvature_direction_is_tangent() -> None:
    from spectralbrain.spectral.lbo.anisotropic import _estimate_curvature_directions

    V, F = _icosphere(2)
    d = _estimate_curvature_directions(V, F, "min")
    n = _vertex_normals(V, F)
    assert np.abs(np.sum(d * n, axis=1)).max() < 0.2


# ----------------------------------------------------------------------
# spectral/operators/finsler.py  (#4)
# ----------------------------------------------------------------------


def test_finsler_conductivity_along_direction() -> None:
    from spectralbrain.spectral.operators.finsler import finsler_laplacian

    P, F = _grid(21)
    cd = np.tile([1.0, 0.0, 0.0], (P.shape[0], 1))
    L, M = finsler_laplacian(P, F, anisotropy_ratio=4.0, custom_directions=cd)
    w, U = sla.eigh(L.toarray(), M.toarray())
    # first non-trivial mode should vary along y (slow direction, lambda~pi^2)
    u = U[:, 1]
    assert abs(np.corrcoef(u, np.cos(np.pi * P[:, 1]))[0, 1]) > 0.99
    assert w[1] == pytest.approx(np.pi**2, rel=0.05)
    # the x-varying mode is sped up by g = 4
    lam_x = [
        w[i]
        for i in range(1, 6)
        if abs(np.corrcoef(U[:, i], np.cos(np.pi * P[:, 0]))[0, 1]) > 0.99
    ]
    assert lam_x and lam_x[0] == pytest.approx(4 * np.pi**2, rel=0.05)


# ----------------------------------------------------------------------
# spectral/operators/hamiltonian.py  (#5, #14, #26)
# ----------------------------------------------------------------------


def test_hamiltonian_signed_potential_not_clamped() -> None:
    from spectralbrain.spectral.operators.hamiltonian import hamiltonian_decompose

    V, F = _icosphere(3)
    lbo = BrainMesh(V, F).decompose(k=10).eigenvalues
    d = hamiltonian_decompose(V, F, potential=-3.0 * np.ones(len(V)), k=10, backend="numpy")
    np.testing.assert_allclose(d.eigenvalues, lbo - 3.0, atol=1e-6)


def test_compressed_modes_differ_from_lbo() -> None:
    from spectralbrain.spectral.operators.hamiltonian import compute_compressed_modes

    V, F = _bumped(3)
    V = V * 50.0  # mm-like scale
    lbo = BrainMesh(V, F).decompose(k=6)
    cm = compute_compressed_modes(V, F, k=6, n_iter=2, mu=5.0, backend="numpy")
    # potential is commensurate with the spectrum, not a constant ~mu^2 shift
    assert cm.eigenvalues.max() < 50 * lbo.eigenvalues.max()
    overlap = np.abs(cm.eigenvectors.T @ (lbo.mass @ lbo.eigenvectors))
    assert overlap.max(axis=1).min() < 0.99


def test_siwks_scale_invariant() -> None:
    from spectralbrain.spectral.operators.hamiltonian import compute_siwks

    V, F = _bumped(2)
    a = compute_siwks(V, F, k=20, n_energies=10, potential="zero", backend="numpy")
    b = compute_siwks(3.0 * V, F, k=20, n_energies=10, potential="zero", backend="numpy")
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-8)


# ----------------------------------------------------------------------
# spectral/lbo/collections.py  (#6, #7, #20)
# ----------------------------------------------------------------------


def test_functional_self_map_is_identity() -> None:
    from spectralbrain.spectral.lbo.collections import compute_functional_map

    V, F = _bumped(3)
    d = BrainMesh(V, F).decompose(k=30)
    C = compute_functional_map(d, d, n_basis=20)
    assert np.linalg.norm(C - np.eye(20)) / np.sqrt(20) < 0.1


def test_conformal_shape_difference_formula() -> None:
    from spectralbrain.spectral.lbo.collections import shape_difference_operator

    rng = np.random.default_rng(0)
    C = rng.normal(size=(5, 5))
    la = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lb = np.array([0.0, 1.5, 2.5, 3.5, 4.5])
    D = shape_difference_operator(C, type="conformal", evals_a=la, evals_b=lb)
    inv = np.array([0.0, 1.0, 0.5, 1 / 3, 0.25])
    np.testing.assert_allclose(D, C.T @ np.diag(lb) @ C @ np.diag(inv))
    with pytest.raises(ValueError):
        shape_difference_operator(C, type="conformal")


@pytest.mark.parametrize("diff_type", ["area", "conformal"])
@pytest.mark.parametrize("n_energies", [5, 50])
def test_dwks_finite(diff_type: str, n_energies: int) -> None:
    from spectralbrain.spectral.lbo.collections import compute_dwks

    V, F = _bumped(2)
    a = BrainMesh(V, F).decompose(k=20)
    b = BrainMesh(V * np.array([1.1, 1.0, 1.0]), F).decompose(k=20)
    w = compute_dwks(a, b, n_basis=15, n_energies=n_energies, diff_type=diff_type)
    assert w.shape == (V.shape[0], n_energies)
    assert np.all(np.isfinite(w))


# ----------------------------------------------------------------------
# spectral/lbo/descriptors.py, wavelets.py, distances.py  (#8, #9, #15, #16, #17, #26b)
# ----------------------------------------------------------------------


def test_bates_sign_invariant(ellipsoid_decomp: SpectralDecomposition) -> None:
    from spectralbrain.spectral.lbo.descriptors import compute_bates_signatures, compute_hks

    t = np.array([0.1, 0.5, 1.0])
    b1 = compute_bates_signatures(ellipsoid_decomp, t_values=t, order=3)
    b2 = compute_bates_signatures(_flip_sign(ellipsoid_decomp, 3), t_values=t, order=3)
    np.testing.assert_allclose(b1, b2, atol=1e-12)
    # e1 equals the HKS
    hks = compute_hks(ellipsoid_decomp, t_values=t)
    np.testing.assert_allclose(b1[:, 0::3], hks, rtol=1e-10)


def test_sgw_descriptor_sign_invariant(ellipsoid_decomp: SpectralDecomposition) -> None:
    from spectralbrain.spectral.lbo.wavelets import sgw_descriptor

    s1 = sgw_descriptor(ellipsoid_decomp)
    s2 = sgw_descriptor(_flip_sign(ellipsoid_decomp, 3))
    np.testing.assert_allclose(s1, s2, atol=1e-12)


def test_gps_skip_zero_false_no_blowup(ellipsoid_decomp: SpectralDecomposition) -> None:
    from spectralbrain.spectral.lbo.descriptors import compute_gps

    g = compute_gps(ellipsoid_decomp, skip_zero=False)
    assert g.shape[1] == ellipsoid_decomp.n_eigenvalues
    assert np.all(g[:, 0] == 0.0)
    assert np.abs(g).max() < 10.0


def test_shapedna_fiedler_skip_zero_false(ellipsoid_decomp: SpectralDecomposition) -> None:
    from spectralbrain.spectral.lbo.descriptors import compute_shapedna

    dna = compute_shapedna(ellipsoid_decomp, normalize="fiedler", skip_zero=False)
    assert dna[1] == pytest.approx(1.0)
    assert np.all(np.isfinite(dna))


def test_shared_hks_times(ellipsoid_decomp: SpectralDecomposition) -> None:
    from spectralbrain.spectral.lbo.descriptors import compute_hks, shared_hks_times

    V, F = _icosphere(3)
    other = BrainMesh(2.0 * V, F).decompose(k=20)
    t = shared_hks_times([ellipsoid_decomp, other], n_times=7)
    assert t.shape == (7,) and np.all(np.diff(t) > 0)
    assert compute_hks(other, t_values=t).shape == (V.shape[0], 7)


def test_correlation_distance_scalar_raises() -> None:
    from spectralbrain.spectral.lbo.distances import descriptor_distance

    with pytest.raises(ValueError):
        descriptor_distance(np.arange(5.0), np.arange(7.0), method="correlation")


def test_sgw_transform_uses_mass(ellipsoid_decomp: SpectralDecomposition) -> None:
    from spectralbrain.spectral.lbo.wavelets import heat_kernel, sgw_transform

    d = ellipsoid_decomp
    f = d.eigenvectors[:, 4].copy()  # an exact eigenfunction of M^-1L
    out = sgw_transform(
        d.stiffness, np.array([0.3]), signal=f, kernel=heat_kernel,
        chebyshev_order=40, mass=d.mass,
    )  # fmt: skip
    np.testing.assert_allclose(out[0], np.exp(-0.3 * d.eigenvalues[4]) * f, atol=1e-4)


# ----------------------------------------------------------------------
# spectral/operators/persistent.py  (#10, #11, #12)
# ----------------------------------------------------------------------


def test_persistent_laplacian_schur_complement() -> None:
    from spectralbrain.spectral.operators.persistent import persistent_laplacian

    # path a - c - b, K = {a, b}: one persistent component
    A = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], float)
    L = persistent_laplacian(A, subset=np.array([0, 1])).toarray()
    np.testing.assert_allclose(L, [[0.5, -0.5], [-0.5, 0.5]], atol=1e-12)
    assert np.sum(np.abs(np.linalg.eigvalsh(L)) < 1e-9) == 1
    # an unreachable component in the complement must not break it
    A2 = np.zeros((5, 5))
    A2[0, 2] = A2[2, 0] = A2[1, 2] = A2[2, 1] = 1.0
    A2[3, 4] = A2[4, 3] = 1.0
    L2 = persistent_laplacian(A2, subset=np.array([0, 1])).toarray()
    np.testing.assert_allclose(L2, L, atol=1e-12)


def test_betti_curve_counts_essential_bars() -> None:
    from spectralbrain.spectral.operators.persistent import betti_curve, graph_persistence_h0

    dg = graph_persistence_h0(3, np.array([[0, 1], [1, 2]]), np.array([1.0, 2.0]))
    curve = betti_curve(dg, n_bins=5, value_range=(0.0, 2.5))
    # t = 0, 0.625 -> 3 comps; 1.25, 1.875 -> 2; 2.5 -> 1
    np.testing.assert_allclose(curve, [3, 3, 2, 2, 1])


def test_persistence_image_ranges_and_pixels() -> None:
    from spectralbrain.spectral.operators.persistent import persistence_image

    dg = np.array([[10.0, 30.0], [20.0, 50.0], [0.0, np.inf]])
    img = persistence_image(dg, pixels=(12, 8), spread=2.0)
    assert img.shape == (12, 8)
    # total mass ~ Sigma persistence (bars well inside the fitted range)
    assert img.sum() == pytest.approx(20.0 + 30.0, rel=0.05)
    fixed = persistence_image(dg, pixels=(10, 10), birth_range=(0, 40), pers_range=(0, 40))
    assert fixed.shape == (10, 10) and fixed.sum() > 0


# ----------------------------------------------------------------------
# spectral/operators/graphs.py  (#13, #19, #25)
# ----------------------------------------------------------------------


def test_netlsd_complete_normalisation() -> None:
    from spectralbrain.spectral.operators.graphs import netlsd

    n = 6
    A = np.ones((n, n)) - np.eye(n)
    t = np.array([0.1, 1.0, 10.0])
    np.testing.assert_allclose(netlsd(A, normalize="complete", timescales=t), 1.0, rtol=1e-8)
    np.testing.assert_allclose(
        netlsd(A, kind="wave", normalize="complete", timescales=t), 1.0, rtol=1e-8
    )


def test_normalized_laplacian_isolated_node() -> None:
    from spectralbrain.spectral.operators.graphs import _laplacian_spectrum, netlsd

    A = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float))
    np.testing.assert_allclose(_laplacian_spectrum(A, normalized=True, k=None), [0, 0, 2], atol=1e-12)
    h = netlsd(A, timescales=np.array([1e4]))
    assert h[0] == pytest.approx(2.0)  # b0 = 2 components


def test_fgsd_biharmonic_filter() -> None:
    from spectralbrain.spectral.operators.graphs import fgsd

    # path graph on 3 nodes
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    L = np.diag(A.sum(1)) - A
    w, V = np.linalg.eigh(L)
    w, V = w[1:], V[:, 1:]
    S = ((V[0] - V[2]) ** 2 / w**2).sum()
    h = fgsd(A, bins=3, hist_range=(0.0, 3.0), density=False, f="biharmonic")
    # pairs (0,1),(1,2) share one value; (0,2) has S
    assert h.sum() == 3
    assert h[int(S // 1.0)] >= 1
    fgsd(A, bins=4, hist_range=(0.0, 3.0), f="harmonic")


# ----------------------------------------------------------------------
# spectral/operators/_base.py, extrinsic.py  (#21, #22, #23)
# ----------------------------------------------------------------------


def test_rotate_coord_sys_perpendicular() -> None:
    from spectralbrain.spectral.operators._base import _rotate_coord_sys

    up = np.array([[1.0, 0, 0]])
    vp = np.array([[0, 1.0, 0]])
    nn = np.array([[0, np.sin(0.5), np.cos(0.5)]])
    a, b = _rotate_coord_sys(up, vp, nn)
    assert abs(np.sum(a * nn)) < 1e-12 and abs(np.sum(b * nn)) < 1e-12
    assert abs(np.sum(a * b)) < 1e-12


def test_shape_index_umbilic_sphere() -> None:
    from spectralbrain.spectral.operators._base import _shape_index_from_k, shape_index

    V, F = _icosphere(3)
    s = shape_index(V, F)
    assert s.min() > 0.9  # convex sphere -> ~ +1 everywhere
    np.testing.assert_allclose(
        _shape_index_from_k(np.array([1.0, -1.0, 0.0]), np.array([1.0, -1.0, 0.0])), [1, -1, 0]
    )


def test_dks_uses_squared_dirac_spectrum() -> None:
    from spectralbrain.spectral.operators.extrinsic import compute_dks, dirac_decompose

    V, F = _icosphere(2)
    d = dirac_decompose(V, F, k=12, backend="numpy")
    assert d.metadata["squared_dirac"] is True
    assert d.eigenvalues.min() > -1e-8  # PSD as documented
    t = np.array([0.5])
    dks = compute_dks(V, F, k=12, t_values=t, backend="numpy")
    mu = np.clip(d.eigenvalues, 0, None)
    expected = (d.eigenvectors**2) @ np.exp(-mu * 0.5)
    np.testing.assert_allclose(dks[:, 0], expected, rtol=1e-6)


# ----------------------------------------------------------------------
# spectral/operators/topologic.py  (#24)
# ----------------------------------------------------------------------


def test_ricci_warns_when_weights_ignored(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("networkx")
    from spectralbrain.spectral.operators.topologic import (
        forman_ricci_curvature,
        ollivier_ricci_curvature,
    )

    A = np.array([[0, 2.0, 1.0], [2.0, 0, 0.5], [1.0, 0.5, 0]])
    with caplog.at_level("WARNING"):
        forman_ricci_curvature(A)
        ollivier_ricci_curvature(A, method="builtin")
    assert sum("weights" in r.getMessage() for r in caplog.records) >= 2


def test_cotangent_reference_unchanged() -> None:
    """Sanity: the isotropic path of anisotropic_laplacian is the cotan L."""
    from spectralbrain.spectral.lbo.anisotropic import anisotropic_laplacian

    V, F = _icosphere(2)
    L0, _ = _cotangent_laplacian(V, F)
    L, _ = anisotropic_laplacian(V, F, anisotropy=0.0)
    assert abs(L - L0).max() < 1e-14
