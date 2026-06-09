"""Numerical tests for the spectral core on a synthetic triangulated sphere.

These tests exercise the actual numerical pipeline (Laplace-Beltrami
eigendecomposition, spectral descriptors, spectral distances) and verify
the mathematical properties they must satisfy, rather than only checking
that imports succeed.

The unit sphere is a convenient analytic benchmark: the eigenvalues of its
Laplace-Beltrami operator are ``l * (l + 1)`` with multiplicity ``2l + 1``
(``0, 2, 2, 2, 6, 6, ...``), which lets us validate the solver directly.
"""

import numpy as np
import pytest

import spectralbrain as sb


# ----------------------------------------------------------------------
# Synthetic geometry
# ----------------------------------------------------------------------
def _icosphere(subdivisions: int = 3):
    """Return (vertices, faces) of a unit icosphere.

    Parameters
    ----------
    subdivisions : int
        Number of 1-to-4 triangle subdivision passes applied to the base
        icosahedron. ``3`` gives 642 vertices / 1280 faces.
    """
    phi = (1.0 + 5.0**0.5) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(subdivisions):
        midpoint: dict[tuple[int, int], int] = {}
        vert_list = list(vertices)
        new_faces = []

        def _mid(a: int, b: int, _mp=midpoint, _vl=vert_list, _v=vertices) -> int:
            key = (min(a, b), max(a, b))
            if key not in _mp:
                _mp[key] = len(_vl)
                _vl.append((_v[a] + _v[b]) / 2.0)
            return _mp[key]

        for a, b, c in faces:
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        vertices = np.asarray(vert_list, dtype=np.float64)
        faces = np.asarray(new_faces, dtype=np.int64)

    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices, faces


@pytest.fixture(scope="module")
def sphere():
    """A unit icosphere mesh (642 vertices)."""
    vertices, faces = _icosphere(3)
    return sb.BrainMesh(vertices, faces)


@pytest.fixture(scope="module")
def decomp(sphere):
    """Spectral decomposition of the sphere with 60 eigenpairs."""
    return sphere.decompose(k=60)


# ----------------------------------------------------------------------
# Mesh construction
# ----------------------------------------------------------------------
def test_mesh_basic_properties(sphere):
    assert sphere.n_vertices == 642
    assert sphere.n_faces == 1280
    # A subdivided icosphere is a closed genus-0 surface.
    assert sphere.is_closed()
    assert sphere.euler_characteristic() == 2
    # Surface area of the unit sphere is 4*pi (mesh under-estimates slightly).
    assert sphere.surface_area() == pytest.approx(4 * np.pi, rel=0.02)


# ----------------------------------------------------------------------
# Eigendecomposition
# ----------------------------------------------------------------------
def test_eigenvalues_sorted_and_nonnegative(decomp):
    evals = decomp.eigenvalues
    assert evals.shape == (60,)
    assert np.all(np.diff(evals) >= -1e-9), "eigenvalues must be ascending"
    assert np.all(evals >= -1e-9), "LBO eigenvalues must be non-negative"


def test_first_eigenvalue_is_zero(decomp):
    # The smallest LBO eigenvalue of a connected closed surface is 0.
    assert abs(decomp.eigenvalues[0]) < 1e-6


def test_sphere_eigenvalue_spectrum(decomp):
    # Analytic spectrum of the unit sphere: l(l+1) with multiplicity 2l+1.
    # First non-trivial cluster (l=1) is a triple eigenvalue near 2.
    evals = decomp.eigenvalues
    assert evals[1] == pytest.approx(2.0, rel=0.05)
    assert evals[2] == pytest.approx(2.0, rel=0.05)
    assert evals[3] == pytest.approx(2.0, rel=0.05)
    # Second cluster (l=2) is a quintuple eigenvalue near 6.
    assert evals[4] == pytest.approx(6.0, rel=0.05)


def test_eigenvector_shape(decomp):
    assert decomp.eigenvectors.shape == (642, 60)
    assert np.all(np.isfinite(decomp.eigenvectors))


def test_eigenvectors_mass_orthonormal(decomp):
    # Eigenvectors are M-orthonormal: Phi^T M Phi = I.
    phi = decomp.eigenvectors
    gram = phi.T @ (decomp.mass @ phi)
    np.testing.assert_allclose(gram, np.eye(phi.shape[1]), atol=1e-6)


# ----------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------
def test_hks_shape_and_positivity(decomp):
    hks = sb.compute_hks(decomp, t_values=np.array([1.0, 10.0, 100.0]))
    assert hks.shape == (642, 3)
    assert np.all(np.isfinite(hks))
    # HKS is a sum of squared eigenfunctions weighted by positive decays.
    assert np.all(hks >= -1e-10)


def test_wks_shape_and_finiteness(decomp):
    wks = sb.compute_wks(decomp, n_energies=50)
    assert wks.shape == (642, 50)
    assert np.all(np.isfinite(wks))


def test_gps_shape(decomp):
    gps = sb.compute_gps(decomp)
    assert gps.shape[0] == 642
    assert np.all(np.isfinite(gps))


def test_shapedna_shape_and_finiteness(decomp):
    dna = sb.compute_shapedna(decomp)
    # skip_zero=True drops the trivial first eigenvalue.
    assert dna.shape == (59,)
    assert np.all(np.isfinite(dna))


def test_compute_all_descriptors_keys(decomp):
    out = sb.compute_all_descriptors(decomp)
    for key in ("shapedna", "hks", "wks", "gps", "bks"):
        assert key in out
        assert np.all(np.isfinite(np.asarray(out[key])))


# ----------------------------------------------------------------------
# Spectral distances
# ----------------------------------------------------------------------
def test_biharmonic_distance_is_a_metric_shape(decomp):
    dist = sb.biharmonic_distance(decomp)
    assert dist.shape == (642, 642)
    assert np.all(np.isfinite(dist))
    # Symmetry and zero self-distance.
    np.testing.assert_allclose(dist, dist.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)
    assert np.all(dist >= -1e-9)


def test_commute_time_distance_symmetry(decomp):
    dist = sb.commute_time_distance(decomp)
    assert dist.shape == (642, 642)
    np.testing.assert_allclose(dist, dist.T, atol=1e-8)
    assert np.all(np.isfinite(dist))


# ----------------------------------------------------------------------
# Isometry invariance (the defining property of spectral descriptors)
# ----------------------------------------------------------------------
def test_shapedna_invariant_under_rotation(sphere, decomp):
    # A rigid rotation is an isometry: the LBO spectrum must be unchanged.
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = sb.BrainMesh(sphere.coordinates @ rot.T, sphere.faces)
    dna_rotated = sb.compute_shapedna(rotated.decompose(k=60))
    dna_original = sb.compute_shapedna(decomp)
    np.testing.assert_allclose(dna_rotated, dna_original, rtol=1e-3, atol=1e-4)


def test_shapedna_distance_zero_for_identical_shape(decomp):
    d_self = sb.shapedna_distance(decomp.eigenvalues, decomp.eigenvalues)
    assert d_self == pytest.approx(0.0, abs=1e-9)
