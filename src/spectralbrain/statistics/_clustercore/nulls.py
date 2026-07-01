"""Spatial null models that preserve spatial autocorrelation.

For surfaces **with a boundary** (e.g. the unfolded hippocampal sheet) the spin
test is invalid because spherical projection distorts inter-vertex distances and
inflates false positives (Bazinet, Liu & Misic 2025). brainmosaic therefore uses
geometry-aware nulls:

- :func:`eigenstrapping_surrogates` — self-contained random rotation of
  Laplace-Beltrami geometric eigenmodes (Koussis, Pang et al. 2025). Reuses
  brainmosaic's own LBO eigenpairs, so it works natively on bounded surfaces.
- :func:`brainsmash_surrogates` — variogram-matched surrogates from a geodesic
  distance matrix (Burt et al. 2020), via the optional ``brainsmash`` package.
- :func:`paired_label_permutation` — within-subject ipsi/contra label swaps for
  paired designs.

References
----------
- Koussis, Pang, Phogat et al. (2025), Generation of surrogate brain maps...
  geometric eigenmodes, Imaging Neuroscience 3:IMAG.a.71.
- Bazinet, Liu & Misic (2025), The effect of spherical projection on spin tests,
  Imaging Neuroscience 3:IMAG.a.118.
- Burt, Helmer, Shinn, Anticevic & Murray (2020), Generative modeling of brain
  maps with spatial autocorrelation, NeuroImage 220:117038.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger("spectralbrain.statistics._clustercore")


def eigenstrapping_surrogates(data: np.ndarray, eigenvalues: np.ndarray,
                              eigenvectors: np.ndarray, mass: np.ndarray,
                              n_surrogates: int = 1000,
                              eigenvalue_tol: float = 1e-3,
                              random_state: int = 0) -> np.ndarray:
    """Generate SA-preserving surrogates by rotating LBO geometric eigenmodes.

    The map is expanded in the (M-orthonormal) LBO eigenbasis; eigenmodes sharing
    a (near-)degenerate eigenvalue are grouped and a random orthogonal rotation
    is applied within each group; the surrogate is reconstructed from the rotated
    coefficients. Rotations within an eigenvalue group preserve the power
    spectrum and hence the spatial autocorrelation, while randomising phase.

    Parameters
    ----------
    data : np.ndarray, shape (V,)
        Scalar map on the mesh vertices.
    eigenvalues : np.ndarray, shape (K,)
    eigenvectors : np.ndarray, shape (V, K)
        M-orthonormal LBO eigenvectors (from
        :func:`brainmosaic.core.decompose_laplacian`).
    mass : np.ndarray, shape (V,)
        Lumped mass diagonal (vertex areas) for the M-inner product.
    n_surrogates : int
    eigenvalue_tol : float
        Relative tolerance for grouping near-degenerate eigenvalues.
    random_state : int

    Returns
    -------
    np.ndarray, shape (n_surrogates, V)
        Surrogate maps with matched spatial autocorrelation.

    Notes
    -----
    This is the recommended null for bounded surfaces. It does not perform
    residual/amplitude matching of the original eigenstrapping variants; the
    constant mode is preserved (kept fixed) so surrogates share the map's mean.
    """
    data = np.asarray(data, float).ravel()
    evals = np.asarray(eigenvalues, float).ravel()
    evecs = np.asarray(eigenvectors, float)
    m = np.asarray(mass, float).ravel()
    V, K = evecs.shape
    if data.shape[0] != V:
        raise ValueError(f"data length {data.shape[0]} != #vertices {V}.")

    # Coefficients via the M-inner product: c_k = <phi_k, data>_M.
    coeffs = evecs.T @ (m * data)               # (K,)

    # Group modes by near-degenerate eigenvalue.
    groups = []
    start = 0
    for k in range(1, K + 1):
        if k == K or abs(evals[k] - evals[start]) > eigenvalue_tol * (abs(evals[start]) + 1e-12):
            groups.append(list(range(start, k)))
            start = k

    rng = np.random.default_rng(random_state)
    surrogates = np.empty((n_surrogates, V), dtype=np.float64)
    for s in range(n_surrogates):
        rot_coeffs = coeffs.copy()
        for grp in groups:
            if len(grp) == 1:
                continue  # constant mode / non-degenerate: keep fixed
            g = np.asarray(grp)
            # Random orthogonal matrix via QR of a Gaussian (Haar-ish).
            A = rng.standard_normal((g.size, g.size))
            Q, R = np.linalg.qr(A)
            Q *= np.sign(np.diag(R))            # fix QR sign ambiguity
            rot_coeffs[g] = Q @ coeffs[g]
        surrogates[s] = evecs @ rot_coeffs
    return surrogates


def brainsmash_surrogates(data: np.ndarray, distance: np.ndarray,
                          n_surrogates: int = 1000, **kwargs) -> np.ndarray:
    """Variogram-matched surrogates via the optional ``brainsmash`` package.

    Parameters
    ----------
    data : np.ndarray, shape (V,)
    distance : np.ndarray, shape (V, V)
        Geodesic distance matrix on the surface.
    n_surrogates : int
    **kwargs
        Passed to ``brainsmash.mapgen.base.Base``.

    Returns
    -------
    np.ndarray, shape (n_surrogates, V)

    Raises
    ------
    ImportError
        If brainsmash is not installed.
    """
    try:
        from brainsmash.mapgen.base import Base  # lazy
    except Exception as exc:
        raise ImportError(
            "brainsmash_surrogates requires the optional 'brainsmash' package "
            "(pip install brainsmash)."
        ) from exc
    gen = Base(x=np.asarray(data, float), D=np.asarray(distance, float), **kwargs)
    return np.asarray(gen(n=n_surrogates))


def paired_label_permutation(values_ipsi: np.ndarray, values_contra: np.ndarray,
                             n_perm: int = 5000, random_state: int = 0):
    """Paired permutation by random within-subject ipsi/contra swaps.

    Parameters
    ----------
    values_ipsi, values_contra : np.ndarray, shape (n_subjects,)
        Per-subject summary statistics for the two hemispheres.
    n_perm : int
    random_state : int

    Returns
    -------
    observed : float
        Mean paired difference (ipsi - contra).
    p_value : float
        Two-sided permutation p-value.
    null : np.ndarray, shape (n_perm,)
        Null distribution of the mean paired difference.
    """
    ipsi = np.asarray(values_ipsi, float)
    contra = np.asarray(values_contra, float)
    if ipsi.shape != contra.shape:
        raise ValueError("ipsi and contra must have the same shape.")
    diff = ipsi - contra
    observed = float(diff.mean())
    rng = np.random.default_rng(random_state)
    n = diff.shape[0]
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * diff[None, :]).mean(axis=1)
    p = float((np.abs(null) >= abs(observed)).mean())
    return observed, p, null
