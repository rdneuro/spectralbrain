"""Spatial null models that preserve spatial autocorrelation.

For surfaces **with a boundary** (e.g. the unfolded hippocampal sheet) the spin
test is invalid because spherical projection distorts inter-vertex distances and
inflates false positives (Bazinet, Liu & Misic 2025). brainmosaic therefore uses
geometry-aware nulls:

- :func:`eigenstrapping_surrogates` -- self-contained random rotation of
  Laplace-Beltrami geometric eigenmodes (Koussis, Pang et al. 2025). Reuses
  brainmosaic's own LBO eigenpairs, so it works natively on bounded surfaces.
- :func:`brainsmash_surrogates` -- variogram-matched surrogates from a geodesic
  distance matrix (Burt et al. 2020), via the optional ``brainsmash`` package.
- :func:`paired_label_permutation` -- within-subject ipsi/contra label swaps for
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

import numpy as np

logger = logging.getLogger("spectralbrain.statistics.clustering._core")


def _harmonic_groups(K: int) -> list:
    """Koussis-style eigenmode blocks: mode 0 alone, then sizes 3, 5, 7, ...

    These mirror the spherical-harmonic multiplets (2l+1 modes per degree l)
    whose rotation preserves the power spectrum on a sphere; on general
    surfaces they group modes of similar wavelength (Koussis et al. 2025).
    """
    groups = [[0]] if K > 0 else []
    start, size = 1, 3
    while start < K:
        stop = min(start + size, K)
        groups.append(list(range(start, stop)))
        start, size = stop, size + 2
    return groups


def _degenerate_groups(evals: np.ndarray, tol: float) -> list:
    """Group consecutive modes whose eigenvalues agree within relative ``tol``."""
    K = evals.size
    groups = []
    start = 0
    for k in range(1, K + 1):
        if k == K or abs(evals[k] - evals[start]) > tol * (abs(evals[start]) + 1e-12):
            groups.append(list(range(start, k)))
            start = k
    return groups


def eigenstrapping_surrogates(
    data: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    mass: np.ndarray,
    n_surrogates: int = 1000,
    eigenvalue_tol: float = 1e-3,
    random_state: int = 0,
    grouping: str = "harmonic",
    residual: str = "permute",
) -> np.ndarray:
    """Generate SA-preserving surrogates by rotating LBO geometric eigenmodes.

    The map is expanded in the (M-orthonormal) LBO eigenbasis; eigenmodes are
    grouped and a random orthogonal rotation is applied within each group;
    the surrogate is reconstructed from the rotated coefficients. Rotations
    within a group preserve the group's power (hence, approximately, the
    spatial autocorrelation) while randomising phase.

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
        Relative tolerance for grouping near-degenerate eigenvalues
        (``grouping="degenerate"`` only).
    random_state : int
    grouping : {"harmonic", "degenerate"}
        ``"harmonic"`` (default): Koussis et al. blocks of sizes 1, 3, 5, ...
        ``"degenerate"``: group modes with near-equal eigenvalues. On real,
        irregular surfaces eigenvalues are almost never degenerate, so this
        mode usually leaves the map (nearly) unchanged and a warning is
        emitted when no group has more than one mode.
    residual : {"permute", "add", "none"}
        How to treat the part of the map not captured by the K modes:
        ``"permute"`` (default) adds a random permutation of the residual so
        surrogates match the map's total variance; ``"add"`` adds the
        observed residual unchanged; ``"none"`` returns the low-pass
        reconstruction only.

    Returns
    -------
    np.ndarray, shape (n_surrogates, V)
        Surrogate maps with matched spatial autocorrelation.

    Notes
    -----
    The constant mode (mode 0) is kept fixed so surrogates share the map's
    mean.
    """
    import warnings

    data = np.asarray(data, float).ravel()
    evals = np.asarray(eigenvalues, float).ravel()
    evecs = np.asarray(eigenvectors, float)
    m = np.asarray(mass, float).ravel()
    V, K = evecs.shape
    if data.shape[0] != V:
        raise ValueError(f"data length {data.shape[0]} != #vertices {V}.")

    # Coefficients via the M-inner product: c_k = <phi_k, data>_M.
    coeffs = evecs.T @ (m * data)  # (K,)
    resid = data - evecs @ coeffs

    if grouping == "harmonic":
        if eigenvalue_tol != 1e-3:
            warnings.warn("eigenvalue_tol is only used with grouping='degenerate'.", stacklevel=2)
        groups = _harmonic_groups(K)
    elif grouping == "degenerate":
        groups = _degenerate_groups(evals, eigenvalue_tol)
        if all(len(g) == 1 for g in groups):
            warnings.warn(
                "No near-degenerate eigenvalue groups found: degenerate-grouping "
                "eigenstrapping would return copies of the map. Use "
                "grouping='harmonic'.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        raise ValueError("grouping must be 'harmonic' or 'degenerate'.")
    if residual not in ("permute", "add", "none"):
        raise ValueError("residual must be 'permute', 'add' or 'none'.")

    rng = np.random.default_rng(random_state)
    surrogates = np.empty((n_surrogates, V), dtype=np.float64)
    for s in range(n_surrogates):
        rot_coeffs = coeffs.copy()
        for grp in groups:
            if len(grp) == 1:
                continue  # constant mode / singleton: keep fixed
            g = np.asarray(grp)
            # Haar-random orthogonal matrix via QR of a Gaussian.
            A = rng.standard_normal((g.size, g.size))
            Q, R = np.linalg.qr(A)
            Q *= np.sign(np.diag(R))  # fix QR sign ambiguity
            rot_coeffs[g] = Q @ coeffs[g]
        surr = evecs @ rot_coeffs
        if residual == "permute":
            surr = surr + resid[rng.permutation(V)]
        elif residual == "add":
            surr = surr + resid
        surrogates[s] = surr
    return surrogates


def brainsmash_surrogates(
    data: np.ndarray, distance: np.ndarray, n_surrogates: int = 1000, **kwargs
) -> np.ndarray:
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


def paired_label_permutation(
    values_ipsi: np.ndarray, values_contra: np.ndarray, n_perm: int = 5000, random_state: int = 0
):
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
    # Add-one correction: the observed labelling is itself a member of the
    # permutation distribution, so p can never be exactly zero.
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return observed, p, null
