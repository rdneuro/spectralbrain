"""Biharmonic and polyharmonic spectral descriptors (eigenpair reuse).

The biharmonic operator ``Δ²`` and the wider polyharmonic family ``Δᵐ``
share the **same eigenfunctions** as the Laplace–Beltrami operator; only
the eigenvalues are transformed (``λ → λᵐ``).  This makes the whole family
a *zero-eigensolve* extension: given one LBO
:class:`~spectralbrain.core.base.SpectralDecomposition`, every biharmonic
and polyharmonic descriptor below is obtained by re-weighting the existing
spectrum — no new operator assembly, no new eigensolve.

Why biharmonic descriptors
--------------------------
The biharmonic operator down-weights high frequencies as ``1/λ²`` instead
of the heat kernel's ``e^{-tλ}``, yielding descriptors that are smoother,
globally aware, and far less sensitive to local meshing noise than HKS —
a desirable property for noisy cortical/subcortical surfaces.  The
biharmonic distance (Lipman et al. 2010) is already provided by
:mod:`spectralbrain.spectral.distances`; here we add the **multiscale
biharmonic kernel** (Rustamov et al. 2011) and the per-vertex **biharmonic
kernel signature**, plus the general polyharmonic spectrum.

Provided
--------
- :func:`polyharmonic_decompose` — reuse LBO eigenpairs, return a
  ``Δᵐ`` decomposition (``λ → λᵐ``).
- :func:`biharmonic_kernel_signature` — per-vertex multiscale signature
  ``BKS_t(x) = Σ_{k≥1} e^{-tλ_k}/λ_k^order · φ_k(x)²``.
- :func:`multiscale_biharmonic_kernel` — the kernel diagonal or full
  ``(N, N)`` matrix at a given scale.
- :func:`compute_bhks` — convenience wrapper from ``(vertices, faces)``.

References
----------
Lipman Y, Rustamov RM, Funkhouser TA. "Biharmonic distance."
*ACM TOG* 29(3):27, 2010.
Rustamov RM, Lipman Y, Funkhouser T. "Multiscale biharmonic kernels."
*Computer Graphics Forum* 30(5):1521–1531, 2011.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    Faces,
    ScalarMap,
    Vertices,
    get_logger,
)

from spectralbrain.expanded._base import BackendSpec

logger = get_logger(__name__)

_EPS = 1e-12


# ======================================================================
# §1  HELPERS
# ======================================================================


def _decomp_from_input(
    obj: SpectralDecomposition | tuple,
    faces: Faces | None,
    *,
    k: int,
    backend: BackendSpec | Any,
) -> SpectralDecomposition:
    """Accept a decomposition or raw ``(vertices, faces)`` and return one.

    The reuse path: if ``obj`` is already a SpectralDecomposition, it is
    returned unchanged (no eigensolve).  Otherwise a cotangent LBO
    decomposition is computed once.
    """
    if isinstance(obj, SpectralDecomposition):
        return obj
    if faces is None:
        raise ValueError(
            "Pass either a SpectralDecomposition, or (vertices, faces)."
        )
    from spectralbrain.core.meshes import BrainMesh
    from spectralbrain.expanded._base import resolve_backend

    mesh = BrainMesh(np.asarray(obj), np.asarray(faces))
    return mesh.decompose(k=k, backend=resolve_backend(backend))


def _nonzero_spectrum(
    decomp: SpectralDecomposition,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) with the null mode(s) removed.

    Modes with ``λ ≈ 0`` (the constant mode and any disconnected-component
    nulls) are dropped because biharmonic weights involve ``1/λ``.
    """
    evals = np.asarray(decomp.eigenvalues, dtype=np.float64)
    evecs = np.asarray(decomp.eigenvectors, dtype=np.float64)
    keep = evals > _EPS
    return evals[keep], evecs[:, keep]


# ======================================================================
# §2  POLYHARMONIC SPECTRUM (Δᵐ) — eigenpair reuse
# ======================================================================


def polyharmonic_decompose(
    obj: SpectralDecomposition | Vertices,
    faces: Faces | None = None,
    *,
    order: int = 2,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Return a ``Δᵐ`` decomposition by reusing LBO eigenpairs.

    Since ``Δᵐ φ = λᵐ φ`` shares the LBO eigenfunctions, this only raises
    the eigenvalues to the power ``order`` (``order=2`` → biharmonic).

    Parameters
    ----------
    obj : SpectralDecomposition or ndarray (N, 3)
        An existing LBO decomposition (reused, no eigensolve), or vertices.
    faces : ndarray (F, 3), optional
        Required only if ``obj`` is a vertex array.
    order : int
        Polyharmonic order ``m ≥ 1`` (1 = LBO, 2 = biharmonic, …).
    k : int
        Eigenpairs to compute if ``obj`` is raw geometry.
    backend : str or backend object

    Returns
    -------
    SpectralDecomposition
        Same eigenvectors; eigenvalues ``λᵐ``; ``metadata["operator"] ==
        "polyharmonic"``.
    """
    if order < 1:
        raise ValueError(f"order must be ≥ 1, got {order}.")
    decomp = _decomp_from_input(obj, faces, k=k, backend=backend)
    return SpectralDecomposition(
        eigenvalues=np.asarray(decomp.eigenvalues, dtype=np.float64) ** order,
        eigenvectors=decomp.eigenvectors,
        stiffness=decomp.stiffness,
        mass=decomp.mass,
        surface_area=decomp.surface_area,
        metadata={
            **decomp.metadata,
            "operator": "polyharmonic",
            "polyharmonic_order": int(order),
        },
    )


# ======================================================================
# §3  MULTISCALE BIHARMONIC KERNEL
# ======================================================================


def _auto_biharmonic_times(evals: np.ndarray, n_times: int) -> np.ndarray:
    """Logarithmically spaced time scales from the non-zero spectrum."""
    nz = evals[evals > _EPS]
    if nz.size == 0:
        return np.logspace(-2, 2, n_times)
    t_min = 4.0 * np.log(10.0) / nz.max()
    t_max = 4.0 * np.log(10.0) / nz.min()
    return np.logspace(np.log10(t_min), np.log10(t_max), n_times)


def biharmonic_kernel_signature(
    obj: SpectralDecomposition | Vertices,
    faces: Faces | None = None,
    *,
    t_values: np.ndarray | None = None,
    n_times: int = 100,
    order: int = 2,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
    normalize: bool = False,
) -> DescriptorMatrix:
    """Per-vertex multiscale biharmonic kernel signature.

    Defines, for each vertex ``x`` and scale ``t``,

        BKS_t(x) = Σ_{k≥1} e^{-t λ_k} / λ_k^order · φ_k(x)²,

    interpolating between the heat kernel signature (``order=0``) and a
    biharmonic-weighted, noise-robust signature (``order=2``).  Always
    non-negative (sum of positively weighted squares).

    Parameters
    ----------
    obj : SpectralDecomposition or ndarray (N, 3)
        Reused LBO eigenpairs (no eigensolve) or vertices.
    faces : ndarray (F, 3), optional
    t_values : ndarray (T,), optional
        Explicit time scales; auto-chosen from the spectrum if ``None``.
    n_times : int
        Number of auto scales.
    order : int
        Spectral down-weighting exponent (``2`` = biharmonic).
    k : int
        Eigenpairs if ``obj`` is raw geometry.
    backend : str or backend object
    normalize : bool
        If ``True``, divide each vertex's curve by its scale-0 value
        (per-vertex L1-style normalisation across scales).

    Returns
    -------
    ndarray, shape (N, T)
    """
    decomp = _decomp_from_input(obj, faces, k=k, backend=backend)
    evals, evecs = _nonzero_spectrum(decomp)  # (k',), (N, k')

    if t_values is None:
        t_values = _auto_biharmonic_times(evals, n_times)
    t_values = np.asarray(t_values, dtype=np.float64)  # (T,)

    weight = np.power(evals, -order)  # (k',)  spectral down-weighting
    decay = np.exp(-evals[None, :] * t_values[:, None])  # (T, k')
    weighted = decay * weight[None, :]  # (T, k')
    phi_sq = evecs**2  # (N, k')
    bks = phi_sq @ weighted.T  # (N, T)
    bks = np.clip(bks, 0.0, None)

    if normalize:
        ref = bks[:, :1]
        bks = bks / np.clip(ref, _EPS, None)
    return bks


def multiscale_biharmonic_kernel(
    obj: SpectralDecomposition | Vertices,
    faces: Faces | None = None,
    *,
    t: float = 0.0,
    order: int = 2,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
    full: bool = False,
) -> np.ndarray:
    """Multiscale biharmonic kernel at a single scale ``t``.

    With ``K_t(x, y) = Σ_{k≥1} e^{-t λ_k} / λ_k^order · φ_k(x) φ_k(y)``:

    - ``full=False`` → the diagonal ``K_t(x, x)`` (per-vertex, shape (N,));
    - ``full=True``  → the dense kernel matrix (shape (N, N)).

    At ``t = 0`` and ``order = 2`` this is the Green's function of the
    biharmonic operator, whose induced distance equals the biharmonic
    distance of Lipman et al. (2010).

    Parameters
    ----------
    obj : SpectralDecomposition or ndarray (N, 3)
    faces : ndarray (F, 3), optional
    t : float
        Scale parameter (``0`` = pure biharmonic Green's function).
    order : int
    k : int
    backend : str or backend object
    full : bool
        Return the dense matrix instead of the diagonal.

    Returns
    -------
    ndarray
        Shape ``(N,)`` if ``full`` is ``False`` else ``(N, N)``.

    Warnings
    --------
    The dense kernel is ``O(N²)`` memory; use ``full=True`` only on small
    (sub-cortical) meshes.
    """
    decomp = _decomp_from_input(obj, faces, k=k, backend=backend)
    evals, evecs = _nonzero_spectrum(decomp)
    weight = np.exp(-t * evals) * np.power(evals, -order)  # (k',)

    if full:
        n = evecs.shape[0]
        if n > 20000:
            logger.warning(
                "multiscale_biharmonic_kernel(full=True): N=%d → %.1f GB "
                "dense matrix.  Consider full=False.",
                n,
                n * n * 8 / 1e9,
            )
        return (evecs * weight[None, :]) @ evecs.T  # (N, N)

    return np.sum((evecs**2) * weight[None, :], axis=1)  # (N,)


def compute_bhks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 100,
    n_times: int = 100,
    order: int = 2,
    backend: BackendSpec | Any = "auto",
) -> DescriptorMatrix:
    """Convenience: biharmonic kernel signature from ``(vertices, faces)``.

    Equivalent to computing an LBO decomposition and calling
    :func:`biharmonic_kernel_signature`.

    Returns
    -------
    ndarray, shape (N, n_times)
    """
    return biharmonic_kernel_signature(
        vertices,
        faces,
        n_times=n_times,
        order=order,
        k=k,
        backend=backend,
    )


__all__ = [
    "biharmonic_kernel_signature",
    "compute_bhks",
    "multiscale_biharmonic_kernel",
    "polyharmonic_decompose",
]
