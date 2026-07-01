"""Anisotropic (Finsler) Laplacians steered by principal curvature.

The isotropic Laplace–Beltrami operator diffuses equally in every tangent
direction.  An **anisotropic** Laplacian replaces the scalar conductivity
with a per-face symmetric tensor ``G`` that diffuses preferentially along a
chosen direction — here the principal curvature axes estimated by the
Rusinkiewicz method (:func:`spectralbrain.expanded._base.principal_curvatures`),
which are considerably more reliable than the 1-ring PCA proxy used by the
simplified :func:`spectralbrain.spectral.anisotropic.anisotropic_laplacian`.

Discretisation
--------------
Generalising the cotangent identity ``L_f[a][b] = (e_a · e_b)/(4 A_f)`` to a
conductivity tensor ``G_f`` gives the **Finsler-style** anisotropic
stiffness

    L_f[a][b] = (e_aᵀ G_f e_b) / (4 A_f),
    G_f = I + (g − 1) · d_f ⊗ d_f,

where ``d_f`` is the in-plane unit principal direction and ``g`` the
anisotropy ratio (conductivity along ``d_f`` relative to the perpendicular).
``g = 1`` recovers the isotropic cotangent Laplacian **exactly**; ``g > 1``
diffuses faster along ``d_f``; ``0 < g < 1`` slows it.  The construction is
symmetric and has zero row sums by ``Σ_b e_b = 0``.

Provided
--------
- :func:`finsler_laplacian` — assemble the anisotropic ``(L, M)``.
- :func:`finsler_decompose` — assemble + eigensolve → decomposition.
- :func:`anisotropic_bank_descriptor` — a filter bank sweeping anisotropy
  ratio × orientation, concatenating an HKS/WKS per setting.

References
----------
Andreux M, Rodolà E, Aubry M, Cremers D. "Anisotropic Laplace–Beltrami
operators for shape analysis." *ECCV NORDIA workshop*, 2014.
Jadhav S, Cremers D. "Finsler-Laplace–Beltrami operators for shape
analysis." *CVPR*, 2024.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    Faces,
    MassMatrix,
    SparseMatrix,
    Vertices,
    get_logger,
    progress_simple,
)

from spectralbrain.expanded._base import (
    BackendSpec,
    _validate_mesh,
    face_areas,
    operator_eigensystem,
    principal_curvatures,
)

logger = get_logger(__name__)

DirectionSpec = Literal["max_curvature", "min_curvature"]
"""Which principal direction the conductivity tensor aligns with."""

_EPS = 1e-12


# ======================================================================
# §1  FACE-WISE PRINCIPAL DIRECTION
# ======================================================================


def _face_principal_direction(
    vertices: np.ndarray,
    faces: np.ndarray,
    which: DirectionSpec,
) -> np.ndarray:
    """In-plane unit principal direction per face (sign-consistent average).

    Principal directions are line fields (defined mod π), so the per-vertex
    directions are sign-aligned within each face before averaging, then
    projected onto the face plane and renormalised.
    """
    k1, k2, dir1, dir2 = principal_curvatures(vertices, faces)
    dvert = dir1 if which == "max_curvature" else dir2  # (N, 3)

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    d0, d1, d2 = dvert[i0], dvert[i1], dvert[i2]  # (F, 3) each
    # align d1, d2 to d0's sign (line-field ambiguity)
    d1 = d1 * np.sign(np.sum(d1 * d0, axis=1, keepdims=True) + _EPS)
    d2 = d2 * np.sign(np.sum(d2 * d0, axis=1, keepdims=True) + _EPS)
    df = d0 + d1 + d2  # (F, 3)

    # project onto the face plane and normalise
    p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
    nf = np.cross(p1 - p0, p2 - p0)
    nf /= np.clip(np.linalg.norm(nf, axis=1, keepdims=True), _EPS, None)
    df = df - nf * np.sum(df * nf, axis=1, keepdims=True)
    norm = np.linalg.norm(df, axis=1, keepdims=True)
    # degenerate (umbilic) faces → fall back to an arbitrary in-plane axis
    degen = norm[:, 0] < _EPS
    fallback = p1 - p0
    fallback /= np.clip(np.linalg.norm(fallback, axis=1, keepdims=True), _EPS, None)
    df[degen] = fallback[degen]
    norm = np.linalg.norm(df, axis=1, keepdims=True)
    return df / np.clip(norm, _EPS, None)


# ======================================================================
# §2  FINSLER / ANISOTROPIC LAPLACIAN
# ======================================================================


def finsler_laplacian(
    vertices: Vertices,
    faces: Faces,
    *,
    anisotropy_ratio: float = 3.0,
    direction: DirectionSpec = "min_curvature",
    custom_directions: np.ndarray | None = None,
) -> tuple[SparseMatrix, MassMatrix]:
    """Curvature-aligned anisotropic cotangent Laplacian.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    anisotropy_ratio : float
        Conductivity ``g`` along the chosen direction relative to the
        perpendicular.  ``g = 1`` → isotropic cotangent Laplacian; ``g > 1``
        → faster diffusion along ``direction``.  Must be > 0.
    direction : str
        ``"max_curvature"`` (align with κ₁ axis) or ``"min_curvature"``
        (align with κ₂ axis, i.e. along ridges/valleys).
    custom_directions : ndarray, shape (N, 3), optional
        Per-vertex preferred directions overriding the curvature axes
        (averaged onto faces, projected to the tangent plane).

    Returns
    -------
    L : sparse matrix, shape (N, N)
        Anisotropic stiffness (CSC), symmetric, zero row sums.
    M : sparse matrix, shape (N, N)
        Lumped mass (identical to the isotropic cotangent mass).
    """
    if anisotropy_ratio <= 0:
        raise ValueError(f"anisotropy_ratio must be > 0, got {anisotropy_ratio}.")
    v, f = _validate_mesh(vertices, faces)
    n = v.shape[0]
    g = float(anisotropy_ratio)

    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    e = {0: v[i2] - v[i1], 1: v[i0] - v[i2], 2: v[i1] - v[i0]}  # opposite edges
    fa = face_areas(v, f)
    inv4a = 1.0 / np.clip(4.0 * fa, _EPS, None)  # (F,)

    if custom_directions is not None:
        cd = np.asarray(custom_directions, dtype=np.float64)
        if cd.shape != (n, 3):
            raise ValueError(f"custom_directions must be (N, 3), got {cd.shape}.")
        d0, d1, d2 = cd[i0], cd[i1], cd[i2]
        d1 = d1 * np.sign(np.sum(d1 * d0, axis=1, keepdims=True) + _EPS)
        d2 = d2 * np.sign(np.sum(d2 * d0, axis=1, keepdims=True) + _EPS)
        df = d0 + d1 + d2
        df /= np.clip(np.linalg.norm(df, axis=1, keepdims=True), _EPS, None)
    else:
        df = _face_principal_direction(v, f, direction)  # (F, 3)

    idx = {0: i0, 1: i1, 2: i2}
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for a in range(3):
        ea = e[a]
        ea_d = np.sum(ea * df, axis=1)  # (F,)
        for b in range(3):
            eb = e[b]
            # e_aᵀ G e_b = e_a·e_b + (g-1)(e_a·d)(e_b·d)
            eb_d = np.sum(eb * df, axis=1)
            wab = (np.sum(ea * eb, axis=1) + (g - 1.0) * ea_d * eb_d) * inv4a
            rows.append(idx[a])
            cols.append(idx[b])
            vals.append(wab)

    L = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsc()
    L = 0.5 * (L + L.T)  # symmetric by construction; scrub fp drift

    point_area = np.zeros(n, dtype=np.float64)
    for c in range(3):
        np.add.at(point_area, f[:, c], fa / 3.0)
    M = sp.diags(np.clip(point_area, _EPS, None), format="csc")
    logger.info(
        "Finsler Laplacian: g=%.2f direction=%s nnz=%d", g, direction, L.nnz
    )
    return L, M


def finsler_decompose(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 100,
    anisotropy_ratio: float = 3.0,
    direction: DirectionSpec = "min_curvature",
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Assemble and eigendecompose the Finsler/anisotropic Laplacian.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Number of eigenpairs.
    anisotropy_ratio : float
        Conductivity ratio ``g`` (see :func:`finsler_laplacian`).
    direction : str
        Principal-direction alignment.
    backend : str or backend object
        Multi-backend selector.

    Returns
    -------
    SpectralDecomposition
        With ``metadata["operator"] == "finsler"``.
    """
    L, M = finsler_laplacian(
        vertices,
        faces,
        anisotropy_ratio=anisotropy_ratio,
        direction=direction,
    )
    sa = float(face_areas(*_validate_mesh(vertices, faces)).sum())
    return operator_eigensystem(
        L,
        M,
        k=k,
        backend=backend,
        sigma=-0.01,
        which="LM",
        clamp_nonneg=True,
        surface_area=sa,
        operator="finsler",
        metadata={"anisotropy_ratio": float(anisotropy_ratio), "direction": direction},
    )


# ======================================================================
# §3  ANISOTROPY FILTER BANK
# ======================================================================


def anisotropic_bank_descriptor(
    vertices: Vertices,
    faces: Faces,
    *,
    ratios: tuple[float, ...] = (1.0, 3.0, 6.0),
    directions: tuple[DirectionSpec, ...] = ("min_curvature", "max_curvature"),
    descriptor: Literal["hks", "wks"] = "hks",
    k: int = 80,
    n_components: int = 16,
    backend: BackendSpec | Any = "auto",
) -> DescriptorMatrix:
    """Concatenate HKS/WKS over a bank of anisotropy ratios × orientations.

    Each (ratio, direction) pair yields one Finsler Laplacian whose spectrum
    feeds the chosen descriptor; the per-setting descriptors are stacked
    column-wise into a single multi-orientation, multi-scale feature.

    Parameters
    ----------
    vertices, faces : arrays
    ratios : tuple of float
        Anisotropy ratios to sweep (include ``1.0`` for the isotropic band).
    directions : tuple of str
        Principal-direction alignments to sweep.
    descriptor : str
        ``"hks"`` or ``"wks"``.
    k : int
        Eigenpairs per setting.
    n_components : int
        Time scales (HKS) or energies (WKS) per setting.
    backend : str or backend object
        Multi-backend selector (threaded into each eigensolve).

    Returns
    -------
    ndarray, shape (N, len(ratios) × len(directions) × n_components)
    """
    from spectralbrain.spectral.descriptors import compute_hks, compute_wks

    v, f = _validate_mesh(vertices, faces)
    blocks: list[np.ndarray] = []
    total = len(ratios) * len(directions)

    with progress_simple("Anisotropy bank", total=total) as tick:
        for direction in directions:
            for g in ratios:
                decomp = finsler_decompose(
                    v,
                    f,
                    k=k,
                    anisotropy_ratio=g,
                    direction=direction,
                    backend=backend,
                )
                if descriptor == "hks":
                    blocks.append(compute_hks(decomp, n_times=n_components))
                elif descriptor == "wks":
                    blocks.append(compute_wks(decomp, n_energies=n_components))
                else:
                    raise ValueError(
                        f"descriptor must be 'hks' or 'wks', got {descriptor!r}."
                    )
                tick(1)

    return np.hstack(blocks)  # (N, |ratios|·|dirs|·n_components)


__all__ = [
    "DirectionSpec",
    "anisotropic_bank_descriptor",
    "finsler_decompose",
    "finsler_laplacian",
]
