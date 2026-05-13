"""Collection-aware spectral descriptors.

These descriptors operate on **pairs or collections** of shapes
rather than single shapes.  They quantify *how* a shape deforms
relative to a reference or within a cohort.

Implemented
-----------
- **Shape Difference Operator** — captures the spectral change
  between two shapes via functional maps.
- **DWKS** — Deformation Wave Kernel Signature (Magnet & Ovsjanikov,
  ICCV 2021): applies the WKS filter to shape-difference operators
  to produce a pointwise deformation descriptor.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    ScalarMap,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  FUNCTIONAL MAP ESTIMATION
# ======================================================================

def compute_functional_map(
    decomp_a: SpectralDecomposition,
    decomp_b: SpectralDecomposition,
    *,
    n_basis: int = 30,
    descriptor_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    regularize: float = 1e-3,
) -> np.ndarray:
    """Estimate a functional map C between two shapes.

    The functional map C : F(M_a) → F(M_b) is represented in the
    truncated eigenbasis as a (n_basis × n_basis) matrix satisfying:

        Φ_b^T · M_b · f_b ≈ C · (Φ_a^T · M_a · f_a)

    for corresponding functions f_a, f_b.

    Parameters
    ----------
    decomp_a, decomp_b : SpectralDecomposition
        Source and target spectral decompositions.
    n_basis : int
        Truncation size for the functional map.
    descriptor_pairs : list of (ndarray, ndarray), optional
        Pairs of corresponding descriptors (f_a, f_b) on the two
        shapes.  If ``None``, uses HKS at several time-scales as
        default correspondence signals.
    regularize : float
        Tikhonov regularisation weight.

    Returns
    -------
    C : ndarray, shape (n_basis, n_basis)
        Functional map matrix.
    """
    k_a = min(n_basis, decomp_a.n_eigenvalues)
    k_b = min(n_basis, decomp_b.n_eigenvalues)
    k = min(k_a, k_b)

    Phi_a = decomp_a.eigenvectors[:, :k]                   # (N_a, k)
    Phi_b = decomp_b.eigenvectors[:, :k]                   # (N_b, k)

    if descriptor_pairs is None:
        # Default: use HKS at 5 time-scales.
        from spectralbrain.spectral.descriptors import compute_hks
        hks_a = compute_hks(decomp_a, n_times=5)            # (N_a, 5)
        hks_b = compute_hks(decomp_b, n_times=5)            # (N_b, 5)
        descriptor_pairs = [
            (hks_a[:, t], hks_b[:, t]) for t in range(5)
        ]

    # Project descriptors onto eigenbases.
    # M_a-weighted projection: a_coeff = Φ_a^T · M_a · f_a
    M_a = decomp_a.mass
    M_b = decomp_b.mass

    A_coeffs = []  # coefficients on shape A
    B_coeffs = []  # coefficients on shape B

    for f_a, f_b in descriptor_pairs:
        if M_a is not None:
            a_c = Phi_a.T @ (M_a @ f_a)
        else:
            a_c = Phi_a.T @ f_a
        if M_b is not None:
            b_c = Phi_b.T @ (M_b @ f_b)
        else:
            b_c = Phi_b.T @ f_b
        A_coeffs.append(a_c)
        B_coeffs.append(b_c)

    A_mat = np.column_stack(A_coeffs)                       # (k, n_desc)
    B_mat = np.column_stack(B_coeffs)                       # (k, n_desc)

    # Solve: C · A_mat ≈ B_mat  →  C = B_mat · A_mat^+ (regularised)
    # Ridge: C = B_mat · A_mat^T · (A_mat · A_mat^T + λI)^{-1}
    AAt = A_mat @ A_mat.T + regularize * np.eye(k)
    C = B_mat @ A_mat.T @ np.linalg.inv(AAt)

    logger.debug("Functional map: %d × %d", C.shape[0], C.shape[1])
    return C


# ======================================================================
# §2  SHAPE DIFFERENCE OPERATORS
# ======================================================================

def shape_difference_operator(
    C: np.ndarray,
    *,
    type: Literal["area", "conformal"] = "area",
) -> np.ndarray:
    """Compute a shape-difference operator from a functional map.

    Parameters
    ----------
    C : ndarray, shape (k, k)
        Functional map from shape A to shape B.
    type : str
        ``"area"`` — D_area = C^T · C (captures area distortion).
        ``"conformal"`` — D_conf = C^T · Λ_B · C · Λ_A^{-1}
        (simplified: uses C^T · C − I to capture conformal distortion).

    Returns
    -------
    D : ndarray, shape (k, k)
        Shape-difference operator (symmetric positive semi-definite
        for area type).

    References
    ----------
    Rustamov RM, Ovsjanikov M, Azencot O, Ben-Chen M, Chazal F,
    Guibas LJ. Map-based exploration of intrinsic shape differences
    and variability. *ACM TOG* 32(4):72, 2013.
    """
    if type == "area":
        return C.T @ C
    elif type == "conformal":
        I = np.eye(C.shape[0])
        return C.T @ C - I
    else:
        raise ValueError(f"Unknown type: {type!r}")


# ======================================================================
# §3  DWKS — Deformation Wave Kernel Signature
# ======================================================================

def compute_dwks(
    decomp_source: SpectralDecomposition,
    decomp_target: SpectralDecomposition,
    *,
    n_basis: int = 30,
    n_energies: int = 50,
    sigma: Optional[float] = None,
    diff_type: Literal["area", "conformal"] = "area",
    descriptor_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> DescriptorMatrix:
    """Deformation Wave Kernel Signature.

    Applies the WKS band-pass filter to the eigenvalues of a
    shape-difference operator, producing a pointwise descriptor
    of **deformation** at each vertex.

    Unlike HKS/WKS which describe *geometry*, DWKS describes
    *how geometry changed* between two shapes.

    Parameters
    ----------
    decomp_source : SpectralDecomposition
        Source (reference) shape.
    decomp_target : SpectralDecomposition
        Target (deformed) shape.
    n_basis : int
        Functional map truncation.
    n_energies : int
        Number of WKS energy levels applied to the difference
        operator spectrum.
    sigma : float, optional
        WKS bandwidth.  ``None`` = auto.
    diff_type : str
        Shape-difference type (``"area"`` or ``"conformal"``).
    descriptor_pairs : list, optional
        Descriptor correspondences for functional map estimation.

    Returns
    -------
    ndarray, shape (N_source, n_energies)
        Per-vertex deformation descriptor on the source shape.

    References
    ----------
    Magnet R, Ovsjanikov M. DWKS: A Local Descriptor of
    Deformations Between Meshes and Point Clouds. *ICCV 2021*.
    """
    # Step 1: Functional map C: source → target.
    C = compute_functional_map(
        decomp_source, decomp_target,
        n_basis=n_basis,
        descriptor_pairs=descriptor_pairs,
    )

    # Step 2: Shape difference operator.
    D = shape_difference_operator(C, type=diff_type)

    # Step 3: Eigendecompose D.
    D_evals, D_evecs = np.linalg.eigh(D)                   # (k,), (k, k)
    D_evals = np.clip(D_evals, 1e-10, None)

    # Step 4: Apply WKS filter to D's eigenvalues.
    log_D_evals = np.log(D_evals)
    e_min = log_D_evals[0]
    e_max = log_D_evals[-1]

    if sigma is None:
        sigma = 7.0 * (e_max - e_min) / max(n_energies, 1)
        sigma = max(sigma, 1e-4)

    energies = np.linspace(e_min + 2 * sigma, e_max - 2 * sigma, n_energies)
    if len(energies) == 0:
        energies = np.linspace(e_min, e_max, n_energies)

    # WKS on D's spectrum: (n_energies, k)
    diff = energies[:, None] - log_D_evals[None, :]
    gauss = np.exp(-diff ** 2 / (2 * sigma ** 2))
    C_norm = gauss.sum(axis=1, keepdims=True)
    C_norm = np.clip(C_norm, 1e-30, None)
    gauss_norm = gauss / C_norm                             # (n_energies, k)

    # Step 5: Pull back to source shape via eigenvectors.
    # D_evecs live in the spectral basis; pull to spatial via Φ_source.
    k = min(n_basis, decomp_source.n_eigenvalues, D_evecs.shape[0])
    Phi_source = decomp_source.eigenvectors[:, :k]          # (N, k)
    D_evecs_trunc = D_evecs[:k, :]                          # (k, k)

    # DWKS(x, e) = Σ_j g_e(μ_j) · (Φ · ψ_j)²(x)
    # where μ_j are D's eigenvalues and ψ_j are D's eigenvectors.
    spatial_modes = Phi_source @ D_evecs_trunc               # (N, k)
    spatial_modes_sq = spatial_modes ** 2                     # (N, k)
    dwks = spatial_modes_sq @ gauss_norm.T                   # (N, n_energies)

    logger.info(
        "DWKS: N=%d, n_energies=%d, n_basis=%d, diff=%s",
        dwks.shape[0], n_energies, n_basis, diff_type,
    )
    return dwks


def compute_dwks_collection(
    reference: SpectralDecomposition,
    collection: Dict[str, SpectralDecomposition],
    *,
    n_basis: int = 30,
    n_energies: int = 50,
    diff_type: str = "area",
) -> Dict[str, DescriptorMatrix]:
    """Compute DWKS for each shape in a collection against a reference.

    Parameters
    ----------
    reference : SpectralDecomposition
        Template / mean shape.
    collection : dict of {name: SpectralDecomposition}
        Collection of shapes (e.g. subjects).
    n_basis, n_energies, diff_type : as in :func:`compute_dwks`.

    Returns
    -------
    dict of {name: ndarray}
        DWKS descriptor per shape.
    """
    results: Dict[str, DescriptorMatrix] = {}

    with progress_simple("DWKS collection", total=len(collection)) as tick:
        for name, decomp in collection.items():
            results[name] = compute_dwks(
                reference, decomp,
                n_basis=n_basis,
                n_energies=n_energies,
                diff_type=diff_type,
            )
            tick(1)

    logger.info(
        "DWKS collection: %d shapes vs reference", len(results),
    )
    return results


# ======================================================================

__all__: List[str] = [
    "compute_functional_map",
    "shape_difference_operator",
    "compute_dwks",
    "compute_dwks_collection",
]
