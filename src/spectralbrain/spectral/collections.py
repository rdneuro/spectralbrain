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

from typing import Literal

import numpy as np

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  FUNCTIONAL MAP ESTIMATION
# ======================================================================


def _default_descriptor_pairs(
    decomp_a: SpectralDecomposition,
    decomp_b: SpectralDecomposition,
    *,
    n_hks: int = 32,
    n_wks: int = 32,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """HKS + WKS correspondence signals evaluated on a **shared** grid.

    Both shapes are sampled at the same diffusion times / log-energies
    (pooled from both spectra) so that column ``j`` on shape A truly
    corresponds to column ``j`` on shape B.
    """
    from spectralbrain.spectral.descriptors import (
        _auto_wks_params,
        compute_hks,
        compute_wks,
        shared_hks_times,
    )

    t_shared = shared_hks_times([decomp_a, decomp_b], n_times=n_hks)
    hks_a = compute_hks(decomp_a, t_values=t_shared)
    hks_b = compute_hks(decomp_b, t_values=t_shared)
    pooled = np.concatenate([decomp_a.eigenvalues, decomp_b.eigenvalues])
    e_shared, sig = _auto_wks_params(np.sort(pooled), n_wks)
    wks_a = compute_wks(decomp_a, e_values=e_shared, sigma=sig, normalize=False)
    wks_b = compute_wks(decomp_b, e_values=e_shared, sigma=sig, normalize=False)
    pairs = [(hks_a[:, t], hks_b[:, t]) for t in range(hks_a.shape[1])]
    pairs += [(wks_a[:, e], wks_b[:, e]) for e in range(wks_a.shape[1])]
    return pairs


def compute_functional_map(
    decomp_a: SpectralDecomposition,
    decomp_b: SpectralDecomposition,
    *,
    n_basis: int = 30,
    descriptor_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    regularize: float = 1e-8,
    laplacian_weight: float = 1e-1,
) -> np.ndarray:
    """Estimate a functional map C between two shapes.

    The functional map C : F(M_a) → F(M_b) is represented in the
    truncated eigenbasis as a (n_basis × n_basis) matrix satisfying:

        Φ_b^T · M_b · f_b ≈ C · (Φ_a^T · M_a · f_a)

    for corresponding functions f_a, f_b.  It solves

        min_C ‖C A − B‖² + w_L ‖C Λ_a − Λ_b C‖² + ρ ‖C‖²

    where A, B stack the projected descriptors (each pair normalised to
    unit norm on the source side), Λ are the eigenvalues (normalised by
    the pooled spectral scale) and the second term is the standard
    Laplacian-commutativity regulariser (Ovsjanikov et al. 2012).  The
    problem decouples into one small linear system per row of C.

    Parameters
    ----------
    decomp_a, decomp_b : SpectralDecomposition
        Source and target spectral decompositions.
    n_basis : int
        Truncation size for the functional map.
    descriptor_pairs : list of (ndarray, ndarray), optional
        Pairs of corresponding descriptors (f_a, f_b) on the two
        shapes.  If ``None``, uses HKS (32 times) and WKS (32 energies)
        evaluated on time/energy grids **shared** by both shapes.
    regularize : float
        Tikhonov regularisation weight ρ (only for conditioning; the
        Laplacian term is the structural prior).  Larger values (the old
        default was 1e-3) shrink weakly-constrained rows of C towards 0.
    laplacian_weight : float
        Weight w_L of the Laplacian-commutativity term (0 disables it).

    Returns
    -------
    C : ndarray, shape (n_basis, n_basis)
        Functional map matrix.
    """
    k_a = min(n_basis, decomp_a.n_eigenvalues)
    k_b = min(n_basis, decomp_b.n_eigenvalues)
    k = min(k_a, k_b)

    Phi_a = decomp_a.eigenvectors[:, :k]  # (N_a, k)
    Phi_b = decomp_b.eigenvectors[:, :k]  # (N_b, k)

    if descriptor_pairs is None:
        descriptor_pairs = _default_descriptor_pairs(decomp_a, decomp_b)

    # Project descriptors onto eigenbases.
    # M_a-weighted projection: a_coeff = Φ_a^T · M_a · f_a
    M_a = decomp_a.mass
    M_b = decomp_b.mass
    if M_a is None or M_b is None:
        logger.warning(
            "compute_functional_map: mass matrix missing — projecting with "
            "the Euclidean inner product (only exact for orthonormal Φ)."
        )

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
        # Normalise each pair by the same scalar so that descriptors with
        # large dynamic range (e.g. small-t HKS) do not dominate.
        nrm = float(np.linalg.norm(a_c))
        if nrm > 1e-30:
            a_c = a_c / nrm
            b_c = b_c / nrm
        A_coeffs.append(a_c)
        B_coeffs.append(b_c)

    A_mat = np.column_stack(A_coeffs)  # (k, n_desc)
    B_mat = np.column_stack(B_coeffs)  # (k, n_desc)

    lam_a = np.asarray(decomp_a.eigenvalues[:k], dtype=np.float64)
    lam_b = np.asarray(decomp_b.eigenvalues[:k], dtype=np.float64)
    lam_scale = max(float(np.max(np.abs(np.concatenate([lam_a, lam_b])))), 1e-30)
    lam_a = lam_a / lam_scale
    lam_b = lam_b / lam_scale

    AAt = A_mat @ A_mat.T  # (k, k)
    AB = A_mat @ B_mat.T  # (k, k): column i = A · B_iᵀ
    C = np.zeros((k, k), dtype=np.float64)
    eye = np.eye(k)
    for i in range(k):
        # Row i: (AAᵀ + w_L·diag((λa − λb_i)²) + ρ I) c_i = A B_iᵀ
        lap = laplacian_weight * (lam_a - lam_b[i]) ** 2
        lhs = AAt + np.diag(lap) + regularize * eye
        C[i] = np.linalg.solve(lhs, AB[:, i])

    logger.debug("Functional map: %d × %d", C.shape[0], C.shape[1])
    return C


# ======================================================================
# §2  SHAPE DIFFERENCE OPERATORS
# ======================================================================


def shape_difference_operator(
    C: np.ndarray,
    *,
    type: Literal["area", "conformal"] = "area",
    evals_a: np.ndarray | None = None,
    evals_b: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a shape-difference operator from a functional map.

    Parameters
    ----------
    C : ndarray, shape (k, k)
        Functional map from shape A to shape B.
    type : str
        ``"area"`` — D_area = C^T · C (captures area distortion).
        ``"conformal"`` — D_conf = C^T · Λ_B · C · Λ_A^{+}
        (Rustamov et al. 2013; Λ_A^{+} is the pseudo-inverse, i.e. the
        null/constant mode is excluded).  Requires *evals_a*/*evals_b*.
    evals_a, evals_b : ndarray, shape (≥k,), optional
        Eigenvalues of shapes A and B (needed for ``"conformal"``).

    Returns
    -------
    D : ndarray, shape (k, k)
        Shape-difference operator (symmetric positive semi-definite
        for area type; similar to a symmetric PSD matrix for conformal).

    References
    ----------
    Rustamov RM, Ovsjanikov M, Azencot O, Ben-Chen M, Chazal F,
    Guibas LJ. Map-based exploration of intrinsic shape differences
    and variability. *ACM TOG* 32(4):72, 2013.
    """
    if type == "area":
        return C.T @ C
    elif type == "conformal":
        if evals_a is None or evals_b is None:
            raise ValueError(
                "type='conformal' needs the eigenvalues: pass evals_a=... and "
                "evals_b=... (D_conf = Cᵀ Λ_B C Λ_A⁺)."
            )
        k = C.shape[0]
        lam_a = np.asarray(evals_a, dtype=np.float64)[:k]
        lam_b = np.asarray(evals_b, dtype=np.float64)[: C.shape[0]]
        inv_a = np.zeros_like(lam_a)
        nz = lam_a > 1e-10
        inv_a[nz] = 1.0 / lam_a[nz]
        return C.T @ np.diag(lam_b) @ C @ np.diag(inv_a)
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
    sigma: float | None = None,
    diff_type: Literal["area", "conformal"] = "area",
    descriptor_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
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
        decomp_source,
        decomp_target,
        n_basis=n_basis,
        descriptor_pairs=descriptor_pairs,
    )

    # Step 2–3: Shape difference operator and its eigendecomposition
    # (as functions on the source shape, coefficients in Φ_source).
    k_c = C.shape[0]
    if diff_type == "area":
        D = shape_difference_operator(C, type="area")
        D_evals, D_evecs = np.linalg.eigh(D)  # (k,), (k, k)
    elif diff_type == "conformal":
        # D_conf = Cᵀ Λ_B C Λ_A⁺ is similar to the symmetric PSD matrix
        # S = Λ_A^{+½} Cᵀ Λ_B C Λ_A^{+½}; its eigenfunctions on A (the
        # eigenvectors of Λ_A⁺ Cᵀ Λ_B C) are Λ_A^{+½} u.  The null
        # (constant) mode carries no conformal information and is dropped.
        lam_a = np.asarray(decomp_source.eigenvalues[:k_c], dtype=np.float64)
        lam_b = np.asarray(decomp_target.eigenvalues[:k_c], dtype=np.float64)
        nz = lam_a > 1e-10
        isq = np.zeros_like(lam_a)
        isq[nz] = 1.0 / np.sqrt(lam_a[nz])
        S = (isq[:, None] * (C.T @ np.diag(lam_b) @ C)) * isq[None, :]
        S = 0.5 * (S + S.T)
        S_nz = S[np.ix_(nz, nz)]
        D_evals, U = np.linalg.eigh(S_nz)
        D_evecs = np.zeros((k_c, U.shape[1]), dtype=np.float64)
        D_evecs[nz] = isq[nz][:, None] * U
        D_evecs /= np.clip(np.linalg.norm(D_evecs, axis=0, keepdims=True), 1e-30, None)
    else:
        raise ValueError(f"Unknown diff_type: {diff_type!r}")
    D_evals = np.clip(D_evals, 1e-10, None)

    # Step 4: Apply WKS filter to D's eigenvalues.
    log_D_evals = np.log(D_evals)
    e_min = log_D_evals[0]
    e_max = log_D_evals[-1]

    if sigma is None:
        sigma = 7.0 * (e_max - e_min) / max(n_energies, 1)
        sigma = max(sigma, 1e-4)

    e_lo, e_hi = e_min + 2 * sigma, e_max - 2 * sigma
    if e_lo >= e_hi:
        # The 2σ inward shift crosses over (few energies / narrow
        # spectrum): use the full range with a matching bandwidth, as in
        # the WKS auto-parameters, instead of a reversed grid.
        e_lo, e_hi = e_min, e_max
        sigma = max((e_max - e_min) / (2 * max(n_energies, 1)), 1e-4)
    energies = np.linspace(e_lo, e_hi, n_energies)

    # WKS on D's spectrum: (n_energies, k)
    diff = energies[:, None] - log_D_evals[None, :]
    gauss = np.exp(-(diff**2) / (2 * sigma**2))
    C_norm = gauss.sum(axis=1, keepdims=True)
    C_norm = np.clip(C_norm, 1e-30, None)
    gauss_norm = gauss / C_norm  # (n_energies, k)

    # Step 5: Pull back to source shape via eigenvectors.
    # D_evecs live in the spectral basis; pull to spatial via Φ_source.
    k = min(n_basis, decomp_source.n_eigenvalues, D_evecs.shape[0])
    Phi_source = decomp_source.eigenvectors[:, :k]  # (N, k)
    D_evecs_trunc = D_evecs[:k, :]  # (k, k)

    # DWKS(x, e) = Σ_j g_e(μ_j) · (Φ · ψ_j)²(x)
    # where μ_j are D's eigenvalues and ψ_j are D's eigenvectors.
    spatial_modes = Phi_source @ D_evecs_trunc  # (N, k)
    spatial_modes_sq = spatial_modes**2  # (N, k)
    dwks = spatial_modes_sq @ gauss_norm.T  # (N, n_energies)

    logger.info(
        "DWKS: N=%d, n_energies=%d, n_basis=%d, diff=%s",
        dwks.shape[0],
        n_energies,
        n_basis,
        diff_type,
    )
    return dwks


def compute_dwks_collection(
    reference: SpectralDecomposition,
    collection: dict[str, SpectralDecomposition],
    *,
    n_basis: int = 30,
    n_energies: int = 50,
    diff_type: str = "area",
) -> dict[str, DescriptorMatrix]:
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
    results: dict[str, DescriptorMatrix] = {}

    with progress_simple("DWKS collection", total=len(collection)) as tick:
        for name, decomp in collection.items():
            results[name] = compute_dwks(
                reference,
                decomp,
                n_basis=n_basis,
                n_energies=n_energies,
                diff_type=diff_type,
            )
            tick(1)

    logger.info(
        "DWKS collection: %d shapes vs reference",
        len(results),
    )
    return results


# ======================================================================

__all__: list[str] = [
    "compute_dwks",
    "compute_dwks_collection",
    "compute_functional_map",
    "shape_difference_operator",
]
