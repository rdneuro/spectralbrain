"""Collection-aware spectral descriptors.

These descriptors operate on **pairs or collections** of shapes
rather than single shapes.  They quantify *how* a shape deforms
relative to a reference or within a cohort.

Implemented
-----------
- **Shape Difference Operator** -- captures the spectral change
  between two shapes via functional maps.
- **DWKS** -- Deformation Wave Kernel Signature (Magnet & Ovsjanikov,
  ICCV 2021): applies the WKS filter to shape-difference operators
  to produce a pointwise deformation descriptor.
- **Spectral deformation** -- correspondence-free, registration-free
  per-vertex scale function that aligns the LBO spectrum of one mesh to
  another's (Hu et al., IEEE TVCG 2017), with a left-vs-right
  lateralisation wrapper (:func:`spectral_deformation`,
  :func:`lateralization_map`).
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import DescriptorMatrix, get_logger, progress_simple

logger = get_logger(__name__)


# ======================================================================
# S1  FUNCTIONAL MAP ESTIMATION
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
    from spectralbrain.spectral.lbo.descriptors import (
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

    The functional map C : F(M_a) -> F(M_b) is represented in the
    truncated eigenbasis as a (n_basis x n_basis) matrix satisfying:

        Phi_b^T * M_b * f_b ~ C * (Phi_a^T * M_a * f_a)

    for corresponding functions f_a, f_b.  It solves

        min_C ||C A - B||^2 + w_L ||C Lambda_a - Lambda_b C||^2 + rho ||C||^2

    where A, B stack the projected descriptors (each pair normalised to
    unit norm on the source side), Lambda are the eigenvalues (normalised by
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
        Tikhonov regularisation weight rho (only for conditioning; the
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
    # M_a-weighted projection: a_coeff = Phi_a^T * M_a * f_a
    M_a = decomp_a.mass
    M_b = decomp_b.mass
    if M_a is None or M_b is None:
        logger.warning(
            "compute_functional_map: mass matrix missing -- projecting with "
            "the Euclidean inner product (only exact for orthonormal Phi)."
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
    AB = A_mat @ B_mat.T  # (k, k): column i = A * B_i^T
    C = np.zeros((k, k), dtype=np.float64)
    eye = np.eye(k)
    for i in range(k):
        # Row i: (AA^T + w_L*diag((lambdaa - lambdab_i)^2) + rho I) c_i = A B_i^T
        lap = laplacian_weight * (lam_a - lam_b[i]) ** 2
        lhs = AAt + np.diag(lap) + regularize * eye
        C[i] = np.linalg.solve(lhs, AB[:, i])

    logger.debug("Functional map: %d x %d", C.shape[0], C.shape[1])
    return C


# ======================================================================
# S2  SHAPE DIFFERENCE OPERATORS
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
        ``"area"`` -- D_area = C^T * C (captures area distortion).
        ``"conformal"`` -- D_conf = C^T * Lambda_B * C * Lambda_A^{+}
        (Rustamov et al. 2013; Lambda_A^{+} is the pseudo-inverse, i.e. the
        null/constant mode is excluded).  Requires *evals_a*/*evals_b*.
    evals_a, evals_b : ndarray, shape (>=k,), optional
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
                "evals_b=... (D_conf = C^T Lambda_B C Lambda_A^+)."
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
# S3  DWKS -- Deformation Wave Kernel Signature
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
    # Step 1: Functional map C: source -> target.
    C = compute_functional_map(
        decomp_source,
        decomp_target,
        n_basis=n_basis,
        descriptor_pairs=descriptor_pairs,
    )

    # Step 2-3: Shape difference operator and its eigendecomposition
    # (as functions on the source shape, coefficients in Phi_source).
    k_c = C.shape[0]
    if diff_type == "area":
        D = shape_difference_operator(C, type="area")
        D_evals, D_evecs = np.linalg.eigh(D)  # (k,), (k, k)
    elif diff_type == "conformal":
        # D_conf = C^T Lambda_B C Lambda_A^+ is similar to the symmetric PSD matrix
        # S = Lambda_A^{+1/2} C^T Lambda_B C Lambda_A^{+1/2}; its eigenfunctions on A (the
        # eigenvectors of Lambda_A^+ C^T Lambda_B C) are Lambda_A^{+1/2} u.  The null
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
        # The 2sigma inward shift crosses over (few energies / narrow
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
    # D_evecs live in the spectral basis; pull to spatial via Phi_source.
    k = min(n_basis, decomp_source.n_eigenvalues, D_evecs.shape[0])
    Phi_source = decomp_source.eigenvectors[:, :k]  # (N, k)
    D_evecs_trunc = D_evecs[:k, :]  # (k, k)

    # DWKS(x, e) = Sigma_j g_e(mu_j) * (Phi * psi_j)^2(x)
    # where mu_j are D's eigenvalues and psi_j are D's eigenvectors.
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
# S4  CORRESPONDENCE-FREE SPECTRAL DEFORMATION (Hu et al., TVCG 2017)
# ======================================================================
#
# Non-isometric deformation between two surfaces quantified through the
# variation of their Laplace-Beltrami spectra -- no registration and no
# correspondence (Hu, Hamidian, Zhong & Hua, IEEE TVCG 2017, "Visualizing
# Shape Deformations with Variation of Geometric Spectrum").
#
# Discrete LBO:  W v = lambda S v  (W cotangent stiffness, S lumped mass).
# A deformation is a positive per-vertex scale function omega on the metric,
# g_omega = omega * g, giving the weighted problem
#
#     W v_i = lambda_i (Omega S) v_i,     <v_i, v_i>_{Omega S} = 1.
#
# Theorem 2 (eigenvalue derivative):  d lambda_i = -lambda_i v_i^T dOmega S v_i.
# Interpolating the eigenvalues of N linearly towards those of M gives a
# linear system in the scale derivative d = diag(dOmega) (Eq. 18):
#
#     (m_N * v_i * v_i)^T d = (lambda_i(q) - lambda_M,i) / lambda_i(q),  i=1..k1
#
# The system is under-determined (k1 << n); the minimum smoothing-energy
# solution is taken (Eq. 25):
#
#     min_d  d^T W d + 2 c^T d,   c = W v_Omega
#     s.t.   A d = b                      (eigenvalue alignment)
#            h_l <= v_Omega + d <= h_u    (positive, bounded scale)
#
# integrated linearly over K steps with the spectrum re-initialised at each
# step (Eq. 31):  Omega(q+1) = Omega(q) + dOmega(q) / (K - q),  Omega(0) = I.
#
# Output: the per-vertex scale omega on N; log(omega) > 0 means N must expand
# locally to match M, log(omega) < 0 that it must contract. The nodal lines of
# the deformation (log omega = 0) are what matters.
#
# Engineering notes: the cotangent stiffness and lumped mass are the library's
# own (BrainMesh / core.meshes), so this method shares the LBO of every other
# descriptor. The QP is solved with sparse OSQP; the alignment constraint stays
# a solver constraint, so the dense A^T A (n x n) is never formed. ``align_tol``
# relaxes the equality into a band ("soft" mode), which is more stable when the
# scale bounds are tight.


def _lbo_operators(V: np.ndarray, F: np.ndarray) -> tuple[sp.csc_matrix, np.ndarray]:
    """Cotangent stiffness ``W`` and lumped-mass diagonal ``m`` of a mesh."""
    from spectralbrain.core.meshes import _cotangent_laplacian

    W, M = _cotangent_laplacian(np.asarray(V, dtype=float), np.asarray(F, dtype=np.intp))
    return W.tocsc(), np.asarray(M.diagonal(), dtype=float)


def _weighted_eigs(
    W: sp.csc_matrix,
    mass_diag: np.ndarray,
    k_nonzero: int,
    sigma: float = -1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``W v = lambda B v`` with ``B = diag(mass_diag)``.

    Returns the ``k_nonzero`` smallest NON-zero eigenpairs (the constant
    mode, lambda ~ 0, is skipped). Eigenvectors are B-orthonormal.
    """
    B = sp.diags(mass_diag).tocsc()
    vals, vecs = eigsh(W, k=k_nonzero + 1, M=B, sigma=sigma, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    return vals[1 : k_nonzero + 1], vecs[:, 1 : k_nonzero + 1]


def _qp_step(
    W: sp.csc_matrix,
    v_omega: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    h_low: np.ndarray,
    h_high: np.ndarray,
    align_tol: float,
    ridge: float = 1e-8,
) -> np.ndarray:
    """One QP step (Eq. 19/25/27) with sparse OSQP.

        min_d  d^T W d + 2 c^T d,  c = W v_Omega
        s.t.   b - tol <= A d <= b + tol          (alignment; tol=0 => equality)
               h_l - v_Omega <= d <= h_u - v_Omega  (scale bounds)

    The equality stays a solver constraint -- the dense A^T A is never formed.
    """
    try:
        import osqp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OSQP is required for spectral_deformation: pip install osqp") from exc

    n = W.shape[0]
    P = (2.0 * W + ridge * sp.identity(n, format="csc")).tocsc()
    q = 2.0 * (W.dot(v_omega))

    A_align = sp.csc_matrix(A)  # (k1 x n)
    C = sp.vstack([A_align, sp.identity(n, format="csc")], format="csc")
    lo = np.concatenate([b - align_tol, h_low - v_omega])
    hi = np.concatenate([b + align_tol, h_high - v_omega])

    prob = osqp.OSQP()
    prob.setup(
        P=P,
        q=q,
        A=C,
        l=lo,
        u=hi,
        verbose=False,
        polish=True,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=8000,
    )
    res = prob.solve()
    status = res.info.status_val
    if status not in (1, 2):  # 1 = solved, 2 = solved_inaccurate
        warnings.warn(
            f"OSQP did not fully converge (status={res.info.status}). "
            "Consider a larger `align_tol` (soft mode) or looser scale bounds.",
            stacklevel=2,
        )
    d = res.x
    if d is None or not np.all(np.isfinite(d)):
        warnings.warn("QP returned an invalid solution; step set to zero.", stacklevel=2)
        return np.zeros(n)
    return np.asarray(d, dtype=float)


def spectral_deformation(
    V_N: np.ndarray,
    F_N: np.ndarray,
    V_M: np.ndarray,
    F_M: np.ndarray,
    k1: int = 100,
    n_steps: int = 10,
    bounds: tuple[float, float] = (0.1, 10.0),
    align_tol: float = 0.0,
    eig_sigma: float = -1e-6,
    return_history: bool = True,
) -> dict[str, object]:
    """Align the first ``k1`` eigenvalues of mesh N to those of mesh M.

    Correspondence-free and registration-free (Hu et al., TVCG 2017): the
    result is a per-vertex scale function ``omega`` on N whose metric makes
    N's LBO spectrum match M's. Meshes may have different vertex counts.

    Parameters
    ----------
    V_N, F_N : ndarray
        Vertices / faces of the SOURCE mesh N (the scale function lives here).
    V_M, F_M : ndarray
        Vertices / faces of the TARGET mesh M (only its eigenvalues are used).
    k1 : int
        Number of non-zero eigenvalues to align. Low -> global / coarse
        deformation; high -> finer-frequency deformation (paper: 100).
    n_steps : int
        K steps of the linear integration (paper: K = 10 suffices).
    bounds : (h_l, h_u)
        Bounds of the per-vertex scale (positive, finite).
    align_tol : float
        0.0 = hard equality (as in the paper); > 0 = band ("soft" mode).
    eig_sigma : float
        Shift of the eigsh shift-invert (a small negative avoids singularity).
    return_history : bool
        Also return the per-step mean relative alignment error.

    Returns
    -------
    dict
        ``scale`` (n_N,) per-vertex omega on N; ``log_scale`` (n_N,) log(omega),
        the field to visualise; ``lambda_M`` (k1,) target eigenvalues;
        ``lambda_N0`` (k1,) initial eigenvalues of N; ``lambda_final`` (k1,)
        eigenvalues of N after the deformation (~ ``lambda_M``);
        ``align_error`` (K,) if ``return_history``; ``k1``, ``n_steps``.

    Notes
    -----
    Cost is dominated by K + 2 shift-invert eigensolves of size n. On
    hippocampal meshes (~5k-8k vertices) with k1 = 100 and K = 10 it runs in
    seconds to a few minutes. ``align_error`` should decrease monotonically
    across the K steps.
    """
    V_N = np.asarray(V_N, float)
    F_N = np.asarray(F_N, int)
    V_M = np.asarray(V_M, float)
    F_M = np.asarray(F_M, int)
    n_N = V_N.shape[0]
    h_l, h_u = float(bounds[0]), float(bounds[1])
    h_low = np.full(n_N, h_l)
    h_high = np.full(n_N, h_u)

    # operators of N (fixed lumped mass; only the scale Omega changes)
    W_N, m_N = _lbo_operators(V_N, F_N)

    # target eigenvalues of M (once)
    W_M, m_M = _lbo_operators(V_M, F_M)
    lam_M, _ = _weighted_eigs(W_M, m_M, k1, sigma=eig_sigma)

    # initial state: Omega(0) = I  ->  v_Omega = 1
    v_omega = np.ones(n_N)
    lam_N0, _ = _weighted_eigs(W_N, m_N, k1, sigma=eig_sigma)

    align_err = []
    K = int(n_steps)
    for q in range(K):
        # current eigenpairs of N under the scaled metric, B = diag(v_Omega * m_N)
        b_diag = v_omega * m_N
        lam_q, vecs_q = _weighted_eigs(W_N, b_diag, k1, sigma=eig_sigma)

        if return_history:
            rel = np.mean(np.abs(lam_q - lam_M) / np.maximum(np.abs(lam_M), 1e-12))
            align_err.append(float(rel))

        # linear system A d = b (Eq. 18/19): A_i = m_N * v_i * v_i
        A = m_N[None, :] * (vecs_q.T**2)  # (k1 x n_N)
        b = (lam_q - lam_M) / np.maximum(lam_q, 1e-12)

        d = _qp_step(W_N, v_omega, A, b, h_low, h_high, align_tol)

        # linear integration (Eq. 31) + projection onto the bounds
        v_omega = v_omega + d / (K - q)
        v_omega = np.clip(v_omega, h_l, h_u)

    lam_final, _ = _weighted_eigs(W_N, v_omega * m_N, k1, sigma=eig_sigma)

    out: dict[str, object] = {
        "scale": v_omega,
        "log_scale": np.log(np.clip(v_omega, 1e-12, None)),
        "lambda_M": lam_M,
        "lambda_N0": lam_N0,
        "lambda_final": lam_final,
        "k1": k1,
        "n_steps": K,
    }
    if return_history:
        out["align_error"] = np.asarray(align_err, dtype=float)
    return out


def lateralization_map(
    V_ipsi: np.ndarray,
    F_ipsi: np.ndarray,
    V_contra: np.ndarray,
    F_contra: np.ndarray,
    k1: int = 100,
    n_steps: int = 10,
    bounds: tuple[float, float] = (0.1, 10.0),
    align_tol: float = 0.0,
) -> dict[str, object]:
    """Left-vs-right lateralisation map, defined on the IPSILATERAL mesh.

    Convention (MTLE-HS): N = ipsilateral (side of the sclerosis), M =
    contralateral. ``log(omega) > 0`` marks where the ipsilateral side must
    expand to match the contralateral one -- regions of relative ipsilateral
    atrophy; ``log(omega) < 0`` marks regions where it is relatively larger.

    The method is correspondence-free and reflection-invariant: do NOT mirror
    or register the meshes; pass them as produced by HippUnfold or the
    segmentation.
    """
    res = spectral_deformation(
        V_ipsi,
        F_ipsi,
        V_contra,
        F_contra,
        k1=k1,
        n_steps=n_steps,
        bounds=bounds,
        align_tol=align_tol,
    )
    res["convention"] = (
        "N=ipsilateral, M=contralateral; log(omega)>0 => relative ipsilateral atrophy"
    )
    return res


__all__: list[str] = [
    "compute_dwks",
    "compute_dwks_collection",
    "compute_functional_map",
    "lateralization_map",
    "shape_difference_operator",
    "spectral_deformation",
]
