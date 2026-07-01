"""Morphometric Gaussian Process (M-GP) active-learning landmarking.

Bayesian feature selection for spectral shape analysis: reduce a shape to a
concise, interpretable set of *landmarks* — the vertices of maximal Gaussian
Process posterior variance — together with the full residual-variance map.
After Fan, Wang, Dong, Liu, Leporé & Wang, *Med. Image Anal.* 72 (2021) 102123,
"Tetrahedral spectral feature-Based bayesian manifold learning".

The GP kernel fuses two spectral cues, both derived from the same
:class:`~spectralbrain.core.base.SpectralDecomposition` that powers the rest of
``spectral/``:

* **SIWKS** — Scale-Invariant Wave Kernel Signature (:func:`compute_si_wks`),
  the Wave Kernel Signature multiplied by ``1/λ_K`` for scale invariance;
* **HFE**   — Heat Flow Entropy (:func:`heat_flow_entropy`), the Shannon entropy
  of heat-gradient magnitudes across the 1-ring, a per-vertex saliency.

Kernel (Eq. 11/13):  ``K = M̄ H M̄ᵀ`` with ``M̄`` the row-normalised, KNN-sparse
SIWKS distance map and ``H = diag(HFE)``. The symmetric form is PSD because
``HFE ≥ 0`` — a valid covariance.

Landmarking (Eq. 15–18) is greedy maximisation of the GP posterior variance,
implemented as a **pivoted Cholesky** of ``K``: the residual diagonal after
``l`` pivots equals the posterior variance exactly and its argmax is the next
landmark. Cost ``O(N·L)``; the dense ``N×N`` kernel is never formed — its
columns come from sparse mat-vecs.

Concatenating the SIWKS of the landmarks yields a compact per-subject global
descriptor with strong statistical power and clear anatomical localisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    ScalarMap,
    get_logger,
)
from spectralbrain.spectral.descriptors import compute_wks

logger = get_logger(__name__)

__all__ = [
    "MGPLandmarkResult",
    "compute_si_wks",
    "heat_flow_entropy",
    "mgp_landmarks",
]


# ───────────────────────────── result container ──────────────────────────────
@dataclass
class MGPLandmarkResult:
    """Output of :func:`mgp_landmarks`.

    Attributes
    ----------
    landmarks : ndarray (L,)
        Vertex indices, in selection order.
    descriptor : ndarray (L * E,)
        Concatenated SIWKS of the landmarks — the concise subject descriptor.
    uncertainty : ScalarMap (N,)
        Residual GP posterior variance after the ``L`` landmarks.
    selection_scores : ndarray (L,)
        Posterior variance at each landmark's selection (saturation curve).
    siwks : DescriptorMatrix (N, E)
        Per-vertex SIWKS feature matrix.
    hfe : ScalarMap (N,)
        Heat Flow Entropy per vertex.
    metadata : dict
    """

    landmarks: np.ndarray
    descriptor: np.ndarray
    uncertainty: ScalarMap
    selection_scores: np.ndarray
    siwks: DescriptorMatrix
    hfe: ScalarMap
    metadata: dict = field(default_factory=dict)

    @property
    def n_landmarks(self) -> int:
        return int(self.landmarks.shape[0])


# ───────────────────────────── spectral descriptors ──────────────────────────
def compute_si_wks(
    decomp: SpectralDecomposition,
    e_values: np.ndarray | None = None,
    *,
    n_energies: int = 100,
    sigma: float | None = None,
) -> DescriptorMatrix:
    r"""Scale-Invariant Wave Kernel Signature (SIWKS).

    The WKS rescaled by the inverse of the largest used eigenvalue,

    .. math::

        \mathrm{SIWKS}(x, e) = \frac{1}{\lambda_K}\, \mathrm{WKS}(x, e),

    which makes the descriptor invariant to global scaling of the shape
    (under :math:`\beta M`, :math:`\lambda \to \beta^2 \lambda`). Complements
    :func:`~spectralbrain.spectral.descriptors.compute_si_hks` on the WKS side.

    Parameters
    ----------
    decomp : SpectralDecomposition
    e_values : ndarray (E,), optional
        Log-energy levels; ``None`` -> auto from eigenvalues.
    n_energies : int
        Number of auto energy levels.
    sigma : float, optional
        Gaussian bandwidth; ``None`` -> Aubry convention.

    Returns
    -------
    DescriptorMatrix, shape (N, E)
    """
    wks = compute_wks(
        decomp, e_values, n_energies=n_energies, sigma=sigma, normalize=False
    )
    lam = np.asarray(decomp.eigenvalues, dtype=float)
    lam_K = float(lam[lam > 1e-10][-1])  # largest non-trivial eigenvalue
    return np.asarray(wks, dtype=float) / lam_K


def _adjacency_from_decomp(
    decomp: SpectralDecomposition,
    faces: Optional[np.ndarray],
    n_vertices: int,
) -> sp.csr_matrix:
    """1-ring adjacency: from ``faces`` if given, else from the stiffness pattern."""
    if faces is not None:
        F = np.asarray(faces, dtype=int)
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        A = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)),
                          shape=(n_vertices, n_vertices))
    else:
        if decomp.stiffness is None:
            raise ValueError(
                "HFE needs mesh connectivity: pass `faces`, or provide a "
                "SpectralDecomposition that carries its `stiffness` matrix."
            )
        A = sp.csr_matrix(decomp.stiffness)
        A = (A != 0).astype(float)
    A = A.tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def heat_flow_entropy(
    decomp: SpectralDecomposition,
    *,
    t: float | None = None,
    heat: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> ScalarMap:
    r"""Heat Flow Entropy (HFE) — per-vertex structural saliency.

    Diffuses heat on the surface (HKS field at time ``t``) and measures the
    Shannon entropy of the heat-gradient magnitudes across the 1-ring,

    .. math::

        \mathrm{HFE}(v_i) = -\sum_{j \in N(i)} p_{ij} \log p_{ij},
        \quad p_{ij} = \frac{|h_j - h_i|}{\sum_{j} |h_j - h_i|} \ge 0.

    High HFE marks structurally disordered regions (e.g. pial/white-matter
    transitions), used as the GP kernel weight in :func:`mgp_landmarks`.

    Parameters
    ----------
    decomp : SpectralDecomposition
    t : float, optional
        Heat diffusion time; ``None`` -> spectral mid-time.
    heat : ndarray (N,), optional
        Precomputed heat field (overrides ``t``).
    faces : ndarray (m, 3), optional
        Faces for the neighbourhood graph; if omitted, the stiffness sparsity
        pattern is used.

    Returns
    -------
    ScalarMap, shape (N,)
    """
    lam = np.asarray(decomp.eigenvalues, dtype=float)
    phi = np.asarray(decomp.eigenvectors, dtype=float)
    n = phi.shape[0]

    if heat is None:
        pos = lam > 1e-10
        lp, pp = lam[pos], phi[:, pos]
        if t is None:
            t = float(np.exp(0.5 * (np.log(1.0 / lp[-1]) + np.log(1.0 / lp[0]))))
        h = (pp ** 2) @ np.exp(-lp * t)
    else:
        h = np.asarray(heat, dtype=float)

    A = _adjacency_from_decomp(decomp, faces, n)
    hfe = np.zeros(n)
    indptr, indices = A.indptr, A.indices
    for i in range(n):
        nbr = indices[indptr[i]:indptr[i + 1]]
        if nbr.size == 0:
            continue
        d = np.abs(h[nbr] - h[i])
        s = d.sum()
        if s <= 1e-12:
            continue
        p = d / s
        nz = p > 0
        hfe[i] = -np.sum(p[nz] * np.log(p[nz]))
    return hfe


# ──────────────────────── distance map + kernel operator ─────────────────────
def _siwks_distance_map(S: np.ndarray, knn: int) -> sp.csr_matrix:
    n = S.shape[0]
    k = min(knn + 1, n)
    tree = cKDTree(S)
    _, idx = tree.query(S, k=k)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    w = np.abs(S[rows] - S[cols]).sum(axis=1)
    m = rows != cols
    M = sp.csr_matrix((w[m], (rows[m], cols[m])), shape=(n, n))
    return M.maximum(M.T).tocsr()


def _row_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    M = M.tocsr().copy()
    counts = np.diff(M.indptr).astype(float)
    counts[counts == 0] = 1.0
    return (sp.diags(1.0 / counts) @ M).tocsr()


class _MGPKernel:
    """K = M̄ H M̄ᵀ as an operator (diagonal + columns on demand, never dense)."""

    def __init__(self, Mbar: sp.csr_matrix, hfe: np.ndarray):
        self.Mbar = Mbar
        self.hfe = np.asarray(hfe, float)

    def diag(self) -> np.ndarray:
        return np.asarray(self.Mbar.multiply(self.Mbar) @ self.hfe).ravel()

    def column(self, j: int) -> np.ndarray:
        row_j = self.Mbar.getrow(j).toarray().ravel()
        return np.asarray(self.Mbar @ (self.hfe * row_j)).ravel()


def _pivoted_cholesky(kernel: _MGPKernel, n_landmarks: int, jitter: float):
    N = kernel.Mbar.shape[0]
    L = int(min(n_landmarks, N))
    d = kernel.diag().astype(float)
    G = np.zeros((N, L), float)
    landmarks = np.empty(L, int)
    scores = np.empty(L, float)
    chosen = np.zeros(N, bool)
    for l in range(L):
        j = int(np.argmax(np.where(chosen, -np.inf, d)))
        piv = d[j]
        landmarks[l], scores[l], chosen[j] = j, piv, True
        if piv <= jitter:
            landmarks, scores, G = landmarks[:l + 1], scores[:l + 1], G[:, :l + 1]
            break
        col = kernel.column(j)
        if l > 0:
            col = col - G[:, :l] @ G[j, :l]
        g = col / np.sqrt(piv)
        G[:, l] = g
        d = np.maximum(d - g ** 2, 0.0)
    return landmarks, d, scores


# ───────────────────────────────── public API ────────────────────────────────
def mgp_landmarks(
    decomp: SpectralDecomposition,
    *,
    faces: np.ndarray | None = None,
    n_landmarks: int = 50,
    n_energies: int = 100,
    knn: int = 20,
    sigma: float | None = None,
    heat_t: float | None = None,
    heat: np.ndarray | None = None,
    jitter: float = 1e-10,
) -> MGPLandmarkResult:
    """Select landmarks by M-GP active learning from a spectral decomposition.

    Parameters
    ----------
    decomp : SpectralDecomposition
        The central SpectralBrain object (eigenpairs; ``stiffness`` enables the
        HFE neighbourhood when ``faces`` is not supplied).
    faces : ndarray (m, 3), optional
        Triangular faces for the HFE 1-ring; if omitted, taken from the
        stiffness sparsity pattern.
    n_landmarks : int
        Number of landmarks ``L`` to select.
    n_energies : int
        SIWKS energy scales (per-landmark descriptor length).
    knn : int
        Neighbours in the SIWKS distance-map graph.
    sigma : float, optional
        SIWKS bandwidth (None -> Aubry rule).
    heat_t : float, optional
        Heat-field diffusion time for HFE (None -> spectral mid-time).
    heat : ndarray (N,), optional
        Precomputed heat field (overrides ``heat_t``).
    jitter : float
        Pivot floor for the Cholesky stop.

    Returns
    -------
    MGPLandmarkResult
    """
    S = compute_si_wks(decomp, n_energies=n_energies, sigma=sigma)
    hfe = heat_flow_entropy(decomp, t=heat_t, heat=heat, faces=faces)

    Mbar = _row_normalize(_siwks_distance_map(S, knn))
    kernel = _MGPKernel(Mbar, hfe)
    landmarks, residual_var, scores = _pivoted_cholesky(kernel, n_landmarks, jitter)

    logger.info(
        "M-GP landmarking: selected %d landmarks from %d vertices "
        "(SIWKS E=%d, knn=%d)", landmarks.shape[0], S.shape[0], n_energies, knn,
    )

    return MGPLandmarkResult(
        landmarks=landmarks,
        descriptor=S[landmarks].ravel(),
        uncertainty=residual_var,
        selection_scores=scores,
        siwks=S,
        hfe=hfe,
        metadata=dict(
            method="mgp_landmarking",
            reference="Fan et al., Med. Image Anal. 72 (2021) 102123",
            n_landmarks=int(landmarks.shape[0]),
            n_energies=int(n_energies),
            knn=int(knn),
            structure=(decomp.metadata or {}).get("structure"),
        ),
    )
