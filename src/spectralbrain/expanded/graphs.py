"""Graph-spectral descriptors for connectomes and mesh skeletons.

Where the rest of :mod:`spectralbrain.expanded` works on the *geometry* of a
surface, this module works on the spectrum of a **graph** — a structural or
functional connectome, or a mesh 1-skeleton.  The graph Laplacian and
adjacency spectra yield a family of compact, permutation-invariant
descriptors that summarise global organisation:

- **NetLSD** — the heat/wave trace ``Σ_k e^{-tλ_k}`` of the normalised
  Laplacian, a multiscale "shape of the graph" signature (Tsitsulin et al.
  2018).
- **FGSD** — a histogram of pairwise spectral (e.g. biharmonic/resistance)
  distances, a fast graph fingerprint (Verma & Zhang 2017).
- **Effective resistance / Kirchhoff index** and the **Estrada index** —
  classical spectral invariants of network communicability and robustness.
- **Fiedler / normalised-cut features** — algebraic connectivity, spectral
  gap, spectral radius and related summaries.

All descriptors here are pure NumPy/SciPy (with optional ``networkx`` input
coercion) and need no heavy dependency.

References
----------
Tsitsulin A et al. "NetLSD: Hearing the shape of a graph." *KDD*, 2018.
Verma S, Zhang ZL. "Hunt for the unique, stable, sparse and fast feature
learning on graphs." *NeurIPS*, 2017.
Estrada E. "Characterization of 3D molecular structure." *Chem. Phys.
Lett.* 319:713–718, 2000.
Klein DJ, Randić M. "Resistance distance." *J. Math. Chem.* 12:81–95, 1993.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    GlobalDescriptor,
    get_logger,
)

from spectralbrain.expanded._base import BackendSpec, operator_eigensystem

logger = get_logger(__name__)

_EPS = 1e-12
_DENSE_LIMIT = 3000


# ======================================================================
# §1  INPUT COERCION + LAPLACIANS
# ======================================================================


def _to_sparse_adjacency(adjacency: Any) -> sp.csr_matrix:
    """Coerce a sparse/dense matrix or networkx graph to a CSR adjacency."""
    try:
        import networkx as nx

        if isinstance(adjacency, nx.Graph):
            return nx.to_scipy_sparse_array(adjacency, dtype=float, format="csr")
    except ImportError:
        pass
    return sp.csr_matrix(adjacency, dtype=np.float64)


def _graph_laplacian(
    A: sp.csr_matrix, *, normalized: bool
) -> sp.csr_matrix:
    """Combinatorial ``L = D − A`` or symmetric-normalised Laplacian."""
    A = 0.5 * (A + A.T)  # symmetrise
    deg = np.asarray(A.sum(axis=1)).ravel()
    if normalized:
        dis = 1.0 / np.sqrt(np.clip(deg, _EPS, None))
        n = A.shape[0]
        L = sp.identity(n, format="csr") - sp.diags(dis) @ A @ sp.diags(dis)
    else:
        L = sp.diags(deg) - A
    return sp.csr_matrix(0.5 * (L + L.T))


def _laplacian_spectrum(
    A: sp.csr_matrix, *, normalized: bool, k: int | None
) -> np.ndarray:
    """All (dense) or k smallest Laplacian eigenvalues."""
    L = _graph_laplacian(A, normalized=normalized)
    n = L.shape[0]
    if k is None or n <= _DENSE_LIMIT:
        return np.clip(np.linalg.eigvalsh(L.toarray()), 0.0, None)
    import scipy.sparse.linalg as spla

    k = min(k, n - 2)
    w = spla.eigsh(L, k=k, sigma=-0.01, which="LM", return_eigenvectors=False)
    return np.clip(np.sort(w), 0.0, None)


def graph_laplacian_decompose(
    adjacency: Any,
    *,
    k: int = 100,
    normalized: bool = True,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Eigendecompose a graph Laplacian.

    Parameters
    ----------
    adjacency : sparse matrix, ndarray, or networkx.Graph
    k : int
        Number of eigenpairs.
    normalized : bool
        Use the symmetric-normalised Laplacian.
    backend : str or backend object
        Multi-backend selector.

    Returns
    -------
    SpectralDecomposition
        ``metadata["operator"] == "graph_laplacian"``.
    """
    A = _to_sparse_adjacency(adjacency)
    L = _graph_laplacian(A, normalized=normalized)
    return operator_eigensystem(
        L,
        M=None,
        k=k,
        backend=backend,
        clamp_nonneg=True,
        operator="graph_laplacian",
        metadata={"normalized": normalized, "domain": "nodes"},
    )


# ======================================================================
# §2  NetLSD  (heat / wave trace signature)
# ======================================================================


def netlsd(
    adjacency: Any,
    *,
    kind: Literal["heat", "wave"] = "heat",
    timescales: np.ndarray | None = None,
    n_scales: int = 64,
    k: int | None = None,
    normalize: Literal["none", "empty", "complete"] = "none",
) -> GlobalDescriptor:
    """NetLSD heat- or wave-trace spectral signature of a graph.

    ``heat`` → ``h(t) = Σ_k e^{-t λ_k}``; ``wave`` → ``w(t) = Σ_k cos(t λ_k)``,
    over the normalised-Laplacian spectrum.  A size-aware, multiscale
    fingerprint comparable across graphs.

    Parameters
    ----------
    adjacency : sparse matrix, ndarray, or networkx.Graph
    kind : str
        ``"heat"`` or ``"wave"``.
    timescales : ndarray, optional
        Explicit scales; log-spaced in ``[1e-2, 1e2]`` if ``None``.
    n_scales : int
        Number of auto scales.
    k : int, optional
        Use only ``k`` smallest eigenvalues for large graphs (full spectrum
        if ``None``).
    normalize : str
        ``"none"``; ``"empty"`` divides by the empty-graph trace ``N``;
        ``"complete"`` normalises by the complete-graph trace.

    Returns
    -------
    ndarray, shape (n_scales,)
        The trace evaluated at each timescale.

    Notes
    -----
    ``h(0) = N`` (node count) and ``h(∞) → b₀`` (connected components).
    """
    A = _to_sparse_adjacency(adjacency)
    evals = _laplacian_spectrum(A, normalized=True, k=k)
    n = A.shape[0]
    if timescales is None:
        timescales = np.logspace(-2, 2, n_scales)
    t = np.asarray(timescales, dtype=np.float64)

    if kind == "heat":
        trace = np.exp(-np.outer(t, evals)).sum(axis=1)  # (T,)
    elif kind == "wave":
        trace = np.cos(np.outer(t, evals)).sum(axis=1)
    else:
        raise ValueError(f"kind must be 'heat' or 'wave', got {kind!r}.")

    if normalize == "empty":
        trace = trace / n
    elif normalize == "complete":
        comp = np.exp(-np.outer(t, np.full(n, n))).sum(axis=1) + 1.0
        trace = trace / comp
    return trace


# ======================================================================
# §3  FGSD  (spectral-distance histogram)
# ======================================================================


def fgsd(
    adjacency: Any,
    *,
    bins: int = 200,
    hist_range: tuple[float, float] | None = None,
    f: Literal["biharmonic", "heat"] = "biharmonic",
    density: bool = True,
) -> GlobalDescriptor:
    """Family-of-Graph-Spectral-Distances histogram descriptor.

    Computes a pairwise spectral distance ``S(x, y) = Σ_{k≥1} f(λ_k)
    (φ_k(x) − φ_k(y))²`` and returns its histogram over all node pairs.
    With ``f = 1/λ`` (``"biharmonic"``) this is the effective-resistance /
    biharmonic distance; with ``f = e^{-λ}`` (``"heat"``) a diffusion
    distance.

    Parameters
    ----------
    adjacency : sparse matrix, ndarray, or networkx.Graph
    bins : int
        Number of histogram bins.
    hist_range : (float, float), optional
        Histogram range (auto from the data if ``None``).
    f : str
        Spectral filter: ``"biharmonic"`` (``1/λ``) or ``"heat"``
        (``e^{-λ}``).
    density : bool
        Normalise the histogram to a density.

    Returns
    -------
    ndarray, shape (bins,)
        The spectral-distance histogram (graph fingerprint).
    """
    A = _to_sparse_adjacency(adjacency)
    L = _graph_laplacian(A, normalized=False).toarray()
    w, V = np.linalg.eigh(L)
    nz = w > 1e-8
    w_nz, V_nz = w[nz], V[:, nz]

    if f == "biharmonic":
        filt = 1.0 / w_nz
    elif f == "heat":
        filt = np.exp(-w_nz)
    else:
        raise ValueError(f"f must be 'biharmonic' or 'heat', got {f!r}.")

    # S(x,y) = Σ filt_k (φ_kx − φ_ky)² = g_xx + g_yy − 2 g_xy with
    # g = V_nz diag(filt) V_nzᵀ
    g = (V_nz * filt[None, :]) @ V_nz.T
    diag = np.diag(g)
    S = diag[:, None] + diag[None, :] - 2.0 * g
    iu = np.triu_indices(S.shape[0], k=1)
    dists = S[iu]

    if hist_range is None:
        hist_range = (float(dists.min()), float(dists.max()) + _EPS)
    hist, _ = np.histogram(dists, bins=bins, range=hist_range, density=density)
    return hist


# ======================================================================
# §4  EFFECTIVE RESISTANCE / KIRCHHOFF / ESTRADA
# ======================================================================


def effective_resistance(adjacency: Any) -> np.ndarray:
    """Pairwise effective-resistance (commute-distance) matrix.

    ``R(x, y) = L⁺_xx + L⁺_yy − 2 L⁺_xy`` with ``L⁺`` the Laplacian
    pseudoinverse.  For connected graphs ``R`` is a metric.

    Returns
    -------
    ndarray, shape (N, N)
        The effective-resistance matrix.
    """
    A = _to_sparse_adjacency(adjacency)
    L = _graph_laplacian(A, normalized=False).toarray()
    Lp = np.linalg.pinv(L)
    d = np.diag(Lp)
    return d[:, None] + d[None, :] - 2.0 * Lp


def kirchhoff_index(adjacency: Any) -> float:
    """Kirchhoff index ``Kf = Σ_{x<y} R(x, y) = N · Σ_{k≥1} 1/λ_k``.

    A global measure of network "spread"/robustness.
    """
    A = _to_sparse_adjacency(adjacency)
    evals = _laplacian_spectrum(A, normalized=False, k=None)
    nz = evals[evals > 1e-8]
    n = A.shape[0]
    return float(n * np.sum(1.0 / nz))


def estrada_index(adjacency: Any) -> float:
    """Estrada index ``EE = Σ_i e^{μ_i} = tr(exp(A))`` of the adjacency.

    Quantifies subgraph communicability / structural folding.
    """
    A = _to_sparse_adjacency(adjacency)
    As = 0.5 * (A + A.T)
    mu = np.linalg.eigvalsh(As.toarray())
    return float(np.sum(np.exp(mu)))


# ======================================================================
# §5  FIEDLER / NORMALISED-CUT SPECTRAL FEATURES
# ======================================================================


def fiedler_value(adjacency: Any, *, normalized: bool = False) -> float:
    """Algebraic connectivity λ₁ (the Fiedler value) of the Laplacian."""
    A = _to_sparse_adjacency(adjacency)
    evals = _laplacian_spectrum(A, normalized=normalized, k=None)
    return float(evals[1]) if evals.size > 1 else 0.0


def fiedler_vector(adjacency: Any, *, normalized: bool = False) -> np.ndarray:
    """The Fiedler eigenvector (φ₁), the basis of spectral bipartitioning."""
    A = _to_sparse_adjacency(adjacency)
    L = _graph_laplacian(A, normalized=normalized).toarray()
    _, V = np.linalg.eigh(L)
    return V[:, 1]


def spectral_features(adjacency: Any) -> dict[str, float]:
    """A compact vector of global graph-spectral summary statistics.

    Returns
    -------
    dict
        ``n_nodes``, ``n_edges``, ``algebraic_connectivity`` (λ₁),
        ``spectral_gap`` (λ₂ − λ₁), ``laplacian_spectral_radius`` (λ_max),
        ``adjacency_spectral_radius`` (μ_max), ``estrada_index`` and
        ``kirchhoff_index``.
    """
    A = _to_sparse_adjacency(adjacency)
    As = 0.5 * (A + A.T)
    n = A.shape[0]
    n_edges = float(As.nnz) / 2.0
    levals = _laplacian_spectrum(A, normalized=False, k=None)
    mu = np.linalg.eigvalsh(As.toarray())
    nz = levals[levals > 1e-8]
    return {
        "n_nodes": int(n),
        "n_edges": float(n_edges),
        "algebraic_connectivity": float(levals[1]) if n > 1 else 0.0,
        "spectral_gap": float(levals[2] - levals[1]) if n > 2 else 0.0,
        "laplacian_spectral_radius": float(levals[-1]),
        "adjacency_spectral_radius": float(np.max(np.abs(mu))),
        "estrada_index": float(np.sum(np.exp(mu))),
        "kirchhoff_index": float(n * np.sum(1.0 / nz)) if nz.size else float("inf"),
    }


__all__ = [
    "effective_resistance",
    "estrada_index",
    "fgsd",
    "fiedler_value",
    "fiedler_vector",
    "graph_laplacian_decompose",
    "kirchhoff_index",
    "netlsd",
    "spectral_features",
]
