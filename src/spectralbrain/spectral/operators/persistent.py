"""Persistent topology: diagrams, persistent Laplacians, and Hodge-Dirac.

Persistent homology turns a *filtration* of a shape -- a growing family of
complexes indexed by a scale -- into a multiscale topological summary, the
**persistence diagram**.  Its spectral refinements (the **persistent
Laplacian** and the **Hodge-Dirac** operator) recover not only *when*
homology classes are born and die but also the geometry of the harmonic
representatives in between.

This module provides:

Persistence diagrams
    A built-in ``H_0`` diagram via union-find on an edge filtration (exact,
    dependency-free) and full Vietoris-Rips / cubical diagrams through the
    optional ``ripser``/``gudhi`` backends.

Vectorisations
    :func:`betti_curve`, :func:`persistence_landscape` and
    :func:`persistence_statistics` and :func:`persistence_image` (all
    built-in) -- turning a diagram into a
    fixed-length feature vector for statistics and learning.

Persistent Laplacian
    The ``q = 0`` persistent Laplacian of a vertex-subset pair ``K subset of L``
    (Wang, Nguyen & Wei 2020): the Schur complement of ``L``'s graph
    Laplacian onto the vertices of ``K``.  Its kernel dimension is the persistent Betti number
    ``beta_0^{K,L}``; at ``K = L`` it is the ordinary graph Laplacian.

Hodge-Dirac operator
    The combinatorial Dirac ``D = d + d^T`` assembled from the boundary
    matrices.  Its defining identity ``D^2 = L_0 (+) L_1 (+) L_2`` ties it to the
    Hodge Laplacians of :mod:`spectralbrain.spectral.operators.topologic`, and
    ``dim ker D = b_0 + b_1 + b_2`` is the total Betti number.  (Named
    ``combinatorial_dirac`` to distinguish it from the extrinsic
    quaternionic :func:`spectralbrain.spectral.operators.extrinsic.dirac_operator`.)

The optional dependencies (``ripser``, ``gudhi``) are lazily
imported and declared in the ``expanded`` extra; the built-in routines work
without them.

References
----------
Edelsbrunner H, Harer J. *Computational Topology: An Introduction.* AMS,
2010.
Bubenik P. "Statistical topological data analysis using persistence
landscapes." *JMLR* 16:77-102, 2015.
Wang R, Nguyen DD, Wei GW. "Persistent spectral graph." *Int. J. Numer.
Methods Biomed. Eng.* 36(9):e3376, 2020.
Ameneyro B, Siopsis G, Maroulas V. "Quantum persistent homology."
*J. Applied and Computational Topology*, 2024 (Hodge-Dirac formulation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    Faces,
    GlobalDescriptor,
    SparseMatrix,
    Vertices,
    get_logger,
)
from spectralbrain.spectral.operators._base import BackendSpec, require_optional

logger = get_logger(__name__)

_EPS = 1e-12


# ======================================================================
# S1  PERSISTENCE DIAGRAMS
# ======================================================================


class _UnionFind:
    """Union-find with birth tracking for the ``H_0`` elder rule."""

    def __init__(self, birth: np.ndarray) -> None:
        self.parent = np.arange(birth.shape[0])
        self.birth = birth.astype(np.float64).copy()

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int, value: float) -> tuple[float, float] | None:
        """Merge components of ``a`` and ``b`` at filtration ``value``.

        Returns the ``(birth, death)`` pair of the component that dies
        (younger born -> dies), or ``None`` if already merged.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        # elder rule: the younger (later-born) root dies
        if self.birth[ra] <= self.birth[rb]:
            elder, younger = ra, rb
        else:
            elder, younger = rb, ra
        death_birth = self.birth[younger]
        self.parent[younger] = elder
        self.birth[elder] = min(self.birth[elder], self.birth[younger])
        if value > death_birth + _EPS:
            return (float(death_birth), float(value))
        return None


def graph_persistence_h0(
    n_vertices: int,
    edges: np.ndarray,
    edge_values: np.ndarray,
    *,
    vertex_values: np.ndarray | None = None,
) -> DescriptorMatrix:
    """Built-in ``H_0`` persistence diagram of an edge filtration.

    Sweeps edges in increasing filtration value, merging connected
    components with the elder rule.  Equivalent to single-linkage / the
    minimum-spanning-tree death times; dependency-free and exact.

    Parameters
    ----------
    n_vertices : int
    edges : ndarray, shape (E, 2)
        Edge endpoint indices.
    edge_values : ndarray, shape (E,)
        Filtration value at which each edge enters.
    vertex_values : ndarray, shape (N,), optional
        Birth value of each vertex (zeros if ``None``).

    Returns
    -------
    ndarray, shape (n_bars, 2)
        ``(birth, death)`` pairs; the essential component has
        ``death = inf``.
    """
    birth = (
        np.zeros(n_vertices)
        if vertex_values is None
        else np.asarray(vertex_values, dtype=np.float64)
    )
    uf = _UnionFind(birth)
    order = np.argsort(np.asarray(edge_values, dtype=np.float64), kind="stable")
    bars: list[tuple[float, float]] = []
    for e in order:
        a, b = int(edges[e, 0]), int(edges[e, 1])
        res = uf.union(a, b, float(edge_values[e]))
        if res is not None:
            bars.append(res)
    # essential classes (one per surviving component)
    roots = {uf.find(i) for i in range(n_vertices)}
    for r in roots:
        bars.append((float(uf.birth[r]), np.inf))
    return np.asarray(bars, dtype=np.float64).reshape(-1, 2)


def vietoris_rips_diagram(
    points: np.ndarray,
    *,
    max_dim: int = 1,
    max_edge_length: float | None = None,
    backend: str = "auto",
) -> list[np.ndarray]:
    """Vietoris-Rips persistence diagram of a finite point set.

    Uses the optional ``ripser`` or ``gudhi`` backend (declared in the
    ``expanded`` extra).  For ``H_0`` only, :func:`graph_persistence_h0` is a
    dependency-free alternative.

    Parameters
    ----------
    points : ndarray, shape (P, D)
        Point coordinates.
    max_dim : int
        Maximum homology dimension.
    max_edge_length : float, optional
        Rips threshold (auto if ``None``).
    backend : str
        ``"ripser"``, ``"gudhi"`` or ``"auto"``.

    Returns
    -------
    list of ndarray
        One ``(n_bars, 2)`` array of ``(birth, death)`` pairs per homology
        dimension ``0 ... max_dim``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if backend in ("auto", "ripser"):
        try:
            from ripser import ripser as _ripser

            kw: dict[str, Any] = {"maxdim": max_dim}
            if max_edge_length is not None:
                kw["thresh"] = max_edge_length
            return list(_ripser(pts, **kw)["dgms"])
        except ImportError:
            if backend == "ripser":
                raise

    gudhi = require_optional("gudhi", purpose="Vietoris-Rips persistence")
    thresh = max_edge_length if max_edge_length is not None else float("inf")
    rips = gudhi.RipsComplex(points=pts, max_edge_length=thresh)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    dgms: list[list[tuple[float, float]]] = [[] for _ in range(max_dim + 1)]
    for dim, (b, d) in st.persistence():
        if dim <= max_dim:
            dgms[dim].append((b, d))
    return [np.asarray(d, dtype=np.float64).reshape(-1, 2) for d in dgms]


# ======================================================================
# S2  DIAGRAM VECTORISATIONS
# ======================================================================


def _finite_bars(diagram: np.ndarray) -> np.ndarray:
    """Drop essential (infinite-death) bars."""
    d = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    return d[np.isfinite(d[:, 1])]


def betti_curve(
    diagram: np.ndarray,
    *,
    n_bins: int = 100,
    value_range: tuple[float, float] | None = None,
    include_essential: bool = True,
) -> GlobalDescriptor:
    """Betti curve ``beta(t)`` = number of bars alive at filtration ``t``.

    A bar ``(b, d)`` is alive on ``b <= t < d``; essential bars
    (``d = inf``, e.g. the surviving connected components in ``H_0``) are
    alive for every ``t >= b`` and are counted by default.

    Parameters
    ----------
    diagram : ndarray, shape (n_bars, 2)
    n_bins : int
        Resolution of the sampling grid.
    value_range : (float, float), optional
        Sampling range (from the bars if ``None``: smallest birth to the
        largest finite death/birth).
    include_essential : bool
        Count infinite-death bars (``True``, the mathematically correct
        Betti number).  ``False`` reproduces a finite-bars-only curve.

    Returns
    -------
    ndarray, shape (n_bins,)
    """
    d_all = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    d_all = d_all[np.isfinite(d_all[:, 0])]
    bars = d_all if include_essential else _finite_bars(d_all)
    if bars.shape[0] == 0:
        return np.zeros(n_bins, dtype=np.float64)
    if value_range is None:
        lo = float(bars[:, 0].min())
        finite_d = bars[np.isfinite(bars[:, 1]), 1]
        hi = max([float(bars[:, 0].max())] + ([float(finite_d.max())] if finite_d.size else []))
        if hi <= lo:
            hi = lo + 1.0
        value_range = (lo, hi)
    grid = np.linspace(value_range[0], value_range[1], n_bins)
    curve = np.zeros(n_bins, dtype=np.float64)
    for b, d in bars:
        curve += (grid >= b) & (grid < d)
    return curve


def persistence_landscape(
    diagram: np.ndarray,
    *,
    n_layers: int = 5,
    n_bins: int = 100,
    value_range: tuple[float, float] | None = None,
) -> DescriptorMatrix:
    """Persistence landscape ``lambda_k(t)`` (Bubenik 2015).

    Each bar ``(b, d)`` contributes a tent function
    ``Lambda(t) = max(0, min(t - b, d - t))``; the ``k``-th landscape is the
    ``k``-th largest tent value at each ``t``.  Essential (infinite-death)
    bars have unbounded tents and are excluded.

    Parameters
    ----------
    diagram : ndarray, shape (n_bars, 2)
    n_layers : int
        Number of landscape layers ``k``.
    n_bins : int
        Sampling resolution.
    value_range : (float, float), optional

    Returns
    -------
    ndarray, shape (n_layers, n_bins)
    """
    bars = _finite_bars(diagram)
    out = np.zeros((n_layers, n_bins), dtype=np.float64)
    if bars.shape[0] == 0:
        return out
    if value_range is None:
        value_range = (float(bars[:, 0].min()), float(bars[:, 1].max()))
    grid = np.linspace(value_range[0], value_range[1], n_bins)
    tents = np.zeros((bars.shape[0], n_bins), dtype=np.float64)
    for i, (b, d) in enumerate(bars):
        tents[i] = np.maximum(0.0, np.minimum(grid - b, d - grid))
    tents.sort(axis=0)  # ascending; take from the top
    for k in range(min(n_layers, bars.shape[0])):
        out[k] = tents[-(k + 1)]
    return out


def persistence_statistics(diagram: np.ndarray) -> dict[str, float]:
    """Scalar summaries of a persistence diagram.

    Returns
    -------
    dict
        ``n_bars`` (finite; essential infinite bars are excluded from all
        statistics), ``total_persistence`` (Sigma lifetimes),
        ``max_persistence``, ``mean_persistence``, ``std_persistence`` and
        ``persistence_entropy`` (Shannon entropy of the normalised
        lifetimes).
    """
    bars = _finite_bars(diagram)
    if bars.shape[0] == 0:
        return {
            "n_bars": 0.0,
            "total_persistence": 0.0,
            "max_persistence": 0.0,
            "mean_persistence": 0.0,
            "std_persistence": 0.0,
            "persistence_entropy": 0.0,
        }
    lifetimes = bars[:, 1] - bars[:, 0]
    total = float(lifetimes.sum())
    p = lifetimes / max(total, _EPS)
    entropy = float(-(p * np.log(np.clip(p, _EPS, None))).sum())
    return {
        "n_bars": float(bars.shape[0]),
        "total_persistence": total,
        "max_persistence": float(lifetimes.max()),
        "mean_persistence": float(lifetimes.mean()),
        "std_persistence": float(lifetimes.std()),
        "persistence_entropy": entropy,
    }


def persistence_image(
    diagram: np.ndarray,
    *,
    pixels: tuple[int, int] = (20, 20),
    spread: float = 0.1,
    birth_range: tuple[float, float] | None = None,
    pers_range: tuple[float, float] | None = None,
) -> DescriptorMatrix:
    """Persistence image (Adams et al. 2017).

    Each finite bar is mapped to ``(birth, persistence)`` and replaced by
    an isotropic Gaussian of standard deviation *spread*, weighted linearly
    by its persistence; the surface is integrated exactly over a
    ``pixels[0] x pixels[1]`` grid (birth x persistence).  Implemented
    natively (no ``persim`` dependency); matches persim's default
    linear-weight / Gaussian-CDF construction.

    Parameters
    ----------
    diagram : ndarray, shape (n_bars, 2)
    pixels : (int, int)
        Output resolution along (birth, persistence).
    spread : float
        Gaussian kernel bandwidth (same units as the filtration).
    birth_range, pers_range : (float, float), optional
        Image extent.  ``None`` = fitted to this diagram's bars (padded by
        ``3*spread`` so each Gaussian's mass is captured).  **Pass fixed ranges to compare images across
        diagrams** -- fitted ranges differ per diagram.

    Returns
    -------
    ndarray, shape ``pixels``
        The vectorised persistence surface.
    """
    from scipy.special import ndtr

    bars = _finite_bars(diagram)
    nb, npers = int(pixels[0]), int(pixels[1])
    out = np.zeros((nb, npers), dtype=np.float64)
    if bars.shape[0] == 0:
        return out
    births = bars[:, 0]
    pers = bars[:, 1] - bars[:, 0]
    sig = max(float(spread), _EPS)
    if birth_range is None:
        birth_range = (float(births.min()) - 3 * sig, float(births.max()) + 3 * sig)
    if pers_range is None:
        pers_range = (0.0, float(pers.max()) + 3 * sig)
    bx = np.linspace(birth_range[0], birth_range[1], nb + 1)
    py = np.linspace(pers_range[0], pers_range[1], npers + 1)
    for b, p in zip(births, pers):
        cx = np.diff(ndtr((bx - b) / sig))  # (nb,)
        cy = np.diff(ndtr((py - p) / sig))  # (npers,)
        out += p * np.outer(cx, cy)
    return out


# ======================================================================
# S3  PERSISTENT LAPLACIAN  (q = 0, vertex-subset pair K subset of L)
# ======================================================================


def _to_sparse_adjacency(adjacency: Any) -> sp.csr_matrix:
    """Coerce sparse/dense/networkx input to a symmetric CSR adjacency."""
    try:
        import networkx as nx

        if isinstance(adjacency, nx.Graph):
            a = nx.to_scipy_sparse_array(adjacency, dtype=float, format="csr")
            return sp.csr_matrix(0.5 * (a + a.T))
    except ImportError:
        pass
    a = sp.csr_matrix(adjacency, dtype=np.float64)
    return sp.csr_matrix(0.5 * (a + a.T))


def persistent_laplacian(
    adjacency: Any,
    *,
    subset: np.ndarray | None = None,
) -> SparseMatrix:
    """``q = 0`` persistent Laplacian of a vertex-subset pair ``K subset of L``.

    With ``L`` the full graph and ``K`` the induced subgraph on ``subset``,
    the ``0``-persistent Laplacian is the **Schur complement** of ``L``'s
    graph Laplacian onto the vertices of ``K``,

        Delta_0^{K,L} = L_KK - L_KC L_CC^-1 L_CK,   C = V(L) \\ V(K)

    (Wang, Nguyen & Wei 2020; Memoli, Wan & Wang 2022).  Its kernel
    dimension is the persistent Betti number ``beta_0^{K,L}`` -- the number of
    components of ``L`` that contain vertices of ``K`` (components of ``K``
    merged by paths through ``L``).  ``subset = None`` recovers the
    ordinary graph Laplacian.

    Parameters
    ----------
    adjacency : sparse matrix, ndarray, or networkx.Graph
        Adjacency of ``L`` (weights allowed).
    subset : ndarray of int, optional
        Vertex indices defining ``K`` (all vertices if ``None``).

    Returns
    -------
    sparse matrix
        Symmetric positive semi-definite persistent Laplacian on ``K``
        (generally dense: ``|K|^2`` entries).
    """
    a = _to_sparse_adjacency(adjacency)
    deg = np.asarray(a.sum(axis=1)).ravel()
    L_full = sp.csr_matrix(sp.diags(deg) - a)
    if subset is None:
        return sp.csc_matrix(0.5 * (L_full + L_full.T))

    n = a.shape[0]
    idx = np.unique(np.asarray(subset, dtype=np.int64))
    in_k = np.zeros(n, dtype=bool)
    in_k[idx] = True
    comp_idx = np.where(~in_k)[0]
    L_kk = L_full[idx][:, idx].toarray()
    if comp_idx.size == 0:
        return sp.csc_matrix(0.5 * (L_kk + L_kk.T))

    # Components of the complement that never touch K are decoupled
    # (their L_KC block is zero) and would make L_CC singular: drop them.
    from scipy.sparse.csgraph import connected_components

    a_cc = a[comp_idx][:, comp_idx]
    n_comp, lab = connected_components(a_cc, directed=False)
    touches = np.asarray(abs(a[comp_idx][:, idx]).sum(axis=1)).ravel() > 0
    keep_comp = np.zeros(n_comp, dtype=bool)
    keep_comp[np.unique(lab[touches])] = True
    c_keep = comp_idx[keep_comp[lab]]
    if c_keep.size == 0:
        return sp.csc_matrix(0.5 * (L_kk + L_kk.T))

    from scipy.sparse.linalg import splu

    L_cc = sp.csc_matrix(L_full[c_keep][:, c_keep])
    L_ck = L_full[c_keep][:, idx].toarray()
    X = splu(L_cc).solve(L_ck)  # L_CC^-1 L_CK (grounded -> nonsingular)
    schur = L_kk - L_ck.T @ X
    schur = 0.5 * (schur + schur.T)
    schur[np.abs(schur) < 1e-14 * max(1.0, float(np.abs(schur).max()))] = 0.0
    return sp.csc_matrix(schur)


def persistent_laplacian_spectrum(
    adjacency: Any,
    *,
    subset: np.ndarray | None = None,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Eigendecompose the ``q = 0`` persistent Laplacian.

    The multiplicity of the zero eigenvalue is the persistent Betti number
    ``beta_0^{K,L}``.

    Returns
    -------
    SpectralDecomposition
        ``metadata["operator"] == "persistent_laplacian"``.
    """
    from spectralbrain.spectral.operators._base import operator_eigensystem

    L = persistent_laplacian(adjacency, subset=subset)
    n_sub = L.shape[0]
    return operator_eigensystem(
        L,
        M=None,
        k=min(k, n_sub),
        backend=backend,
        clamp_nonneg=True,
        operator="persistent_laplacian",
        metadata={"domain": "nodes", "n_subset": int(n_sub)},
    )


# ======================================================================
# S4  COMBINATORIAL / PERSISTENT HODGE-DIRAC
# ======================================================================


def _induced_incidence(
    vertices: np.ndarray,
    faces: np.ndarray,
    subset: np.ndarray | None,
) -> tuple[SparseMatrix, SparseMatrix, int, int, int]:
    """Boundary matrices of the subcomplex induced on ``subset`` vertices."""
    from spectralbrain.spectral.operators.topologic import build_incidence

    if subset is None:
        B1, B2, _ = build_incidence(vertices, faces)
        return B1, B2, B1.shape[0], B1.shape[1], B2.shape[1]

    keep = np.zeros(vertices.shape[0], dtype=bool)
    keep[np.asarray(subset, dtype=np.int64)] = True
    fmask = keep[faces].all(axis=1)
    sub_faces = faces[fmask]
    remap = -np.ones(vertices.shape[0], dtype=np.int64)
    idx = np.where(keep)[0]
    remap[idx] = np.arange(idx.shape[0])
    B1, B2, _ = build_incidence(vertices[idx], remap[sub_faces])
    return B1, B2, B1.shape[0], B1.shape[1], B2.shape[1]


def combinatorial_dirac(
    vertices: Vertices,
    faces: Faces,
    *,
    subset: np.ndarray | None = None,
) -> SparseMatrix:
    """Combinatorial Hodge-Dirac operator ``D = d + d^T`` of the mesh.

    Assembles the symmetric block operator on the direct sum of vertex,
    edge and triangle chains,

        D = [[0,   B_1,  0 ],
             [B_1^T, 0,   B_2],
             [0,   B_2^T, 0 ]],

    whose square is block-diagonal with the Hodge Laplacians,
    ``D^2 = L_0 (+) L_1 (+) L_2``.  Consequently the nonzero eigenvalues of ``D``
    are ``+/-sqrt mu`` for each nonzero Hodge eigenvalue ``mu``, and
    ``dim ker D = b_0 + b_1 + b_2`` (the total Betti number / homology
    dimension).  Passing ``subset`` restricts to the induced subcomplex,
    giving a *persistent* Dirac for a vertex-subset pair.

    Parameters
    ----------
    vertices, faces : arrays
    subset : ndarray of int, optional
        Vertex indices of the subcomplex (full complex if ``None``).

    Returns
    -------
    sparse matrix, shape (V+E+F, V+E+F)
        Symmetric (indefinite) Hodge-Dirac operator.
    """
    from spectralbrain.spectral.operators._base import _validate_mesh

    v, f = _validate_mesh(vertices, faces)
    B1, B2, nv, ne, nf = _induced_incidence(v, f, subset)
    total = nv + ne + nf
    D = sp.lil_matrix((total, total), dtype=np.float64)
    # vertex-edge block
    D[0:nv, nv : nv + ne] = B1
    D[nv : nv + ne, 0:nv] = B1.T
    # edge-triangle block
    if nf > 0:
        D[nv : nv + ne, nv + ne : total] = B2
        D[nv + ne : total, nv : nv + ne] = B2.T
    D = D.tocsc()
    return sp.csc_matrix(0.5 * (D + D.T))


def combinatorial_dirac_spectrum(
    vertices: Vertices,
    faces: Faces,
    *,
    subset: np.ndarray | None = None,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Eigendecompose the combinatorial Hodge-Dirac operator.

    Because ``D`` is **indefinite** (spectrum symmetric about 0), the
    eigenpairs nearest zero are returned with the signed spectrum preserved
    (no nonnegativity clamping).  The zero-eigenvalue multiplicity equals
    the total Betti number.

    Returns
    -------
    SpectralDecomposition
        ``eigenvectors`` are per-simplex magnitudes over the chain stack
        ``[vertices | edges | faces]``; ``metadata["operator"] ==
        "hodge_dirac"``.
    """
    from spectralbrain.spectral.operators._base import operator_eigensystem

    D = combinatorial_dirac(vertices, faces, subset=subset)
    n = D.shape[0]
    # D is singular at 0 (the homology kernel); nudge sigma off the exact
    # singularity so the shift-invert factorisation stays well-conditioned
    # while still targeting the smallest-magnitude (topological) eigenvalues.
    return operator_eigensystem(
        D,
        M=None,
        k=min(k, n - 2),
        backend=backend,
        sigma=1e-4,
        clamp_nonneg=False,  # Dirac is indefinite -- keep +/-sqrt mu structure
        operator="hodge_dirac",
        metadata={"domain": "vertices+edges+faces"},
    )


__all__ = [
    "betti_curve",
    "combinatorial_dirac",
    "combinatorial_dirac_spectrum",
    "graph_persistence_h0",
    "persistence_image",
    "persistence_landscape",
    "persistence_statistics",
    "persistent_laplacian",
    "persistent_laplacian_spectrum",
    "vietoris_rips_diagram",
]
