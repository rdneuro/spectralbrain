"""Topology-aware spectral operators: Hodge, Ricci, and bundle Laplacians.

The Laplace–Beltrami operator sees a surface as a smooth Riemannian
manifold; it is blind to *combinatorial topology* (loops, handles, the
connectivity of a connectome) and to *directed* or *bundle-valued* signals.
This module supplies the operators that fill those gaps:

Hodge Laplacians
    The ``k``-form (Hodge) Laplacians ``L_k = d_{k-1} d_{k-1}ᵀ + d_kᵀ d_k``
    built from the mesh's incidence matrices.  ``L_0`` is the graph
    Laplacian; the **1-form Laplacian ``L_1``** has a kernel of dimension
    ``b₁`` (independent loops / handles), making its low spectrum a direct
    topological signature.  :func:`betti_numbers` reads ``b₀, b₁, b₂``
    straight from the incidence ranks.

Discrete Ricci curvature
    Edge-wise **Forman–Ricci** (purely combinatorial, built in) and
    **Ollivier–Ricci** (optimal-transport based) curvature — validated for
    brain *connectomes* (Farooq et al. 2019; Simhal et al. 2020).  Operate
    on a graph/connectome adjacency or, via the mesh-edge graph, on a
    surface.

Bundle / directed Laplacians
    The **magnetic Laplacian** (Hermitian, encodes edge direction via a
    phase ``q``; reduces to the graph Laplacian at ``q = 0``), the
    **connection Laplacian** (tangent-bundle parallel transport,
    Singer–Wu vector diffusion), and the **sheaf Laplacian** (arbitrary
    restriction maps; the previous two are special cases).

Reeb graph
    A Mapper-style discrete Reeb graph of a scalar field on the mesh
    (Singh, Mémoli & Carlsson 2007), summarising its level-set topology.

This module's heavier discrete-topology dependencies (``gudhi``,
``GraphRicciCurvature``, ``POT``/``ot``) are optional and lazily imported;
``networkx`` is part of the ``expanded`` extra.  Pure NumPy/SciPy fallbacks
are provided where practical so the core operators work without them.

References
----------
Lim LH. "Hodge Laplacians on graphs." *SIAM Review* 62(3):685–715, 2020.
Sreejith RP et al. "Forman curvature for complex networks." *J. Stat.
Mech.* 063206, 2016.
Ni CC, Lin YY, Gao J, Gu X. "Community detection on networks with Ricci
flow." *Scientific Reports* 9:9984, 2019.
Singer A, Wu HT. "Vector diffusion maps and the connection Laplacian."
*Comm. Pure Appl. Math.* 65(8):1067–1144, 2012.
Hansen J, Ghrist R. "Toward a spectral theory of cellular sheaves."
*J. Applied and Computational Topology* 3:315–358, 2019.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    Faces,
    ScalarMap,
    SparseMatrix,
    Vertices,
    get_logger,
)

from spectralbrain.expanded._base import (
    BackendSpec,
    _validate_mesh,
    operator_eigensystem,
    require_optional,
)

logger = get_logger(__name__)

_EPS = 1e-12


# ======================================================================
# §1  SIMPLICIAL INCIDENCE MATRICES
# ======================================================================


def _unique_edges(faces: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    """Sorted unique undirected edges and a ``(i<j) → edge-index`` lookup."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    edges = np.unique(e, axis=0)
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(edges)}
    return edges, lookup


def build_incidence(
    vertices: Vertices,
    faces: Faces,
) -> tuple[SparseMatrix, SparseMatrix, np.ndarray]:
    """Boundary (incidence) matrices of the mesh simplicial complex.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    B1 : sparse matrix, shape (N, E)
        Vertex–edge boundary operator ``∂₁`` (oriented ``i < j``):
        column ``e=(i,j)`` has ``-1`` at ``i`` and ``+1`` at ``j``.
    B2 : sparse matrix, shape (E, F)
        Edge–triangle boundary operator ``∂₂`` with orientation signs.
    edges : ndarray, shape (E, 2)
        The sorted unique edges (``i < j``).
    """
    v, f = _validate_mesh(vertices, faces)
    n = v.shape[0]
    edges, lookup = _unique_edges(f)
    n_e = edges.shape[0]
    n_f = f.shape[0]

    # B1 : ∂ of each edge = (+1 at j) + (-1 at i), i < j
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([np.arange(n_e), np.arange(n_e)])
    data = np.concatenate([-np.ones(n_e), np.ones(n_e)])
    B1 = sp.coo_matrix((data, (rows, cols)), shape=(n, n_e)).tocsc()

    # B2 : ∂ of each triangle = [a,b] + [b,c] + [c,a] with sign per edge
    er: list[int] = []
    ec: list[int] = []
    ed: list[float] = []
    for fi in range(n_f):
        a, b, c = (int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2]))
        for (x, y) in ((a, b), (b, c), (c, a)):
            key = (x, y) if x < y else (y, x)
            sign = 1.0 if x < y else -1.0
            er.append(lookup[key])
            ec.append(fi)
            ed.append(sign)
    B2 = sp.coo_matrix(
        (np.asarray(ed), (np.asarray(er), np.asarray(ec))), shape=(n_e, n_f)
    ).tocsc()
    return B1, B2, edges


# ======================================================================
# §2  HODGE LAPLACIANS + BETTI NUMBERS
# ======================================================================


def hodge_laplacian(
    vertices: Vertices,
    faces: Faces,
    *,
    order: Literal[0, 1, 2] = 1,
) -> SparseMatrix:
    """Combinatorial Hodge ``k``-Laplacian of the mesh.

    Parameters
    ----------
    vertices, faces : arrays
    order : {0, 1, 2}
        ``0`` → vertex (graph) Laplacian ``B₁ B₁ᵀ``;
        ``1`` → edge/1-form Laplacian ``B₁ᵀ B₁ + B₂ B₂ᵀ`` (kernel dim = b₁);
        ``2`` → triangle Laplacian ``B₂ᵀ B₂``.

    Returns
    -------
    sparse matrix
        Symmetric PSD operator on the requested simplex dimension.
    """
    B1, B2, _ = build_incidence(vertices, faces)
    if order == 0:
        L = B1 @ B1.T
    elif order == 1:
        L = (B1.T @ B1) + (B2 @ B2.T)
    elif order == 2:
        L = B2.T @ B2
    else:
        raise ValueError(f"order must be 0, 1 or 2, got {order}.")
    return sp.csc_matrix(0.5 * (L + L.T))


def hodge_decompose(
    vertices: Vertices,
    faces: Faces,
    *,
    order: Literal[0, 1, 2] = 1,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Eigendecompose a Hodge ``k``-Laplacian.

    The smallest eigenvalues (≈ 0) span the harmonic ``k``-forms, whose
    count is the Betti number ``b_k``.

    Parameters
    ----------
    vertices, faces : arrays
    order : {0, 1, 2}
    k : int
        Number of eigenpairs.
    backend : str or backend object
        Multi-backend selector.

    Returns
    -------
    SpectralDecomposition
        ``eigenvectors`` live on vertices (order 0), edges (order 1) or
        triangles (order 2); ``metadata["operator"] == f"hodge{order}"`` and
        ``metadata["domain"]`` records the simplex dimension.
    """
    L = hodge_laplacian(vertices, faces, order=order)
    domain = {0: "vertices", 1: "edges", 2: "faces"}[order]
    return operator_eigensystem(
        L,
        M=None,
        k=k,
        backend=backend,
        sigma=-0.01,
        which="LM",
        clamp_nonneg=True,
        operator=f"hodge{order}",
        metadata={"domain": domain, "hodge_order": int(order)},
    )


def betti_numbers(
    vertices: Vertices,
    faces: Faces,
    *,
    tol: float | None = None,
) -> tuple[int, int, int]:
    """Betti numbers ``(b₀, b₁, b₂)`` from incidence ranks.

    Uses the rank–nullity identities of the chain complex:
    ``b₀ = N − rank(B₁)``, ``b₂ = F − rank(B₂)``,
    ``b₁ = E − rank(B₁) − rank(B₂)``.

    Parameters
    ----------
    vertices, faces : arrays
    tol : float, optional
        Singular-value threshold for the numerical rank (auto if ``None``).

    Returns
    -------
    (b0, b1, b2) : ints
        ``b₀`` connected components, ``b₁`` independent loops/handles,
        ``b₂`` enclosed voids.

    Notes
    -----
    Exact for moderate meshes via dense SVD of the incidence matrices.  For
    very large surfaces prefer counting near-zero eigenvalues of
    :func:`hodge_laplacian` instead.
    """
    v, f = _validate_mesh(vertices, faces)
    B1, B2, _ = build_incidence(v, f)
    n, n_e = B1.shape
    n_f = B2.shape[1]

    def _rank(mat: SparseMatrix) -> int:
        a = mat.toarray()
        if a.size == 0 or min(a.shape) == 0:
            return 0
        s = np.linalg.svd(a, compute_uv=False)
        thr = tol if tol is not None else max(a.shape) * np.finfo(float).eps * s.max()
        return int((s > thr).sum())

    r1 = _rank(B1)
    r2 = _rank(B2)
    b0 = n - r1
    b1 = n_e - r1 - r2
    b2 = n_f - r2
    return int(b0), int(b1), int(b2)


# ======================================================================
# §3  DISCRETE RICCI CURVATURE  (Forman + Ollivier)
# ======================================================================


def _to_networkx(adjacency: Any) -> Any:
    """Coerce a sparse/dense adjacency or networkx graph to a networkx graph."""
    nx = require_optional("networkx", purpose="discrete Ricci curvature")
    if isinstance(adjacency, nx.Graph):
        return adjacency
    A = sp.csr_matrix(adjacency)
    return nx.from_scipy_sparse_array(A)


def forman_ricci_curvature(
    adjacency: Any,
) -> tuple[ScalarMap, np.ndarray, ScalarMap]:
    """Combinatorial Forman–Ricci curvature of a graph/connectome.

    Uses the unweighted Forman curvature of Sreejith et al. (2016):
    for an edge ``e = (u, v)``,

        Ric_F(e) = 4 − deg(u) − deg(v).

    Highly negative on hub-to-hub edges, near zero on chain-like edges.

    Parameters
    ----------
    adjacency : sparse matrix, ndarray, or networkx.Graph
        The (undirected) graph; for a connectome, a binary/weighted ROI×ROI
        adjacency.

    Returns
    -------
    node_curvature : ndarray, shape (N,)
        Per-node curvature (sum of incident edge curvatures).
    edge_index : ndarray, shape (E, 2)
        Edge endpoint indices.
    edge_curvature : ndarray, shape (E,)
        Per-edge Forman curvature.
    """
    g = _to_networkx(adjacency)
    deg = dict(g.degree())
    n = g.number_of_nodes()
    edge_index: list[tuple[int, int]] = []
    edge_curv: list[float] = []
    node_curv = np.zeros(n, dtype=np.float64)
    for u, v in g.edges():
        fr = 4.0 - deg[u] - deg[v]
        edge_index.append((int(u), int(v)))
        edge_curv.append(fr)
        node_curv[int(u)] += fr
        node_curv[int(v)] += fr
    return node_curv, np.asarray(edge_index, dtype=np.int64), np.asarray(edge_curv)


def _emd_linprog(
    p: np.ndarray,
    q: np.ndarray,
    cost: np.ndarray,
) -> float:
    """Exact 1-Wasserstein (EMD) between two histograms via linear programming.

    Solves ``min_T ⟨T, C⟩`` s.t. ``T·1 = p``, ``Tᵀ·1 = q``, ``T ≥ 0``.
    Intended for the small neighbourhood supports of Ollivier–Ricci.
    """
    from scipy.optimize import linprog

    m, k = cost.shape
    c = cost.reshape(-1)
    # row marginals (m constraints) + column marginals (k constraints)
    a_eq = np.zeros((m + k, m * k))
    for i in range(m):
        a_eq[i, i * k:(i + 1) * k] = 1.0
    for j in range(k):
        a_eq[m + j, j::k] = 1.0
    b_eq = np.concatenate([p, q])
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return float(res.fun) if res.success else float("nan")


def _ollivier_builtin(g: Any, alpha: float) -> tuple[list[tuple[int, int]], list[float]]:
    """Builtin Ollivier–Ricci via exact local EMD (no external OT dependency)."""
    import networkx as nx

    def _walk(x: int) -> dict[int, float]:
        nbrs = list(g.neighbors(x))
        if not nbrs:
            return {x: 1.0}
        dist = {x: alpha}
        w = (1.0 - alpha) / len(nbrs)
        for nb in nbrs:
            dist[nb] = dist.get(nb, 0.0) + w
        return dist

    edge_index: list[tuple[int, int]] = []
    edge_curv: list[float] = []
    for u, v in g.edges():
        mu, mv = _walk(u), _walk(v)
        su, sv = list(mu), list(mv)
        # ground distances between supports (BFS shortest paths)
        cost = np.zeros((len(su), len(sv)))
        for i, a in enumerate(su):
            lengths = nx.single_source_shortest_path_length(g, a, cutoff=4)
            for j, b in enumerate(sv):
                cost[i, j] = lengths.get(b, 5)
        w1 = _emd_linprog(
            np.array([mu[x] for x in su]),
            np.array([mv[x] for x in sv]),
            cost,
        )
        d_uv = 1.0  # adjacent
        edge_index.append((int(u), int(v)))
        edge_curv.append(1.0 - w1 / d_uv)
    return edge_index, edge_curv


def ollivier_ricci_curvature(
    adjacency: Any,
    *,
    alpha: float = 0.5,
    method: Literal["auto", "graphriccicurvature", "builtin"] = "auto",
) -> tuple[ScalarMap, np.ndarray, ScalarMap]:
    """Ollivier–Ricci curvature of a graph/connectome.

    Edge curvature ``κ(u, v) = 1 − W₁(m_u, m_v) / d(u, v)``, where ``m_x`` is
    the lazy random-walk distribution (mass ``alpha`` at ``x``, the rest
    uniform over its neighbours) and ``W₁`` is the 1-Wasserstein distance.
    Positive on well-connected (community-internal) edges, negative on
    bridges — the basis of Ricci-flow community detection and the
    connectome studies of Farooq et al. (2019).

    Parameters
    ----------
    adjacency : sparse matrix, ndarray, or networkx.Graph
    alpha : float
        Laziness of the random walk (mass retained at the source).
    method : str
        ``"graphriccicurvature"`` uses the optimised ``GraphRicciCurvature``
        package; ``"builtin"`` uses an exact local LP solver (no extra
        dependency); ``"auto"`` prefers the package and falls back to
        builtin.

    Returns
    -------
    node_curvature : ndarray, shape (N,)
    edge_index : ndarray, shape (E, 2)
    edge_curvature : ndarray, shape (E,)
    """
    g = _to_networkx(adjacency)
    n = g.number_of_nodes()

    use_pkg = method in ("auto", "graphriccicurvature")
    if use_pkg:
        try:
            from GraphRicciCurvature.OllivierRicci import OllivierRicci

            orc = OllivierRicci(g, alpha=alpha, verbose="ERROR")
            orc.compute_ricci_curvature()
            ei: list[tuple[int, int]] = []
            ec: list[float] = []
            for u, v, data in orc.G.edges(data=True):
                ei.append((int(u), int(v)))
                ec.append(float(data["ricciCurvature"]))
            edge_index, edge_curv = ei, ec
        except ImportError:
            if method == "graphriccicurvature":
                raise
            edge_index, edge_curv = _ollivier_builtin(g, alpha)
    else:
        edge_index, edge_curv = _ollivier_builtin(g, alpha)

    node_curv = np.zeros(n, dtype=np.float64)
    deg = dict(g.degree())
    for (u, v), kappa in zip(edge_index, edge_curv):
        if deg[u]:
            node_curv[u] += kappa / deg[u]
        if deg[v]:
            node_curv[v] += kappa / deg[v]
    return node_curv, np.asarray(edge_index, dtype=np.int64), np.asarray(edge_curv)


def _mesh_edge_graph(vertices: np.ndarray, faces: np.ndarray) -> Any:
    """Build the mesh's 1-skeleton as a networkx graph (edge lengths as weight)."""
    nx = require_optional("networkx", purpose="mesh Ricci curvature")
    _, _, edges = build_incidence(vertices, faces)
    g = nx.Graph()
    g.add_nodes_from(range(vertices.shape[0]))
    for a, b in edges:
        g.add_edge(int(a), int(b))
    return g


def ricci_curvature_mesh(
    vertices: Vertices,
    faces: Faces,
    *,
    kind: Literal["forman", "ollivier"] = "forman",
    alpha: float = 0.5,
) -> ScalarMap:
    """Per-vertex discrete Ricci curvature of a surface's 1-skeleton.

    Parameters
    ----------
    vertices, faces : arrays
    kind : str
        ``"forman"`` (fast, combinatorial) or ``"ollivier"`` (OT-based).
    alpha : float
        Random-walk laziness (Ollivier only).

    Returns
    -------
    ndarray, shape (N,)
        Per-vertex curvature descriptor.
    """
    v, f = _validate_mesh(vertices, faces)
    g = _mesh_edge_graph(v, f)
    if kind == "forman":
        node_curv, _, _ = forman_ricci_curvature(g)
    elif kind == "ollivier":
        node_curv, _, _ = ollivier_ricci_curvature(g, alpha=alpha)
    else:
        raise ValueError(f"kind must be 'forman' or 'ollivier', got {kind!r}.")
    return node_curv


# ======================================================================
# §4  BUNDLE / DIRECTED LAPLACIANS  (magnetic, connection, sheaf)
# ======================================================================


def _hermitian_eigsh(
    H: SparseMatrix,
    k: int,
    *,
    backend: BackendSpec | Any,
    sigma: float = -0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest-k eigenpairs of a complex Hermitian PSD operator (host arrays).

    CPU path uses SciPy ARPACK directly on the complex Hermitian matrix.
    GPU backends use the real ``2N`` representation ``[[Re,-Im],[Im,Re]]``
    (each eigenvalue doubled) and collapse the pairs.  Returns eigenvalues
    and the **per-node magnitudes** ``|ψ|`` (real, ≥ 0).
    """
    from spectralbrain.expanded._base import resolve_backend, solve_eigsh

    be = resolve_backend(backend)
    name = getattr(be, "name", "numpy")
    n = H.shape[0]

    if name in ("numpy", "scipy", "cpu"):
        import scipy.sparse.linalg as spla

        Hc = sp.csc_matrix(H, dtype=np.complex128)
        evals, evecs = spla.eigsh(Hc, k=min(k, n - 2), sigma=sigma, which="LM")
        order = np.argsort(evals.real)
        evals = np.clip(evals.real[order], 0.0, None)
        mag = np.abs(evecs[:, order])
        return evals, mag

    # GPU: real 2N representation, solve, collapse doubled spectrum.
    Re = sp.csc_matrix(H.real, dtype=np.float64)
    Im = sp.csc_matrix(H.imag, dtype=np.float64)
    top = sp.hstack([Re, -Im])
    bot = sp.hstack([Im, Re])
    H2 = sp.vstack([top, bot]).tocsc()
    H2 = 0.5 * (H2 + H2.T)
    evals2, evecs2 = solve_eigsh(
        be, H2, None, min(2 * k, 2 * n - 2), sigma=sigma, clamp_nonneg=True
    )
    evals = evals2[::2][:k]
    re = evecs2[:n, ::2][:, :k]
    im = evecs2[n:, ::2][:, :k]
    mag = np.sqrt(re**2 + im**2)
    return evals, mag


def magnetic_laplacian(
    adjacency: Any,
    *,
    q: float = 0.25,
    normalized: bool = False,
) -> SparseMatrix:
    """Hermitian magnetic Laplacian of a (possibly directed) graph.

    Encodes edge direction in a phase: with the symmetric magnitude
    ``Aₛ = (A + Aᵀ)/2`` and antisymmetric flow ``Θ = 2πq (A − Aᵀ)``,

        L^{(q)} = D − Aₛ ∘ exp(i Θ).

    At ``q = 0`` this reduces to the standard (real) graph Laplacian; a
    nonzero ``q`` makes directed cycles spectrally visible.

    Parameters
    ----------
    adjacency : sparse matrix or ndarray, shape (N, N)
        Graph adjacency (directed allowed).
    q : float
        Magnetic charge / potential.
    normalized : bool
        If ``True``, symmetric-normalise: ``D^{-1/2} L D^{-1/2}``.

    Returns
    -------
    sparse matrix (complex), shape (N, N)
        Hermitian, positive semi-definite.
    """
    A = sp.csr_matrix(adjacency, dtype=np.float64)
    As = 0.5 * (A + A.T)
    flow = A - A.T
    theta = (2.0 * np.pi * q) * flow
    # Hermitian off-diagonal: Aₛ ∘ exp(iΘ)
    As_coo = As.tocoo()
    phase = np.exp(1j * np.asarray(theta[As_coo.row, As_coo.col]).ravel())
    H_off = sp.coo_matrix(
        (As_coo.data * phase, (As_coo.row, As_coo.col)), shape=A.shape
    ).tocsr()
    deg = np.asarray(As.sum(axis=1)).ravel()
    L = sp.diags(deg).astype(np.complex128) - H_off
    if normalized:
        dis = 1.0 / np.sqrt(np.clip(deg, _EPS, None))
        L = sp.diags(dis) @ L @ sp.diags(dis)
    return sp.csc_matrix(0.5 * (L + L.getH()))


def magnetic_decompose(
    adjacency: Any,
    *,
    q: float = 0.25,
    k: int = 100,
    normalized: bool = False,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Eigendecompose the magnetic Laplacian (k smallest eigenvalues).

    Returns
    -------
    SpectralDecomposition
        ``eigenvectors`` are per-node magnitudes ``|ψ|`` (N, k);
        ``metadata["operator"] == "magnetic"`` and records ``q``.
    """
    H = magnetic_laplacian(adjacency, q=q, normalized=normalized)
    n = H.shape[0]
    evals, mag = _hermitian_eigsh(H, min(k, n - 2), backend=backend)
    return SpectralDecomposition(
        eigenvalues=evals,
        eigenvectors=mag,
        stiffness=None,
        mass=None,
        metadata={"operator": "magnetic", "q": float(q), "domain": "nodes"},
    )


def _vertex_tangent_frames(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex unit normal and an orthonormal tangent basis (t1, t2)."""
    from spectralbrain.expanded._base import vertex_normals

    n = vertex_normals(vertices, faces)
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (vertices.shape[0], 1))
    ref[np.abs(n[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, n)
    t1 /= np.clip(np.linalg.norm(t1, axis=1, keepdims=True), _EPS, None)
    t2 = np.cross(n, t1)
    return n, t1, t2


def connection_laplacian(
    vertices: Vertices,
    faces: Faces,
) -> SparseMatrix:
    """Tangent-bundle connection Laplacian (Singer–Wu vector diffusion).

    Builds the block operator on the rank-2 tangent bundle: diagonal blocks
    are the vertex degrees ``deg_i · I₂`` and off-diagonal blocks are
    ``-O_ij`` where ``O_ij ∈ O(2)`` parallel-transports a tangent vector
    from vertex ``j`` to vertex ``i`` (frames aligned by rotating ``n_j``
    onto ``n_i``).  On a flat patch all ``O_ij = I₂`` and the operator
    reduces to the graph Laplacian ⊗ ``I₂``.

    Parameters
    ----------
    vertices, faces : arrays

    Returns
    -------
    sparse matrix, shape (2N, 2N)
        Symmetric positive semi-definite tangent-bundle Laplacian.
    """
    v, f = _validate_mesh(vertices, faces)
    n_v = v.shape[0]
    nrm, t1, t2 = _vertex_tangent_frames(v, f)
    _, _, edges = build_incidence(v, f)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    deg = np.zeros(n_v, dtype=np.float64)

    def _transport(j: int, i: int) -> np.ndarray:
        """2×2 O(2) map taking j's tangent frame into i's frame."""
        nj, ni = nrm[j], nrm[i]
        axis = np.cross(nj, ni)
        s = np.linalg.norm(axis)
        c = float(np.dot(nj, ni))
        if s < _EPS:  # already aligned (or antiparallel)
            t1j, t2j = t1[j], t2[j]
        else:
            axis = axis / s
            # Rodrigues rotation of j's frame onto i's tangent plane
            def rot(x: np.ndarray) -> np.ndarray:
                return (
                    x * c
                    + np.cross(axis, x) * s
                    + axis * np.dot(axis, x) * (1 - c)
                )

            t1j, t2j = rot(t1[j]), rot(t2[j])
        o = np.array(
            [
                [np.dot(t1[i], t1j), np.dot(t1[i], t2j)],
                [np.dot(t2[i], t1j), np.dot(t2[i], t2j)],
            ]
        )
        return o

    for a, b in edges:
        a, b = int(a), int(b)
        deg[a] += 1.0
        deg[b] += 1.0
        o_ab = _transport(b, a)  # j=b → i=a
        # off-diagonal block (a,b) = -O_ab ; (b,a) = -O_abᵀ (symmetry)
        for r in range(2):
            for c2 in range(2):
                rows.append(2 * a + r); cols.append(2 * b + c2); data.append(-o_ab[r, c2])
                rows.append(2 * b + c2); cols.append(2 * a + r); data.append(-o_ab[r, c2])
    S = sp.coo_matrix((data, (rows, cols)), shape=(2 * n_v, 2 * n_v)).tocsc()
    # diagonal blocks deg_i · I2
    diag = np.repeat(deg, 2)
    S = S + sp.diags(diag, format="csc")
    return sp.csc_matrix(0.5 * (S + S.T))


def connection_decompose(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Eigendecompose the tangent-bundle connection Laplacian.

    Returns
    -------
    SpectralDecomposition
        ``eigenvectors`` are per-vertex 2-vector magnitudes (N, k);
        ``metadata["operator"] == "connection"``.
    """
    v, f = _validate_mesh(vertices, faces)
    S = connection_laplacian(v, f)
    dec = operator_eigensystem(
        S, M=None, k=k, backend=backend, clamp_nonneg=True, operator="connection"
    )
    # collapse the 2-vector field to per-vertex magnitudes for descriptors
    n_v = v.shape[0]
    blocks = dec.eigenvectors.reshape(n_v, 2, -1)
    mag = np.sqrt(np.sum(blocks**2, axis=1))  # (N, k)
    return SpectralDecomposition(
        eigenvalues=dec.eigenvalues,
        eigenvectors=mag,
        stiffness=None,
        mass=None,
        metadata={**dec.metadata, "operator": "connection", "domain": "tangent_bundle"},
    )


def sheaf_laplacian(
    n_nodes: int,
    edges: np.ndarray,
    restriction_maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    *,
    stalk_dim: int = 1,
) -> SparseMatrix:
    """Cellular sheaf Laplacian from per-edge restriction maps.

    For each edge ``e = (u, v)`` with restriction maps ``F_{u◁e}`` and
    ``F_{v◁e}`` (each ``d_e × stalk_dim``), the sheaf Laplacian has blocks

        L[u][u] += F_{u◁e}ᵀ F_{u◁e},   L[u][v] = −F_{u◁e}ᵀ F_{v◁e}.

    With ``stalk_dim = 1`` and identity restrictions this is exactly the
    graph Laplacian; with 2×2 rotations it is the connection Laplacian —
    i.e. this is the common generalisation of both.

    Parameters
    ----------
    n_nodes : int
        Number of vertices.
    edges : ndarray, shape (E, 2)
        Edge endpoint indices.
    restriction_maps : dict
        ``(u, v) → (F_u, F_v)`` restriction matrices.
    stalk_dim : int
        Dimension of each vertex stalk ``d``.

    Returns
    -------
    sparse matrix, shape (N·d, N·d)
        Symmetric positive semi-definite sheaf Laplacian.
    """
    d = stalk_dim
    diag_blocks: dict[int, np.ndarray] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def _add_block(bi: int, bj: int, block: np.ndarray) -> None:
        for r in range(d):
            for c in range(d):
                rows.append(bi * d + r)
                cols.append(bj * d + c)
                data.append(float(block[r, c]))

    for u, v in edges:
        u, v = int(u), int(v)
        fu, fv = restriction_maps[(u, v)]
        fu = np.asarray(fu, dtype=np.float64).reshape(-1, d)
        fv = np.asarray(fv, dtype=np.float64).reshape(-1, d)
        diag_blocks[u] = diag_blocks.get(u, np.zeros((d, d))) + fu.T @ fu
        diag_blocks[v] = diag_blocks.get(v, np.zeros((d, d))) + fv.T @ fv
        off = -(fu.T @ fv)
        _add_block(u, v, off)
        _add_block(v, u, off.T)

    for node, block in diag_blocks.items():
        _add_block(node, node, block)

    L = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes * d, n_nodes * d)).tocsc()
    return sp.csc_matrix(0.5 * (L + L.T))


# ======================================================================
# §5  REEB GRAPH  (Mapper-style level-set topology)
# ======================================================================


def reeb_graph(
    vertices: Vertices,
    faces: Faces,
    scalar: ScalarMap,
    *,
    n_levels: int = 20,
    overlap: float = 0.15,
) -> Any:
    """Mapper-style discrete Reeb graph of a scalar field on the mesh.

    Bins the scalar range into overlapping level sets, extracts the
    connected components of the mesh restricted to each bin (the Mapper
    nodes), and links components in adjacent bins that share vertices.  The
    result summarises the level-set topology of ``scalar`` — its loops track
    handles seen by the function (Singh, Mémoli & Carlsson 2007).

    Parameters
    ----------
    vertices, faces : arrays
    scalar : ndarray, shape (N,)
        The Morse function (e.g. a coordinate, geodesic distance, or an LBO
        eigenfunction).
    n_levels : int
        Number of level-set bins.
    overlap : float
        Fractional overlap between adjacent bins (Mapper gain).

    Returns
    -------
    networkx.Graph
        Nodes carry ``bin``, ``size`` and ``centroid`` attributes.
    """
    nx = require_optional("networkx", purpose="Reeb graph")
    v, f = _validate_mesh(vertices, faces)
    s = np.asarray(scalar, dtype=np.float64).ravel()
    if s.shape[0] != v.shape[0]:
        raise ValueError(f"scalar length {s.shape[0]} != n_vertices {v.shape[0]}.")

    _, _, edges = build_incidence(v, f)
    adj = nx.Graph()
    adj.add_nodes_from(range(v.shape[0]))
    adj.add_edges_from((int(a), int(b)) for a, b in edges)

    fmin, fmax = float(s.min()), float(s.max())
    step = (fmax - fmin) / max(n_levels, 1)
    if step < _EPS:
        step = 1.0

    g = nx.Graph()
    members: dict[int, set[int]] = {}
    by_bin: dict[int, list[int]] = {}
    nid = 0
    for b in range(n_levels):
        lo = fmin + b * step - overlap * step
        hi = fmin + (b + 1) * step + overlap * step
        sel = np.where((s >= lo) & (s <= hi))[0]
        if sel.size == 0:
            continue
        for comp in nx.connected_components(adj.subgraph(sel.tolist())):
            comp = set(comp)
            g.add_node(
                nid,
                bin=int(b),
                size=len(comp),
                centroid=v[list(comp)].mean(axis=0),
            )
            members[nid] = comp
            by_bin.setdefault(b, []).append(nid)
            nid += 1

    for b in range(n_levels - 1):
        for ni in by_bin.get(b, []):
            for nj in by_bin.get(b + 1, []):
                if members[ni] & members[nj]:
                    g.add_edge(ni, nj)
    return g


def reeb_graph_features(reeb: Any) -> dict[str, int]:
    """Summary topology of a Reeb graph.

    Parameters
    ----------
    reeb : networkx.Graph
        Output of :func:`reeb_graph`.

    Returns
    -------
    dict
        ``n_nodes``, ``n_edges``, ``n_components``, ``loops`` (first Betti
        number of the Reeb graph = handles seen by the function),
        ``branches`` (degree ≥ 3 saddles) and ``leaves`` (degree 1 extrema).
    """
    nx = require_optional("networkx", purpose="Reeb graph features")
    n_nodes = reeb.number_of_nodes()
    n_edges = reeb.number_of_edges()
    comps = nx.number_connected_components(reeb) if n_nodes else 0
    loops = n_edges - n_nodes + comps
    degrees = [d for _, d in reeb.degree()]
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_components": int(comps),
        "loops": int(loops),
        "branches": int(sum(1 for d in degrees if d >= 3)),
        "leaves": int(sum(1 for d in degrees if d == 1)),
    }


__all__ = [
    # incidence + Hodge
    "build_incidence",
    "hodge_laplacian",
    "hodge_decompose",
    "betti_numbers",
    # Ricci
    "forman_ricci_curvature",
    "ollivier_ricci_curvature",
    "ricci_curvature_mesh",
    # bundle / directed
    "magnetic_laplacian",
    "magnetic_decompose",
    "connection_laplacian",
    "connection_decompose",
    "sheaf_laplacian",
    # Reeb
    "reeb_graph",
    "reeb_graph_features",
]
