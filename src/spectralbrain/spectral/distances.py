"""Spectral distance metrics between shapes and between points.

Two categories of distance:

**Shape-to-shape** — compare two *different* shapes via their
eigenvalue spectra (no point correspondence needed):

- :func:`wesd` — Weighted Spectral Distance (Konukoglu et al. 2013)
- :func:`shapedna_distance` — Euclidean / Mahalanobis on ShapeDNA

**Point-to-point** — distances *within* a single shape, computed
from the eigenpairs of that shape:

- :func:`biharmonic_distance` — Lipman, Rustamov & Funkhouser 2010
- :func:`commute_time_distance` — random-walk commute time
- :func:`diffusion_distance` — Coifman & Lafon 2006

All point-to-point distances share a unifying form:

.. math::

    d^2(x, y) = \\sum_i g(\\lambda_i)\\,
                (\\varphi_i(x) - \\varphi_i(y))^2

where g(λ) is a spectral filter that defines the metric.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DistanceMatrix,
    GlobalDescriptor,
    get_logger,
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# §1  UNIFIED SPECTRAL DISTANCE KERNEL
# ======================================================================

def _spectral_distance_matrix(
    eigenvectors: np.ndarray,
    weights: np.ndarray,
    *,
    indices: Optional[np.ndarray] = None,
) -> DistanceMatrix:
    """Compute pairwise spectral distance matrix.

    d²(x, y) = Σᵢ wᵢ · (φᵢ(x) − φᵢ(y))²

    This is equivalent to the Euclidean distance in the
    weight-scaled eigenfunction embedding.

    Parameters
    ----------
    eigenvectors : ndarray, shape (N, k)
    weights : ndarray, shape (k,)
        Spectral filter weights g(λᵢ).
    indices : ndarray of int, optional
        If given, compute distances only from these source vertices
        to all others.  Returns shape (len(indices), N).

    Returns
    -------
    ndarray, shape (M, N) or (N, N)
    """
    # Weighted embedding: Ψ(x) = φ(x) · √w
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))          # (k,)
    embedding = eigenvectors * sqrt_w[None, :]              # (N, k)

    if indices is not None:
        sources = embedding[indices]                        # (M, k)
        # Squared Euclidean: ||a-b||² = ||a||² + ||b||² - 2·a·b
        sq_src = np.sum(sources ** 2, axis=1, keepdims=True)  # (M, 1)
        sq_all = np.sum(embedding ** 2, axis=1, keepdims=True).T  # (1, N)
        cross = sources @ embedding.T                       # (M, N)
        D2 = sq_src + sq_all - 2 * cross                    # (M, N)
    else:
        sq = np.sum(embedding ** 2, axis=1)                 # (N,)
        cross = embedding @ embedding.T                     # (N, N)
        D2 = sq[:, None] + sq[None, :] - 2 * cross         # (N, N)

    return np.sqrt(np.clip(D2, 0.0, None))


# ======================================================================
# §2  SHAPE-TO-SHAPE DISTANCES
# ======================================================================

def wesd(
    dna_a: GlobalDescriptor,
    dna_b: GlobalDescriptor,
    *,
    p: float = 2.0,
    normalize: bool = True,
) -> float:
    """Weighted Spectral Distance between two ShapeDNA vectors.

    A pseudometric with convergence guarantees — the series
    converges for p > d/2 where d is the dimension of the manifold
    (d = 2 for surfaces, so p > 1 suffices).

    .. math::

        \\text{WESD}^p(\\Omega_1, \\Omega_2) =
        \\left(
            \\sum_{i=1}^{k}
            \\frac{|\\lambda_i^{(1)} - \\lambda_i^{(2)}|}
                  {\\lambda_i^{(1)} \\cdot \\lambda_i^{(2)}}
        \\right)^{1/p}

    Parameters
    ----------
    dna_a, dna_b : ndarray, shape (d,)
        ShapeDNA eigenvalue sequences (skip λ₀).
    p : float
        Exponent (must be > 1 for 2D surfaces).
    normalize : bool
        Map to [0, 1) via WESD / (1 + WESD).

    Returns
    -------
    float
        WESD distance.

    References
    ----------
    Konukoglu E, Glocker B, Criminisi A, Pohl KM. WESD — Weighted
    Spectral Distance for measuring shape dissimilarity. *IEEE TPAMI*
    35(9):2284–2297, 2013.
    """
    a = np.asarray(dna_a, dtype=np.float64)
    b = np.asarray(dna_b, dtype=np.float64)

    # Truncate to common length.
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    # Both must be positive (eigenvalues with λ₀ removed).
    a = np.clip(a, 1e-20, None)
    b = np.clip(b, 1e-20, None)

    terms = np.abs(a - b) / (a * b)
    raw = np.sum(terms ** p) ** (1.0 / p)

    if normalize:
        return float(raw / (1.0 + raw))
    return float(raw)


def wesd_matrix(
    dna_collection: np.ndarray,
    *,
    p: float = 2.0,
    normalize: bool = True,
) -> DistanceMatrix:
    """Pairwise WESD matrix for a collection of ShapeDNA vectors.

    Parameters
    ----------
    dna_collection : ndarray, shape (S, d)
        S shapes, each with d-dimensional ShapeDNA.
    p : float
    normalize : bool

    Returns
    -------
    ndarray, shape (S, S)
        Symmetric WESD distance matrix.
    """
    S = dna_collection.shape[0]
    D = np.zeros((S, S), dtype=np.float64)

    total_pairs = S * (S - 1) // 2
    with progress_simple("WESD matrix", total=total_pairs) as tick:
        for i in range(S):
            for j in range(i + 1, S):
                d = wesd(
                    dna_collection[i], dna_collection[j],
                    p=p, normalize=normalize,
                )
                D[i, j] = d
                D[j, i] = d
                tick(1)

    return D


def shapedna_distance(
    dna_a: GlobalDescriptor,
    dna_b: GlobalDescriptor,
    *,
    metric: Literal["euclidean", "mahalanobis", "cosine"] = "euclidean",
    cov_inv: Optional[np.ndarray] = None,
) -> float:
    """Simple distance between two ShapeDNA vectors.

    Parameters
    ----------
    dna_a, dna_b : ndarray, shape (d,)
    metric : str
        ``"euclidean"`` — L2 distance.
        ``"mahalanobis"`` — requires *cov_inv*.
        ``"cosine"`` — 1 − cos(a, b).
    cov_inv : ndarray, shape (d, d), optional
        Inverse covariance matrix for Mahalanobis.

    Returns
    -------
    float
    """
    a = np.asarray(dna_a, dtype=np.float64)
    b = np.asarray(dna_b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    elif metric == "mahalanobis":
        if cov_inv is None:
            raise ValueError("cov_inv required for Mahalanobis distance.")
        diff = a - b
        return float(np.sqrt(diff @ cov_inv[:n, :n] @ diff))
    elif metric == "cosine":
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-20 or nb < 1e-20:
            return 1.0
        return float(1.0 - dot / (na * nb))
    else:
        raise ValueError(f"Unknown metric: {metric!r}")


# ======================================================================
# §3  BIHARMONIC DISTANCE  (Lipman, Rustamov & Funkhouser, 2010)
# ======================================================================

def biharmonic_distance(
    decomp: SpectralDecomposition,
    *,
    indices: Optional[np.ndarray] = None,
) -> DistanceMatrix:
    """Biharmonic distance — parameter-free intrinsic metric.

    .. math::

        d_B^2(x, y) = \\sum_{i=1}^{k}
            \\frac{(\\varphi_i(x) - \\varphi_i(y))^2}{\\lambda_i^2}

    Smooth, locally isotropic, globally shape-aware, robust to
    topological noise.  No tuneable parameters.

    Parameters
    ----------
    decomp : SpectralDecomposition
    indices : ndarray of int, optional
        Compute distances only from these source vertices.

    Returns
    -------
    ndarray, shape (M, N) or (N, N)

    References
    ----------
    Lipman Y, Rustamov RM, Funkhouser TA. Biharmonic distance.
    *ACM TOG* 29(3):27, 2010.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    nz = evals > 1e-10
    weights = np.zeros_like(evals)
    weights[nz] = 1.0 / (evals[nz] ** 2)

    D = _spectral_distance_matrix(evecs, weights, indices=indices)
    logger.debug("Biharmonic distance: shape %s", D.shape)
    return D


# ======================================================================
# §4  COMMUTE-TIME DISTANCE
# ======================================================================

def commute_time_distance(
    decomp: SpectralDecomposition,
    *,
    indices: Optional[np.ndarray] = None,
    warn_large: bool = True,
) -> DistanceMatrix:
    """Commute-time distance from random walk theory.

    .. math::

        d_{CT}^2(x, y) = \\sum_{i=1}^{k}
            \\frac{(\\varphi_i(x) - \\varphi_i(y))^2}{\\lambda_i}

    .. warning::
        Degenerates on large graphs (N > 50 k) — converges to a
        function of vertex degree only (von Luxburg et al. 2010).
        Use biharmonic distance instead for large meshes.

    Parameters
    ----------
    decomp : SpectralDecomposition
    indices : ndarray of int, optional
    warn_large : bool
        Emit warning if N > 50 000.

    Returns
    -------
    ndarray, shape (M, N) or (N, N)
    """
    if warn_large and decomp.n_vertices > 50_000:
        logger.warning(
            "Commute-time distance degenerates on large graphs "
            "(N=%d > 50k). Consider biharmonic_distance() instead.",
            decomp.n_vertices,
        )

    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    nz = evals > 1e-10
    weights = np.zeros_like(evals)
    weights[nz] = 1.0 / evals[nz]

    D = _spectral_distance_matrix(evecs, weights, indices=indices)
    logger.debug("Commute-time distance: shape %s", D.shape)
    return D


# ======================================================================
# §5  DIFFUSION DISTANCE  (Coifman & Lafon, 2006)
# ======================================================================

def diffusion_distance(
    decomp: SpectralDecomposition,
    t: float,
    *,
    indices: Optional[np.ndarray] = None,
) -> DistanceMatrix:
    """Diffusion distance — multi-scale intrinsic metric.

    .. math::

        D_t^2(x, y) = \\sum_{i=1}^{k}
            e^{-2\\lambda_i t}\\,
            (\\varphi_i(x) - \\varphi_i(y))^2

    Small *t* ≈ geodesic; large *t* ≈ global diffusion.

    Parameters
    ----------
    decomp : SpectralDecomposition
    t : float
        Diffusion time scale.
    indices : ndarray of int, optional

    Returns
    -------
    ndarray, shape (M, N) or (N, N)

    References
    ----------
    Coifman RR, Lafon S. Diffusion maps. *Applied and Computational
    Harmonic Analysis* 21(1):5–30, 2006.
    """
    evals = decomp.eigenvalues
    evecs = decomp.eigenvectors

    weights = np.exp(-2.0 * evals * t)

    D = _spectral_distance_matrix(evecs, weights, indices=indices)
    logger.debug("Diffusion distance (t=%.4f): shape %s", t, D.shape)
    return D


def diffusion_distance_multiscale(
    decomp: SpectralDecomposition,
    t_values: np.ndarray,
    *,
    indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Diffusion distance at multiple time scales.

    Parameters
    ----------
    decomp : SpectralDecomposition
    t_values : ndarray, shape (T,)
        Time scales.
    indices : ndarray of int, optional

    Returns
    -------
    ndarray, shape (T, M, N) or (T, N, N)
        Distance matrix per time scale.
    """
    results = []
    with progress_simple("Diffusion distance", total=len(t_values)) as tick:
        for t in t_values:
            D = diffusion_distance(decomp, t, indices=indices)
            results.append(D)
            tick(1)

    return np.stack(results, axis=0)


# ======================================================================
# §6  SPECTRAL DISTANCE BETWEEN DESCRIPTORS (for geometric connectome)
# ======================================================================

def descriptor_distance(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    *,
    method: Literal[
        "wasserstein", "mmd", "euclidean", "cosine", "correlation",
    ] = "wasserstein",
    **kwargs,
) -> float:
    """Distance between two descriptor distributions.

    Used to build the geometric connectome: for each pair of parcels,
    compute the distance between their descriptor distributions.

    Parameters
    ----------
    desc_a : ndarray, shape (N_a,) or (N_a, T)
        Descriptor values at vertices of parcel A.
    desc_b : ndarray, shape (N_b,) or (N_b, T)
        Descriptor values at vertices of parcel B.
    method : str
        ``"wasserstein"`` — 1D Wasserstein (Earth Mover's Distance).
        ``"mmd"`` — Maximum Mean Discrepancy with Gaussian kernel.
        ``"euclidean"`` — L2 between distribution means.
        ``"cosine"`` — cosine distance between means.
        ``"correlation"`` — 1 − Pearson r between aggregated features.

    Returns
    -------
    float

    Notes
    -----
    For 1D descriptors (ScalarMap), Wasserstein is exact and
    O(N log N).  For multi-dimensional descriptors (DescriptorMatrix),
    the columns are treated independently and distances are averaged.
    """
    a = np.asarray(desc_a, dtype=np.float64)
    b = np.asarray(desc_b, dtype=np.float64)

    if method == "wasserstein":
        return _wasserstein_1d_multi(a, b)
    elif method == "mmd":
        return _mmd_gaussian(a, b, **kwargs)
    elif method == "euclidean":
        ma = a.mean(axis=0) if a.ndim > 1 else np.array([a.mean()])
        mb = b.mean(axis=0) if b.ndim > 1 else np.array([b.mean()])
        return float(np.linalg.norm(ma - mb))
    elif method == "cosine":
        ma = a.mean(axis=0) if a.ndim > 1 else np.array([a.mean()])
        mb = b.mean(axis=0) if b.ndim > 1 else np.array([b.mean()])
        dot = np.dot(ma, mb)
        na, nb = np.linalg.norm(ma), np.linalg.norm(mb)
        if na < 1e-20 or nb < 1e-20:
            return 1.0
        return float(1.0 - dot / (na * nb))
    elif method == "correlation":
        ma = a.mean(axis=0) if a.ndim > 1 else np.array([a.mean()])
        mb = b.mean(axis=0) if b.ndim > 1 else np.array([b.mean()])
        if len(ma) < 2:
            return 0.0
        r = np.corrcoef(ma, mb)[0, 1]
        return float(1.0 - r) if np.isfinite(r) else 1.0
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _wasserstein_1d_multi(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein averaged over columns (if multi-dimensional)."""
    from scipy.stats import wasserstein_distance

    if a.ndim == 1 and b.ndim == 1:
        return float(wasserstein_distance(a, b))

    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    T = min(a.shape[1], b.shape[1])
    dists = [
        wasserstein_distance(a[:, t], b[:, t])
        for t in range(T)
    ]
    return float(np.mean(dists))


def _mmd_gaussian(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sigma: Optional[float] = None,
) -> float:
    """Maximum Mean Discrepancy with Gaussian kernel."""
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    if sigma is None:
        combined = np.vstack([a, b])
        # Median heuristic.
        from scipy.spatial.distance import pdist
        dists = pdist(combined[:min(500, len(combined))])
        sigma = float(np.median(dists)) if len(dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

    gamma = 1.0 / (2.0 * sigma ** 2)

    def _k(X: np.ndarray, Y: np.ndarray) -> float:
        D2 = (
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2 * X @ Y.T
        )
        return float(np.mean(np.exp(-gamma * D2)))

    mmd2 = _k(a, a) + _k(b, b) - 2 * _k(a, b)
    return float(np.sqrt(max(mmd2, 0.0)))


# ======================================================================
# §7  CONNECTOME BUILDER
# ======================================================================

def build_geometric_connectome(
    parcel_descriptors: dict,
    *,
    method: Literal[
        "wasserstein", "mmd", "euclidean", "cosine", "correlation",
    ] = "wasserstein",
    **kwargs,
) -> Tuple[DistanceMatrix, list]:
    """Build a ROI × ROI geometric connectome from parcel descriptors.

    For each pair of parcels, computes the distance between their
    descriptor distributions.

    Parameters
    ----------
    parcel_descriptors : dict of {label: ndarray}
        Mapping from parcel label to descriptor array.
        Each value is shape (N_parcel, T) or (N_parcel,).
    method : str
        Distance method (see :func:`descriptor_distance`).
    **kwargs
        Extra args for the distance function.

    Returns
    -------
    matrix : ndarray, shape (R, R)
        Symmetric distance matrix.
    labels : list
        Ordered parcel labels corresponding to matrix rows/columns.

    Examples
    --------
    >>> parcels = sb.io.apply_parcellation(verts, faces, labels)
    >>> descs = {}
    >>> for lab, (v, f) in parcels.items():
    ...     mesh = BrainMesh(v, f)
    ...     decomp = mesh.decompose(k=30)
    ...     descs[lab] = compute_hks(decomp, n_times=20)
    >>> C, labs = build_geometric_connectome(descs, method="wasserstein")
    """
    labels = sorted(parcel_descriptors.keys())
    R = len(labels)
    matrix = np.zeros((R, R), dtype=np.float64)

    total_pairs = R * (R - 1) // 2
    with progress_simple("Geometric connectome", total=total_pairs) as tick:
        for i in range(R):
            for j in range(i + 1, R):
                d = descriptor_distance(
                    parcel_descriptors[labels[i]],
                    parcel_descriptors[labels[j]],
                    method=method,
                    **kwargs,
                )
                matrix[i, j] = d
                matrix[j, i] = d
                tick(1)

    logger.info(
        "Geometric connectome: %d × %d (method=%s)",
        R, R, method,
    )
    return matrix, labels


def aggregate_to_networks(
    connectome: DistanceMatrix,
    parcel_labels: list,
    network_assignments: dict,
    *,
    aggregation: Literal["mean", "median"] = "mean",
) -> Tuple[np.ndarray, list]:
    """Aggregate a parcel-level connectome to network level.

    Parameters
    ----------
    connectome : ndarray, shape (R, R)
        Parcel-level distance matrix.
    parcel_labels : list
        Parcel labels (from :func:`build_geometric_connectome`).
    network_assignments : dict of {parcel_label: network_name}
        Mapping from each parcel to its canonical network.
    aggregation : str
        ``"mean"`` or ``"median"`` within each block.

    Returns
    -------
    network_matrix : ndarray, shape (K, K)
    network_names : list of str
    """
    networks = sorted(set(network_assignments.values()))
    K = len(networks)
    net_idx = {name: i for i, name in enumerate(networks)}

    # Map parcels to network indices.
    parcel_to_net = []
    for lab in parcel_labels:
        net_name = network_assignments.get(lab)
        if net_name is None:
            parcel_to_net.append(-1)
        else:
            parcel_to_net.append(net_idx[net_name])
    parcel_to_net = np.array(parcel_to_net)

    network_matrix = np.zeros((K, K), dtype=np.float64)
    agg_func = np.mean if aggregation == "mean" else np.median

    for i in range(K):
        for j in range(i, K):
            mask_i = parcel_to_net == i
            mask_j = parcel_to_net == j
            block = connectome[np.ix_(mask_i, mask_j)]

            if i == j:
                # Intra-network: exclude diagonal.
                vals = block[np.triu_indices_from(block, k=1)]
            else:
                vals = block.ravel()

            if len(vals) > 0:
                network_matrix[i, j] = float(agg_func(vals))
                network_matrix[j, i] = network_matrix[i, j]

    logger.info("Network matrix: %d × %d", K, K)
    return network_matrix, networks


# ======================================================================

__all__: List[str] = [
    # Shape-to-shape
    "wesd",
    "wesd_matrix",
    "shapedna_distance",
    # Point-to-point
    "biharmonic_distance",
    "commute_time_distance",
    "diffusion_distance",
    "diffusion_distance_multiscale",
    # Descriptor distributions
    "descriptor_distance",
    # Geometric connectome
    "build_geometric_connectome",
    "aggregate_to_networks",
]
