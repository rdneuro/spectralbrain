"""Distance-dependent Chinese Restaurant Process (ddCRP) with NIW likelihood.

This is brainmosaic's primary contiguous, unsupervised, K-emergent clusterer. Each
vertex links to a neighbouring vertex (or itself) with probability depending on
a decay function of distance; clusters are the connected components of the link
graph. Restricting candidate links to mesh adjacency makes clusters spatially
**contiguous by construction**.

Cluster fit is scored by the **collapsed Normal-Inverse-Wishart marginal
likelihood** (parameters integrated out), NOT a size-scaling sum of
similarities. The latter grows monotonically with cluster size and causes the
degenerate "everything collapses into one cluster" bug; the NIW marginal
likelihood includes an automatic Bayesian Occam penalty (via the determinant
and multivariate-gamma ratios) that keeps cluster sizes finite.

References
----------
- Blei & Frazier (2011), Distance dependent Chinese restaurant processes,
  JMLR 12:2461-2488.
- Baldassano, Beck & Fei-Fei (2015), Parcellating connectivity in spatial maps,
  PeerJ 3:e784.
- Murphy (2007/2012), Conjugate Bayesian analysis of the Gaussian distribution.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.special import gammaln

logger = logging.getLogger("spectralbrain.statistics._clustercore")
_DDCRP_DEBUG = bool(os.environ.get("SB_DDCRP_DEBUG"))


# --------------------------------------------------------------------------- #
# NIW marginal likelihood
# --------------------------------------------------------------------------- #
@dataclass
class NIWPrior:
    """Normal-Inverse-Wishart prior hyperparameters.

    Attributes
    ----------
    mu0 : np.ndarray, shape (d,)
        Prior mean.
    kappa0 : float
        Prior mean confidence (pseudo-count).
    nu0 : float
        Prior degrees of freedom; must satisfy ``nu0 > d - 1``.
    psi0 : np.ndarray, shape (d, d)
        Prior scale matrix (symmetric positive definite).
    """

    mu0: np.ndarray
    kappa0: float
    nu0: float
    psi0: np.ndarray

    @classmethod
    def from_data(cls, X: np.ndarray, kappa0: float = 0.1,
                  nu0_offset: float = 2.0, psi_scale: float = 1.0) -> "NIWPrior":
        """Weakly-informative prior centred on the data's global statistics."""
        X = np.asarray(X, float)
        d = X.shape[1]
        mu0 = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov)
        # Regularise to ensure positive definiteness.
        cov = cov + np.eye(d) * (1e-6 + 1e-3 * np.trace(cov) / max(d, 1))
        nu0 = d + nu0_offset
        psi0 = cov * psi_scale * nu0
        return cls(mu0=mu0, kappa0=float(kappa0), nu0=float(nu0), psi0=psi0)


def _multivariate_gammaln(a: float, d: int) -> float:
    """Log multivariate gamma function ``log Gamma_d(a)``."""
    j = np.arange(1, d + 1)
    return (d * (d - 1) / 4.0) * np.log(np.pi) + gammaln(a + (1.0 - j) / 2.0).sum()


def _logdet_spd(matrix: np.ndarray) -> float:
    """Stable log-determinant of an SPD matrix via Cholesky."""
    try:
        L = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # Jitter and retry.
        eps = 1e-8 * np.trace(matrix) / matrix.shape[0]
        L = np.linalg.cholesky(matrix + np.eye(matrix.shape[0]) * eps)
    return 2.0 * np.log(np.diag(L)).sum()


def niw_log_marginal(n: int, sum_x: np.ndarray, sum_xx: np.ndarray,
                     prior: NIWPrior) -> float:
    """Collapsed NIW log marginal likelihood of a cluster from sufficient stats.

    Parameters
    ----------
    n : int
        Number of points in the cluster.
    sum_x : np.ndarray, shape (d,)
        Sum of the cluster's feature vectors.
    sum_xx : np.ndarray, shape (d, d)
        Sum of outer products ``sum_i x_i x_i^T``.
    prior : NIWPrior

    Returns
    -------
    float
        ``log p(X_cluster)`` with the Gaussian parameters integrated out.
    """
    d = prior.mu0.shape[0]
    if n == 0:
        return 0.0
    xbar = sum_x / n
    scatter = sum_xx - n * np.outer(xbar, xbar)           # S = sum (x-xbar)(x-xbar)^T

    kappa_n = prior.kappa0 + n
    nu_n = prior.nu0 + n
    diff = xbar - prior.mu0
    psi_n = (prior.psi0 + scatter
             + (prior.kappa0 * n / kappa_n) * np.outer(diff, diff))
    psi_n = 0.5 * (psi_n + psi_n.T)                        # symmetrise

    log_ml = (
        -(n * d / 2.0) * np.log(np.pi)
        + _multivariate_gammaln(nu_n / 2.0, d)
        - _multivariate_gammaln(prior.nu0 / 2.0, d)
        + (prior.nu0 / 2.0) * _logdet_spd(prior.psi0)
        - (nu_n / 2.0) * _logdet_spd(psi_n)
        + (d / 2.0) * (np.log(prior.kappa0) - np.log(kappa_n))
    )
    return float(log_ml)


def _make_marginal(prior: NIWPrior) -> Callable[[int, np.ndarray, np.ndarray], float]:
    """Return a fast ``ml(n, sum_x, sum_xx)`` with the prior's constant terms
    precomputed once and the per-``n`` ``log Gamma_d(nu_n/2)`` memoised.

    Numerically identical to :func:`niw_log_marginal`; it just hoists the
    ``log|psi0|`` Cholesky and ``log Gamma_d(nu0/2)`` (both constants, previously
    recomputed on every call) out of the hot loop, and caches the ``nu_n``-gammaln
    by cluster size ``n``.
    """
    d = prior.mu0.shape[0]
    nu0 = float(prior.nu0)
    kappa0 = float(prior.kappa0)
    mu0 = np.asarray(prior.mu0, float)
    psi0 = np.asarray(prior.psi0, float)
    half = np.arange(1, d + 1)
    log_pi = np.log(np.pi)
    log_k0 = np.log(kappa0)

    def _mvgln(a: float) -> float:
        return (d * (d - 1) / 4.0) * log_pi + gammaln(a + (1.0 - half) / 2.0).sum()

    const = (nu0 / 2.0) * _logdet_spd(psi0) - _mvgln(nu0 / 2.0)
    gln_cache: Dict[int, float] = {}

    def ml(n: int, sum_x: np.ndarray, sum_xx: np.ndarray) -> float:
        if n == 0:
            return 0.0
        xbar = sum_x / n
        scatter = sum_xx - n * np.outer(xbar, xbar)
        kappa_n = kappa0 + n
        nu_n = nu0 + n
        diff = xbar - mu0
        psi_n = psi0 + scatter + (kappa0 * n / kappa_n) * np.outer(diff, diff)
        psi_n = 0.5 * (psi_n + psi_n.T)
        g = gln_cache.get(n)
        if g is None:
            g = _mvgln(nu_n / 2.0)
            gln_cache[n] = g
        return float(-(n * d / 2.0) * log_pi + g + const
                     - (nu_n / 2.0) * _logdet_spd(psi_n)
                     + (d / 2.0) * (log_k0 - np.log(kappa_n)))

    return ml


# --------------------------------------------------------------------------- #
# Decay functions
# --------------------------------------------------------------------------- #
def make_decay(kind: str = "window", scale: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Return a log-decay function ``log f(d)`` for ddCRP link priors.

    Parameters
    ----------
    kind : {"window", "exponential", "logistic"}
        ``"window"``  : ``f(d)=1`` if ``d<=scale`` else 0 (hard contiguity).
        ``"exponential"`` : ``f(d)=exp(-d/scale)``.
        ``"logistic"`` : ``f(d)=1/(1+exp((d-scale)/(0.1*scale)))``.
    scale : float
        Decay length. Small values keep parcels compact/contiguous (Ghosh et al.
        2011: small decay -> small contiguous segments).

    Returns
    -------
    callable
        Maps a distance array to a log-prior array (``-inf`` where f(d)=0).
    """
    scale = float(scale)
    if kind == "window":
        def logf(d):
            out = np.where(np.asarray(d) <= scale, 0.0, -np.inf)
            return out
    elif kind == "exponential":
        def logf(d):
            return -np.asarray(d, float) / max(scale, 1e-12)
    elif kind == "logistic":
        def logf(d):
            d = np.asarray(d, float)
            return -np.log1p(np.exp((d - scale) / (0.1 * scale + 1e-12)))
    else:
        raise ValueError(f"Unknown decay kind {kind!r}.")
    return logf


# --------------------------------------------------------------------------- #
# ddCRP sampler
# --------------------------------------------------------------------------- #
@dataclass
class DDCRPResult:
    """Posterior summary of a ddCRP run.

    Attributes
    ----------
    labels : np.ndarray, shape (V,)
        MAP / last-draw partition (relabelled 0..K-1).
    co_association : np.ndarray, shape (V, V)
        Posterior probability that two vertices share a cluster (consensus).
    n_clusters_trace : np.ndarray
        Number of clusters per retained draw (all chains concatenated).
    rhat_n_clusters : float
        Gelman-Rubin R-hat on the cluster count across chains (NaN if 1 chain).
    chains : int
    """

    labels: np.ndarray
    co_association: np.ndarray
    n_clusters_trace: np.ndarray
    rhat_n_clusters: float
    chains: int


def _components_from_links(links: np.ndarray) -> np.ndarray:
    """Connected components (tables) of the undirected link graph."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n = links.shape[0]
    rows = np.arange(n)
    A = sp.coo_matrix((np.ones(n), (rows, links)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.int8)
    _, labels = connected_components(A, directed=False)
    return labels


class DDCRP:
    """Collapsed-Gibbs ddCRP sampler with adjacency-restricted contiguity.

    Parameters
    ----------
    decay_kind : {"window", "exponential", "logistic"}
    decay_scale : float
        Decay length (see :func:`make_decay`).
    alpha : float
        Self-link concentration; larger ``alpha`` -> more, smaller clusters.
    prior : NIWPrior, optional
        NIW prior; if None, built weakly-informative from data in :meth:`fit`.
    n_draws : int
        Retained Gibbs sweeps per chain (after burn-in).
    burn_in : int
        Discarded warmup sweeps per chain.
    thin : int
        Keep every ``thin``-th sweep.
    chains : int
        Independent chains (for R-hat and a smoother co-association).
    random_state : int
    """

    def __init__(self, decay_kind: str = "window", decay_scale: float = 1.0,
                 alpha: float = 1.0, prior: Optional[NIWPrior] = None,
                 n_draws: int = 200, burn_in: int = 100, thin: int = 2,
                 chains: int = 4, random_state: int = 0) -> None:
        self.logf = make_decay(decay_kind, decay_scale)
        self.alpha = float(alpha)
        self.log_alpha = np.log(max(self.alpha, 1e-300))
        self.prior = prior
        self.n_draws = int(n_draws)
        self.burn_in = int(burn_in)
        self.thin = max(int(thin), 1)
        self.chains = int(chains)
        self.random_state = int(random_state)

    @staticmethod
    def _subtree(i, in_links):
        """Nodes that reach ``i`` by following links (i.e. i + its descendants in
        the parent-pointer forest). After i's outgoing link is removed and i is
        self-linked, this set is exactly i's connected component *iff* removal
        split the cluster; if a cycle ran through i, ``old`` is found inside it.
        """
        seen = {i}
        stack = [i]
        while stack:
            u = stack.pop()
            for w in in_links[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    def fit(self, X: np.ndarray, adjacency_list: Sequence[np.ndarray],
            distances: Optional[Sequence[np.ndarray]] = None,
            vertices: Optional[np.ndarray] = None,
            progress: bool = True) -> DDCRPResult:
        """Run the sampler.

        Parameters
        ----------
        X : np.ndarray, shape (V, d)
            Per-vertex feature vectors (e.g. fused/fPCA descriptor scores).
        adjacency_list : sequence of int arrays
            ``adjacency_list[i]`` lists vertex ``i``'s mesh neighbours. Candidate
            links are restricted to these (plus self), enforcing contiguity.
        distances : sequence of float arrays, optional
            ``distances[i]`` are distances to ``adjacency_list[i]``. If None and
            ``vertices`` is given, Euclidean edge distances are computed so the
            decay function modulates contiguity by physical distance; if both are
            None, all neighbours are treated as equidistant (decay acts as a
            hard window gate only).
        vertices : np.ndarray, optional, shape (V, 3)
            Mesh vertex coordinates; used to derive ``distances`` when those are
            not supplied.
        progress : bool
            Show a rich progress bar.

        Returns
        -------
        DDCRPResult

        Notes
        -----
        Per-component sufficient statistics ``(n, sum_x, sum_xx)`` are maintained
        incrementally: removing i's link detaches at most i's ``_subtree`` (whose
        stats are subtracted from the parent cluster), and a merge adds two cached
        stat tuples in O(d^2). The cluster scatter is therefore never recomputed
        from scratch inside the candidate loop, and the collapsed NIW marginal is
        cached per cluster. This is what makes the sampler tractable on dense
        surfaces (thousands of vertices).
        """
        from ._progress import progress_bar

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be (V, d); got {X.shape}.")
        n, d = X.shape
        if len(adjacency_list) != n:
            raise ValueError("adjacency_list length must equal #vertices.")
        prior = self.prior or NIWPrior.from_data(X)
        if distances is None and vertices is not None:
            from ._meshgeom import edge_distances
            distances = edge_distances(vertices, adjacency_list)
        if distances is None:
            distances = [np.ones(len(neigh)) for neigh in adjacency_list]

        # Precompute per-neighbourhood log-priors (self log-prior = log_alpha).
        log_prior_neigh = [self.logf(np.asarray(dd, float)) for dd in distances]
        ml = _make_marginal(prior)             # fast marginal with cached constants
        # Per-node outer products x x^T (reused when summing small subtrees).
        XX = np.einsum("ij,ik->ijk", X, X)     # (V, d, d)

        rng_master = np.random.default_rng(self.random_state)
        all_labels_draws: List[np.ndarray] = []
        n_clusters_per_chain: List[List[int]] = []

        total_sweeps = self.chains * (self.burn_in + self.n_draws)
        with progress_bar("ddCRP Gibbs", total=total_sweeps, disable=not progress) as advance:
            for chain in range(self.chains):
                rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
                links = np.arange(n)                       # all self-linked
                in_links: List[set] = [set() for _ in range(n)]

                # Component bookkeeping: every node starts as its own singleton.
                cid_of = np.arange(n)
                members_of: Dict[int, set] = {c: {c} for c in range(n)}
                stats_of: Dict[int, tuple] = {
                    c: (1, X[c], XX[c]) for c in range(n)}
                ml_of: Dict[int, float] = {}
                next_cid = n
                chain_counts: List[int] = []

                def get_ml(c):
                    v = ml_of.get(c)
                    if v is None:
                        s = stats_of[c]
                        v = ml(s[0], s[1], s[2])
                        ml_of[c] = v
                    return v

                for sweep in range(self.burn_in + self.n_draws):
                    order = rng.permutation(n)
                    for i in order:
                        old = links[i]
                        sub = self._subtree(i, in_links)       # i + descendants
                        if old != i:
                            in_links[old].discard(i)
                        links[i] = i
                        ci_old = cid_of[i]
                        split = (old != i) and (old not in sub)

                        if split:
                            idxS = np.fromiter(sub, np.int64, len(sub))
                            subX = X[idxS]
                            nS = idxS.size
                            sxS = subX.sum(axis=0)
                            sxxS = subX.T @ subX
                            cS = next_cid
                            next_cid += 1
                            for u in sub:
                                cid_of[u] = cS
                            members_of[cS] = set(sub)
                            stats_of[cS] = (nS, sxS, sxxS)
                            so = stats_of[ci_old]
                            members_of[ci_old] -= sub
                            stats_of[ci_old] = (so[0] - nS, so[1] - sxS, so[2] - sxxS)
                            ml_of.pop(ci_old, None)
                            comp_i = cS
                            ni, sxi, sxxi = nS, sxS, sxxS
                        else:
                            comp_i = ci_old
                            ni, sxi, sxxi = stats_of[ci_old]

                        ml_i = get_ml(comp_i)
                        neigh = adjacency_list[i]
                        cand = list(neigh) + [i]
                        cand_lp = list(log_prior_neigh[i]) + [self.log_alpha]

                        scores = np.full(len(cand), -np.inf)
                        for k, (j, lp) in enumerate(zip(cand, cand_lp)):
                            if not np.isfinite(lp):
                                continue
                            if j == i or cid_of[j] == comp_i:
                                scores[k] = lp                 # no merge
                            else:
                                cj = cid_of[j]
                                sj = stats_of[cj]
                                ml_merge = ml(ni + sj[0], sxi + sj[1], sxxi + sj[2])
                                scores[k] = lp + ml_merge - ml_i - get_ml(cj)

                        finite = np.isfinite(scores)
                        s = scores[finite]
                        s = s - s.max()
                        probs = np.exp(s)
                        probs /= probs.sum()
                        chosen_local = rng.choice(np.where(finite)[0], p=probs)
                        j_new = cand[chosen_local]

                        links[i] = j_new
                        if j_new != i:
                            in_links[j_new].add(i)

                        if j_new == i or cid_of[j_new] == comp_i:
                            pass                                # i's cluster unchanged
                        else:
                            cj = cid_of[j_new]
                            mi, mj = members_of[comp_i], members_of[cj]
                            si, sj = stats_of[comp_i], stats_of[cj]
                            merged = (si[0] + sj[0], si[1] + sj[1], si[2] + sj[2])
                            # union by size: relabel the smaller member set.
                            if len(mi) <= len(mj):
                                for u in mi:
                                    cid_of[u] = cj
                                mj |= mi
                                stats_of[cj] = merged
                                ml_of.pop(cj, None)
                                members_of.pop(comp_i, None)
                                stats_of.pop(comp_i, None)
                                ml_of.pop(comp_i, None)
                            else:
                                for u in mj:
                                    cid_of[u] = comp_i
                                mi |= mj
                                stats_of[comp_i] = merged
                                ml_of.pop(comp_i, None)
                                members_of.pop(cj, None)
                                stats_of.pop(cj, None)
                                ml_of.pop(cj, None)

                    labels = _relabel_consecutive(cid_of.copy())
                    chain_counts.append(int(len(np.unique(labels))))
                    if _DDCRP_DEBUG:
                        from collections import defaultdict
                        agg = defaultdict(lambda: [0, np.zeros(d), np.zeros((d, d))])
                        for node in range(n):
                            c = int(cid_of[node])
                            agg[c][0] += 1
                            agg[c][1] += X[node]
                            agg[c][2] += XX[node]
                        for c, (nn, sx, sxx) in agg.items():
                            st = stats_of[c]
                            assert st[0] == nn, (c, st[0], nn)
                            assert np.allclose(st[1], sx, atol=1e-7)
                            assert np.allclose(st[2], sxx, atol=1e-5)
                    if sweep >= self.burn_in and ((sweep - self.burn_in) % self.thin == 0):
                        all_labels_draws.append(labels.copy())
                    advance()
                n_clusters_per_chain.append(chain_counts[self.burn_in:])

        # Co-association across all retained draws.
        co = np.zeros((n, n), dtype=np.float64)
        for labels in all_labels_draws:
            co += (labels[:, None] == labels[None, :]).astype(np.float64)
        co /= max(len(all_labels_draws), 1)

        rhat = _rhat([np.asarray(c, float) for c in n_clusters_per_chain])
        last_labels = all_labels_draws[-1] if all_labels_draws else _components_from_links(links)
        last_labels = _relabel_consecutive(last_labels)
        n_trace = np.concatenate([np.asarray(c) for c in n_clusters_per_chain]) \
            if n_clusters_per_chain else np.array([])
        return DDCRPResult(labels=last_labels, co_association=co,
                           n_clusters_trace=n_trace, rhat_n_clusters=rhat,
                           chains=self.chains)


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    return inv


def _rhat(chains: List[np.ndarray]) -> float:
    """Gelman-Rubin R-hat for a scalar across chains of equal length."""
    chains = [c for c in chains if c.size > 1]
    if len(chains) < 2:
        return float("nan")
    m = len(chains)
    n = min(len(c) for c in chains)
    arr = np.stack([c[:n] for c in chains])          # (m, n)
    chain_means = arr.mean(axis=1)
    chain_vars = arr.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = (1 - 1 / n) * W + B / n
    if W <= 0:
        return float("nan")
    return float(np.sqrt(var_hat / W))


def cluster_ddcrp(X, adjacency_list, decay_kind="window", decay_scale=1.0,
                  alpha=1.0, prior=None, n_draws=200, burn_in=100, thin=2,
                  chains=4, random_state=0, distances=None, vertices=None,
                  progress=True) -> DDCRPResult:
    """Functional wrapper around :class:`DDCRP` (convenience entry point).

    Pass ``vertices`` (mesh coordinates) to let the decay function use real
    Euclidean edge distances; otherwise neighbours are treated as equidistant.
    """
    sampler = DDCRP(decay_kind=decay_kind, decay_scale=decay_scale, alpha=alpha,
                    prior=prior, n_draws=n_draws, burn_in=burn_in, thin=thin,
                    chains=chains, random_state=random_state)
    return sampler.fit(X, adjacency_list, distances=distances, vertices=vertices,
                       progress=progress)
