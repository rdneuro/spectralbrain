"""Functional ddCRP: cluster whole HKS/WKS *curves* via fPCA + NIW ddCRP.

CAESAR (Janssen, Jylanki & van Gerven 2016) couples a ddCRP with Gaussian-process
temporal priors, but the GP scales cubically in the number of time points. Here
we obtain the same "use the whole temporal/energy profile" benefit by compressing
each vertex's HKS profile over diffusion time (and/or WKS over energy) with
functional PCA, then running the spatial NIW ddCRP on the low-dimensional fPCA
scores. This keeps per-cluster scoring at NIW cost with no GP inversion.

References
----------
- Janssen, Jylanki & van Gerven (2016), CAESAR, PLOS ONE 11(12):e0164703.
- Ramsay & Silverman (2005), Functional Data Analysis, 2nd ed., Springer.
- Blei & Frazier (2011), Distance dependent CRPs, JMLR 12.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .ddcrp import DDCRP, DDCRPResult, NIWPrior

logger = logging.getLogger("spectralbrain.statistics._clustercore")


def fpca_compress(curves: np.ndarray, n_components: int = 5,
                  center: bool = True) -> Tuple[np.ndarray, dict]:
    """Compress a stack of curves to functional-PCA scores.

    Uses scikit-fda's FPCA when available (proper basis expansion + smoothing),
    otherwise falls back to plain PCA on the sampled curves (equivalent when the
    sampling grid is dense and uniform).

    Parameters
    ----------
    curves : np.ndarray, shape (V, T)
        One sampled curve per vertex (e.g. HKS over T diffusion times).
    n_components : int
        Number of functional principal components (scores) to keep.
    center : bool
        Subtract the mean curve before decomposition.

    Returns
    -------
    scores : np.ndarray, shape (V, n_components)
    info : dict
        Diagnostics: ``explained_variance_ratio`` and ``backend``.
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError(f"curves must be (V, T); got {curves.shape}.")
    n_components = int(min(n_components, min(curves.shape)))

    try:
        from skfda import FDataGrid
        from skfda.preprocessing.dim_reduction import FPCA
        grid = np.arange(curves.shape[1], dtype=float)
        fd = FDataGrid(data_matrix=curves, grid_points=grid)
        fpca = FPCA(n_components=n_components)
        scores = fpca.fit_transform(fd)
        evr = getattr(fpca, "explained_variance_ratio_", None)
        return np.asarray(scores), {"explained_variance_ratio": evr, "backend": "skfda"}
    except Exception:
        from sklearn.decomposition import PCA
        X = curves - curves.mean(axis=0, keepdims=True) if center else curves
        pca = PCA(n_components=n_components, random_state=0)
        scores = pca.fit_transform(X)
        return scores, {"explained_variance_ratio": pca.explained_variance_ratio_,
                        "backend": "pca-fallback"}


def cluster_ddcrp_functional(
    curves_blocks: Sequence[np.ndarray],
    adjacency_list: Sequence[np.ndarray],
    n_fpca: int = 5,
    decay_kind: str = "window",
    decay_scale: float = 1.0,
    alpha: float = 1.0,
    prior: Optional[NIWPrior] = None,
    n_draws: int = 200,
    burn_in: int = 100,
    thin: int = 2,
    chains: int = 4,
    random_state: int = 0,
    distances: Optional[Sequence[np.ndarray]] = None,
    vertices: Optional[np.ndarray] = None,
    progress: bool = True,
) -> Tuple[DDCRPResult, dict]:
    """Functional ddCRP over one or more curve blocks (e.g. HKS and WKS).

    Each block is fPCA-compressed independently and the scores concatenated
    (MFA-style first-singular-value balancing is applied across blocks so a wider
    block does not dominate), then the spatial NIW ddCRP clusters the result.

    Parameters
    ----------
    curves_blocks : sequence of (V, T_b) arrays
        Curve stacks to fuse (e.g. ``[hks_VxT, wks_VxE]``).
    adjacency_list : sequence of int arrays
        Mesh neighbours per vertex (candidate links -> contiguity).
    n_fpca : int
        fPCA components kept per block.
    decay_kind, decay_scale, alpha, prior, n_draws, burn_in, thin, chains,
    random_state, distances, progress
        Passed to :class:`~brainmosaic.clustering.spatial.ddcrp.DDCRP`.

    Returns
    -------
    result : DDCRPResult
    info : dict
        Per-block fPCA diagnostics.
    """
    if len(curves_blocks) == 0:
        raise ValueError("Provide at least one curve block.")

    score_blocks = []
    info = {}
    for b, curves in enumerate(curves_blocks):
        scores, diag = fpca_compress(curves, n_components=n_fpca)
        # MFA-style balancing: divide block by its first singular value.
        sv = np.linalg.svd(scores - scores.mean(0, keepdims=True),
                           compute_uv=False)
        scale = float(max(sv[0], 1e-12)) if sv.size else 1.0
        score_blocks.append(scores / scale)
        info[f"block_{b}"] = {**diag, "sv1": scale}

    X = np.hstack(score_blocks)                       # (V, sum n_fpca)
    sampler = DDCRP(decay_kind=decay_kind, decay_scale=decay_scale, alpha=alpha,
                    prior=prior, n_draws=n_draws, burn_in=burn_in, thin=thin,
                    chains=chains, random_state=random_state)
    result = sampler.fit(X, adjacency_list, distances=distances,
                         vertices=vertices, progress=progress)
    info["fused_dim"] = X.shape[1]
    return result, info
