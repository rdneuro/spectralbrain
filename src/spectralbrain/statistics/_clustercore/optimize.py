"""Data-driven hyperparameter optimisation for the ddCRP.

Selects ddCRP hyperparameters from the data instead of hand-tuning, so the user
never has to guess ``psi_scale`` / ``kappa0`` / ``alpha`` / ``decay_scale`` to
avoid the K=1 collapse. Three optimiser backends are supported and lazily
imported, chosen by the ``backend`` argument:

- ``"optuna"``   : Tree-structured Parzen Estimator (handles mixed spaces; default).
- ``"hyperopt"`` : TPE via hyperopt.
- ``"botorch"``  : Gaussian-process Bayesian optimisation (BoTorch/torch) over the
  continuous parameters, enumerating the categorical decay kind.

If the requested backend is not installed, a reproducible **random search**
fallback runs so the routine works with only the core stack.

The objective (see :mod:`brainmosaic.tuning.objective`) maximises a partition-quality
score with an anti-degeneracy guard. Each trial runs the ddCRP at a *reduced*
sampling budget for speed; the best configuration can be refit at full budget.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .ddcrp import NIWPrior, cluster_ddcrp
from .objective import DEGENERATE_SCORE, VALID_OBJECTIVES, score_partition

logger = logging.getLogger("spectralbrain.statistics._clustercore")

VALID_TUNERS = ("optuna", "hyperopt", "botorch", "random")
DECAY_KINDS = ("window", "exponential", "logistic")


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
@dataclass
class SearchSpace:
    """Bounds for the ddCRP hyperparameter search.

    Continuous bounds marked ``log`` are sampled log-uniformly.
    """

    decay_kinds: Sequence[str] = DECAY_KINDS
    decay_scale: tuple = (0.25, 8.0)        # log
    alpha: tuple = (1e-2, 1e2)              # log
    kappa0: tuple = (1e-3, 10.0)            # log
    nu0_offset: tuple = (1.0, 10.0)         # linear
    psi_scale: tuple = (1e-2, 1e2)          # log
    n_pca: Optional[tuple] = None           # e.g. (2, 20) to also search a PCA reduction


@dataclass
class TuningResult:
    """Outcome of a ddCRP hyperparameter optimisation.

    Attributes
    ----------
    best_params : dict
        Best hyperparameters found (keys: decay_kind, decay_scale, alpha,
        kappa0, nu0_offset, psi_scale, optionally n_pca).
    best_score : float
    history : list of dict
        Per-trial ``{"params": ..., "score": ...}`` records.
    backend : str
        Optimiser backend actually used.
    study : object
        Backend-native study/trials object when available (else None).
    refit_result : DDCRPResult or None
        Full-budget ddCRP fit with ``best_params`` (if ``refit=True``).
    """

    best_params: Dict[str, object]
    best_score: float
    history: List[Dict[str, object]]
    backend: str
    study: object = None
    refit_result: object = None


def _tuner_available(name: str) -> bool:
    mods = {"optuna": "optuna", "hyperopt": "hyperopt", "botorch": "botorch"}
    if name == "random":
        return True
    try:
        return importlib.util.find_spec(mods[name]) is not None
    except (ImportError, ValueError, KeyError):
        return False


def _resolve_tuner(backend: Optional[str]) -> str:
    if backend is None:
        for cand in ("optuna", "hyperopt", "botorch"):
            if _tuner_available(cand):
                return cand
        return "random"
    backend = backend.lower()
    if backend not in VALID_TUNERS:
        raise ValueError(f"Unknown tuner {backend!r}; choose {VALID_TUNERS}.")
    if not _tuner_available(backend):
        logger.warning("Tuner %r not installed; falling back to random search.", backend)
        return "random"
    return backend


def _maybe_pca(X: np.ndarray, n_pca: Optional[int]) -> np.ndarray:
    if not n_pca:
        return X
    from sklearn.decomposition import PCA
    k = int(min(n_pca, min(X.shape)))
    return PCA(n_components=k, random_state=0).fit_transform(
        X - X.mean(0, keepdims=True))


# --------------------------------------------------------------------------- #
# Core evaluator (shared by all backends)
# --------------------------------------------------------------------------- #
class _Evaluator:
    """Builds + scores a ddCRP for a given hyperparameter dict (reduced budget)."""

    def __init__(self, X, adjacency_list, *, objective, spatial_distance,
                 distances, eval_draws, eval_burn_in, eval_thin, eval_chains,
                 random_state):
        self.X = np.asarray(X, float)
        self.adjacency_list = adjacency_list
        self.objective = objective
        self.spatial_distance = spatial_distance
        self.distances = distances
        self.eval_draws = eval_draws
        self.eval_burn_in = eval_burn_in
        self.eval_thin = eval_thin
        self.eval_chains = eval_chains
        self.random_state = random_state
        self.history: List[Dict[str, object]] = []

    def __call__(self, params: Dict[str, object]) -> float:
        Xr = _maybe_pca(self.X, params.get("n_pca"))
        try:
            prior = NIWPrior.from_data(
                Xr, kappa0=float(params["kappa0"]),
                nu0_offset=float(params["nu0_offset"]),
                psi_scale=float(params["psi_scale"]))
            res = cluster_ddcrp(
                Xr, self.adjacency_list, decay_kind=params["decay_kind"],
                decay_scale=float(params["decay_scale"]), alpha=float(params["alpha"]),
                prior=prior, n_draws=self.eval_draws, burn_in=self.eval_burn_in,
                thin=self.eval_thin, chains=self.eval_chains,
                random_state=self.random_state, distances=self.distances,
                progress=False)
            score = score_partition(
                Xr, res.labels, objective=self.objective,
                spatial_distance=self.spatial_distance,
                co_association=res.co_association)
        except Exception as exc:  # a bad config should not abort the whole search
            logger.debug("Trial failed for %s: %s", params, exc)
            score = DEGENERATE_SCORE
        self.history.append({"params": dict(params), "score": float(score)})
        return float(score)


# --------------------------------------------------------------------------- #
# Backend drivers
# --------------------------------------------------------------------------- #
def _run_optuna(evaluator, space, n_trials, random_state, progress):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "decay_kind": trial.suggest_categorical("decay_kind", list(space.decay_kinds)),
            "decay_scale": trial.suggest_float("decay_scale", *space.decay_scale, log=True),
            "alpha": trial.suggest_float("alpha", *space.alpha, log=True),
            "kappa0": trial.suggest_float("kappa0", *space.kappa0, log=True),
            "nu0_offset": trial.suggest_float("nu0_offset", *space.nu0_offset),
            "psi_scale": trial.suggest_float("psi_scale", *space.psi_scale, log=True),
        }
        if space.n_pca is not None:
            params["n_pca"] = trial.suggest_int("n_pca", *space.n_pca)
        return evaluator(params)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=progress)
    return study.best_params, study.best_value, study


def _run_hyperopt(evaluator, space, n_trials, random_state, progress):
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    kinds = list(space.decay_kinds)
    hp_space = {
        "decay_kind": hp.choice("decay_kind", kinds),
        "decay_scale": hp.loguniform("decay_scale", np.log(space.decay_scale[0]),
                                     np.log(space.decay_scale[1])),
        "alpha": hp.loguniform("alpha", np.log(space.alpha[0]), np.log(space.alpha[1])),
        "kappa0": hp.loguniform("kappa0", np.log(space.kappa0[0]), np.log(space.kappa0[1])),
        "nu0_offset": hp.uniform("nu0_offset", *space.nu0_offset),
        "psi_scale": hp.loguniform("psi_scale", np.log(space.psi_scale[0]),
                                   np.log(space.psi_scale[1])),
    }
    if space.n_pca is not None:
        hp_space["n_pca"] = hp.quniform("n_pca", space.n_pca[0], space.n_pca[1], 1)

    def objective(p):
        params = dict(p)
        if "n_pca" in params:
            params["n_pca"] = int(params["n_pca"])
        return {"loss": -evaluator(params), "status": STATUS_OK}

    trials = Trials()
    fmin(objective, hp_space, algo=tpe.suggest, max_evals=n_trials, trials=trials,
         rstate=np.random.default_rng(random_state), show_progressbar=progress)
    # Recover best params from history (robust against hp.choice index decoding).
    best = max(evaluator.history, key=lambda h: h["score"])
    return best["params"], best["score"], trials


def _run_botorch(evaluator, space, n_trials, random_state, progress):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(random_state)
    dtype = torch.double
    # Continuous dims (some log-scaled): decay_scale, alpha, kappa0, nu0_offset, psi_scale
    log_mask = np.array([True, True, True, False, True])
    lows = np.array([space.decay_scale[0], space.alpha[0], space.kappa0[0],
                     space.nu0_offset[0], space.psi_scale[0]], float)
    highs = np.array([space.decay_scale[1], space.alpha[1], space.kappa0[1],
                      space.nu0_offset[1], space.psi_scale[1]], float)
    tl = np.where(log_mask, np.log(lows), lows)
    th = np.where(log_mask, np.log(highs), highs)

    def decode(unit_row, kind):
        z = tl + np.asarray(unit_row) * (th - tl)
        vals = np.where(log_mask, np.exp(z), z)
        return {"decay_kind": kind, "decay_scale": float(vals[0]),
                "alpha": float(vals[1]), "kappa0": float(vals[2]),
                "nu0_offset": float(vals[3]), "psi_scale": float(vals[4])}

    from ._progress import progress_bar
    d = 5
    n_init = max(4, n_trials // (2 * len(space.decay_kinds)))
    per_kind_iters = max(1, n_trials // len(space.decay_kinds) - n_init)
    rng = np.random.default_rng(random_state)
    global_best, global_best_params = -np.inf, None

    total = len(space.decay_kinds) * (n_init + per_kind_iters)
    with progress_bar("BoTorch BO", total=total, disable=not progress) as advance:
        for kind in space.decay_kinds:
            train_x = torch.tensor(rng.random((n_init, d)), dtype=dtype)
            train_y = torch.tensor(
                [[evaluator(decode(x.numpy(), kind))] for x in train_x], dtype=dtype)
            for _ in range(n_init):
                advance()
            for _ in range(per_kind_iters):
                y = train_y.clone()
                y = (y - y.mean()) / (y.std() + 1e-9)
                gp = SingleTaskGP(train_x, y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                try:
                    fit_gpytorch_mll(mll)
                    acqf = qLogExpectedImprovement(gp, best_f=y.max())
                    bounds = torch.stack([torch.zeros(d, dtype=dtype),
                                          torch.ones(d, dtype=dtype)])
                    cand, _ = optimize_acqf(acqf, bounds=bounds, q=1,
                                            num_restarts=5, raw_samples=64)
                    next_x = cand.detach()
                except Exception as exc:
                    logger.debug("BO step fell back to random: %s", exc)
                    next_x = torch.tensor(rng.random((1, d)), dtype=dtype)
                score = evaluator(decode(next_x.numpy().ravel(), kind))
                train_x = torch.cat([train_x, next_x])
                train_y = torch.cat([train_y, torch.tensor([[score]], dtype=dtype)])
                advance()
            kbest_idx = int(torch.argmax(train_y))
            if float(train_y[kbest_idx]) > global_best:
                global_best = float(train_y[kbest_idx])
                global_best_params = decode(train_x[kbest_idx].numpy(), kind)
    return global_best_params, global_best, None


def _run_random(evaluator, space, n_trials, random_state, progress):
    from ._progress import progress_bar
    rng = np.random.default_rng(random_state)

    def sample():
        def logu(lo, hi):
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        p = {
            "decay_kind": rng.choice(list(space.decay_kinds)),
            "decay_scale": logu(*space.decay_scale),
            "alpha": logu(*space.alpha),
            "kappa0": logu(*space.kappa0),
            "nu0_offset": float(rng.uniform(*space.nu0_offset)),
            "psi_scale": logu(*space.psi_scale),
        }
        if space.n_pca is not None:
            p["n_pca"] = int(rng.integers(space.n_pca[0], space.n_pca[1] + 1))
        return p

    best, best_score = None, -np.inf
    with progress_bar("Random search", total=n_trials, disable=not progress) as advance:
        for _ in range(n_trials):
            params = sample()
            s = evaluator(params)
            if s > best_score:
                best_score, best = s, params
            advance()
    return best, best_score, None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def autotune_ddcrp(
    X: np.ndarray,
    adjacency_list: Sequence[np.ndarray],
    *,
    backend: Optional[str] = None,
    n_trials: int = 40,
    objective: str = "silhouette",
    spatial_distance: Optional[np.ndarray] = None,
    distances: Optional[Sequence[np.ndarray]] = None,
    vertices: Optional[np.ndarray] = None,
    search_space: Optional[SearchSpace] = None,
    eval_draws: int = 30,
    eval_burn_in: int = 20,
    eval_thin: int = 2,
    eval_chains: int = 1,
    random_state: int = 0,
    progress: bool = True,
    refit: bool = True,
    refit_kwargs: Optional[dict] = None,
) -> TuningResult:
    """Optimise ddCRP hyperparameters from the data.

    Parameters
    ----------
    X : np.ndarray, shape (V, d)
        Per-vertex features to cluster (e.g. MFA-fused HKS/WKS, or fPCA scores).
    adjacency_list : sequence of int arrays
        Mesh neighbours per vertex (contiguity).
    backend : {"optuna", "hyperopt", "botorch", "random", None}
        Optimiser backend; None auto-selects the first installed
        (optuna > hyperopt > botorch), else random search.
    n_trials : int
        Number of hyperparameter evaluations.
    objective : str
        Partition-quality criterion; see :mod:`brainmosaic.tuning.objective`.
    spatial_distance : np.ndarray, optional
        Geodesic/graph distance for the spatial silhouette objective.
    distances : sequence of arrays, optional
        Per-neighbour distances for the decay prior (defaults to ones).
    vertices : np.ndarray, optional, shape (V, 3)
        Mesh coordinates. If given (and ``distances`` is None), real Euclidean
        edge distances are used and the ``decay_scale`` search bounds are scaled
        to the mesh's median edge length, making the decay physically meaningful.
    search_space : SearchSpace, optional
        Override the default bounds.
    eval_draws, eval_burn_in, eval_thin, eval_chains : int
        Reduced ddCRP budget per trial (kept small for speed).
    random_state : int
    progress : bool
    refit : bool
        If True, refit the ddCRP at full budget with the best params.
    refit_kwargs : dict, optional
        Overrides for the final refit (e.g. ``n_draws``, ``chains``).

    Returns
    -------
    TuningResult
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; choose {VALID_OBJECTIVES}.")

    # If mesh vertices are supplied, derive real edge distances once so the decay
    # function modulates contiguity physically, and scale the decay_scale search
    # bounds to the mesh's median edge length (works for mm meshes and unit spheres).
    if distances is None and vertices is not None:
        from ._meshgeom import edge_distances
        distances = edge_distances(vertices, adjacency_list)
        if search_space is None:
            nonempty = [d for d in distances if np.size(d)]
            med = float(np.median(np.concatenate(nonempty))) if nonempty else 1.0
            search_space = SearchSpace(decay_scale=(0.5 * med, 12.0 * med))

    space = search_space or SearchSpace()
    tuner = _resolve_tuner(backend)
    logger.info("Autotuning ddCRP with backend=%s, n_trials=%d, objective=%s",
                tuner, n_trials, objective)

    evaluator = _Evaluator(
        X, adjacency_list, objective=objective, spatial_distance=spatial_distance,
        distances=distances, eval_draws=eval_draws, eval_burn_in=eval_burn_in,
        eval_thin=eval_thin, eval_chains=eval_chains, random_state=random_state)

    driver = {"optuna": _run_optuna, "hyperopt": _run_hyperopt,
              "botorch": _run_botorch, "random": _run_random}[tuner]
    best_params, best_score, study = driver(
        evaluator, space, n_trials, random_state, progress)

    result = TuningResult(best_params=best_params, best_score=float(best_score),
                          history=evaluator.history, backend=tuner, study=study)

    if refit and best_params is not None:
        kw = dict(n_draws=200, burn_in=100, thin=2, chains=4)
        kw.update(refit_kwargs or {})
        Xr = _maybe_pca(np.asarray(X, float), best_params.get("n_pca"))
        prior = NIWPrior.from_data(
            Xr, kappa0=float(best_params["kappa0"]),
            nu0_offset=float(best_params["nu0_offset"]),
            psi_scale=float(best_params["psi_scale"]))
        result.refit_result = cluster_ddcrp(
            Xr, adjacency_list, decay_kind=best_params["decay_kind"],
            decay_scale=float(best_params["decay_scale"]),
            alpha=float(best_params["alpha"]), prior=prior, distances=distances,
            random_state=random_state, progress=progress, **kw)
    return result


def autotune_ddcrp_functional(
    curves_blocks: Sequence[np.ndarray],
    adjacency_list: Sequence[np.ndarray],
    *,
    n_fpca: int = 5,
    **kwargs,
) -> TuningResult:
    """Autotune the *functional* ddCRP: fPCA-compress curves, then tune on scores.

    Compresses each HKS/WKS curve block with fPCA (MFA-balanced across blocks),
    then runs :func:`autotune_ddcrp` on the fused scores. All keyword arguments
    of :func:`autotune_ddcrp` are accepted.

    Parameters
    ----------
    curves_blocks : sequence of (V, T_b) arrays
    adjacency_list : sequence of int arrays
    n_fpca : int
        fPCA components per block.

    Returns
    -------
    TuningResult
    """
    from .ddcrp_functional import fpca_compress

    score_blocks = []
    for curves in curves_blocks:
        scores, _ = fpca_compress(curves, n_components=n_fpca)
        sv = np.linalg.svd(scores - scores.mean(0, keepdims=True), compute_uv=False)
        scale = float(max(sv[0], 1e-12)) if sv.size else 1.0
        score_blocks.append(scores / scale)
    X = np.hstack(score_blocks)
    return autotune_ddcrp(X, adjacency_list, **kwargs)
