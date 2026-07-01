"""Shared foundation for the :mod:`spectralbrain.expanded` operator family.

This module is the *trunk* on top of which every non-Laplace–Beltrami
operator in :mod:`spectralbrain.expanded` is built.  The design mirrors
the layered architecture of the library:

    geometry  →  operator assembly (A, M)  →  eigensolve  →  descriptor

The Laplace–Beltrami operator (LBO) is one instance of the *operator
assembly* step.  Every operator here (Hamiltonian, Dirac, Steklov, …)
produces a sparse generalised eigenproblem ``A v = λ M v`` that is solved
by the **same** backend eigensolver used by the core library
(:meth:`spectralbrain.backends.cpu.NumpyBackend.eigsh` and its GPU
mirrors), and the resulting eigenpairs are wrapped in the **same**
central handoff object (:class:`spectralbrain.core.base.SpectralDecomposition`).
This means existing descriptors (HKS, WKS, GPS, …) apply verbatim to the
eigenpairs of *any* operator — e.g. a Dirac Kernel Signature is simply
``compute_hks`` evaluated on a Dirac ``SpectralDecomposition``.

What lives here
---------------
- :func:`resolve_backend` — turn ``"auto"|"numpy"|"torch"|"cupy"|"jax"``
  (or a backend object) into a concrete backend exposing ``.eigsh``.
- :func:`operator_eigensystem` — the shared assemble→solve→wrap routine
  every operator submodule calls.
- :func:`operator_signature` — reuse the LBO descriptor machinery (HKS,
  WKS, GPS) on an arbitrary operator's spectrum.
- Geometry shared by extrinsic / anisotropic / potential operators:
  :func:`face_areas`, :func:`vertex_normals`, :func:`principal_curvatures`
  (Rusinkiewicz 2004), and the derived :func:`mean_curvature`,
  :func:`gaussian_curvature`, :func:`casorati_curvature`,
  :func:`shape_index`.
- :func:`require_optional` — lazy soft-dependency loader with an
  install hint, and :func:`gpu_memory_guard` — a no-op-safe VRAM guard.

Notes
-----
The :mod:`spectralbrain.expanded` subpackage adds *optional* dependencies
(``gudhi``, ``ripser``, ``persim``, ``GraphRicciCurvature``, ``pyefd``,
``robust_laplacian``, ``potpourri3d``, ``pot``).  None are imported at
package import time; they are loaded lazily by :func:`require_optional`
only when the specific descriptor that needs them is called.  Install the
full set with ``pip install 'spectralbrain[expanded]'``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from spectralbrain.backends.cpu import NumpyBackend
from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    Faces,
    MassMatrix,
    Normals,
    ScalarMap,
    SparseMatrix,
    Vertices,
    get_logger,
)

logger = get_logger(__name__)

BackendSpec = Literal["auto", "numpy", "scipy", "cpu", "torch", "cupy", "jax"]
"""Accepted backend selectors for :func:`resolve_backend`.

A backend *object* exposing ``.eigsh`` (e.g. an already-constructed
:class:`~spectralbrain.backends.gpu.TorchBackend`) is also accepted.
"""

_EPS = 1e-12


# ======================================================================
# §1  OPTIONAL-DEPENDENCY LOADER
# ======================================================================


def require_optional(
    module: str,
    *,
    purpose: str = "",
    extra: str = "expanded",
) -> Any:
    """Import an optional dependency or raise an actionable error.

    Parameters
    ----------
    module : str
        Importable module name (e.g. ``"gudhi"``, ``"GraphRicciCurvature"``).
    purpose : str, optional
        Short phrase describing what the module is needed for; included in
        the error message.
    extra : str
        The pip *extra* that bundles the dependency (default
        ``"expanded"``).

    Returns
    -------
    module
        The imported module object.

    Raises
    ------
    ImportError
        If the module is not installed, with an install hint.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only w/o dep
        tail = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"The optional dependency {module!r} is required{tail}, "
            f"but it is not installed.  Install it with:\n"
            f"    pip install 'spectralbrain[{extra}]'\n"
            f"or directly:  pip install {module}"
        ) from exc


# ======================================================================
# §2  BACKEND RESOLUTION
# ======================================================================


def resolve_backend(backend: BackendSpec | Any = "auto") -> Any:
    """Return a concrete compute backend exposing ``.eigsh``.

    Parameters
    ----------
    backend : str or backend object
        One of ``"auto"``, ``"numpy"``/``"scipy"``/``"cpu"`` (CPU),
        ``"torch"``, ``"cupy"``, ``"jax"`` (GPU), or an object that
        already implements the backend protocol (returned unchanged).

        ``"auto"`` selects a GPU backend **only if one is already importable
        and a device is available**, otherwise falls back to the CPU
        :class:`~spectralbrain.backends.cpu.NumpyBackend`.  GPU backends are
        never imported speculatively at module import time.

    Returns
    -------
    backend
        A backend object with an ``.eigsh(L, M, k, ...)`` method and a
        ``.name`` attribute, matching :class:`NumpyBackend`.

    Notes
    -----
    All backend ``eigsh`` implementations return **host** (NumPy) arrays,
    so the choice of backend does not change the dtype/location of the
    eigenpairs flowing into :class:`SpectralDecomposition`; it affects only
    where the eigensolve and (downstream) descriptor arithmetic run.
    """
    if backend is None:
        backend = "auto"

    # Already a backend object?
    if hasattr(backend, "eigsh"):
        return backend

    name = str(backend).lower()

    if name in ("numpy", "scipy", "cpu"):
        return NumpyBackend()

    if name in ("torch", "cupy", "jax"):
        from spectralbrain.backends.gpu import get_gpu_backend

        return get_gpu_backend(name)

    if name == "auto":
        for cand in ("torch", "cupy", "jax"):
            try:
                from spectralbrain.backends.gpu import get_gpu_backend

                be = get_gpu_backend(cand)
            except Exception as exc:  # noqa: BLE001 - any failure → next cand
                logger.debug("expanded: backend %s unavailable (%s)", cand, exc)
                continue
            logger.info("expanded: auto backend → %s", be.name)
            return be
        logger.debug("expanded: auto backend → numpy (no GPU backend available)")
        return NumpyBackend()

    raise ValueError(
        f"Unknown backend {backend!r}.  Expected one of "
        f"'auto', 'numpy', 'scipy', 'cpu', 'torch', 'cupy', 'jax', "
        f"or a backend object with an .eigsh method."
    )


@contextmanager
def gpu_memory_guard(label: str = "expanded op") -> Generator[None, None, None]:
    """VRAM-reporting guard that degrades to a no-op without a GPU backend.

    Wraps :func:`spectralbrain.backends.gpu.vram_guard` when importable,
    otherwise yields a null context.  Safe to wrap around any operator
    assembly / eigensolve regardless of backend.

    Parameters
    ----------
    label : str
        Label reported alongside the VRAM delta.
    """
    try:
        from spectralbrain.backends.gpu import vram_guard
    except Exception:  # noqa: BLE001 - GPU stack absent → no-op
        with nullcontext():
            yield
        return
    with vram_guard(label):
        yield


def free_gpu_memory() -> None:
    """Best-effort GPU cache release; no-op when no GPU backend is present."""
    try:
        from spectralbrain.backends.gpu import vram_gc

        vram_gc()
    except Exception:  # noqa: BLE001
        pass


# ======================================================================
# §2b  GENERALISED EIGENSOLVER  (multi-backend; PSD or indefinite)
# ======================================================================
#
# The core-library backends (`be.eigsh`) clamp eigenvalues to ≥ 0 — the
# correct behaviour for the positive-semidefinite Laplacian and its PSD
# relatives (Hamiltonian with V ≥ 0, biharmonic).  Some expanded operators
# (e.g. the Dirac operator) are **indefinite**: their negative eigenvalues
# are physically meaningful and must not be clamped.  `solve_eigsh` routes
# PSD problems to the fast, tested native backend solver and indefinite
# problems to a non-clamping multi-backend path that selects the ``k``
# eigenvalues nearest ``sigma`` (the Dirac spectrum clusters around 0).


def _scipy_eigsh_nearest(
    A: SparseMatrix,
    M: MassMatrix | None,
    k: int,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU shift-invert ARPACK targeting eigenvalues nearest ``sigma`` (no clamp)."""
    import scipy.sparse.linalg as spla

    A = sp.csc_matrix(A, dtype=np.float64)
    Mm = sp.csc_matrix(M, dtype=np.float64) if M is not None else None
    try:
        evals, evecs = spla.eigsh(A, k=k, M=Mm, sigma=sigma, which="LM")
    except (spla.ArpackNoConvergence, RuntimeError) as exc:
        logger.debug("indefinite eigsh: nudging sigma after %s", exc)
        evals, evecs = spla.eigsh(A, k=k, M=Mm, sigma=sigma + 1e-6, which="LM")
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def _gpu_eigh_nearest(
    be: Any,
    A: SparseMatrix,
    M: MassMatrix | None,
    k: int,
    *,
    sigma: float,
    dense_max: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Densified standardised symmetric eigh on a GPU backend (no clamp).

    Mirrors the core ``TorchBackend.eigsh`` standardisation
    ``Ã = D^{-1/2} A D^{-1/2}`` but keeps the **full signed** spectrum and
    selects the ``k`` eigenvalues nearest ``sigma``.  Falls back to CPU
    shift-invert for large ``N`` or any device failure.  Returns host arrays.
    """
    n = A.shape[0]
    if n > dense_max:
        return _scipy_eigsh_nearest(A, M, k, sigma=sigma)

    name = getattr(be, "name", "numpy")
    d = np.asarray(M.diagonal(), float) if M is not None else np.ones(n)
    d = np.clip(d, 1e-20, None)
    dis = 1.0 / np.sqrt(d)
    Ad = sp.csc_matrix(A, dtype=np.float64).toarray()
    Ad *= dis[:, None]
    Ad *= dis[None, :]
    Ad = 0.5 * (Ad + Ad.T)

    try:
        if name == "torch":
            import torch

            t = torch.as_tensor(Ad, dtype=torch.float64, device=be.device)
            w_t, V_t = torch.linalg.eigh(t)
            w = w_t.detach().cpu().numpy()
            V = V_t.detach().cpu().numpy()
            del t, w_t, V_t
            if getattr(be.device, "type", "") == "cuda":
                torch.cuda.empty_cache()
        elif name == "cupy":
            import cupy as cp

            t = cp.asarray(Ad)
            w_g, V_g = cp.linalg.eigh(t)
            w = cp.asnumpy(w_g)
            V = cp.asnumpy(V_g)
            del t, w_g, V_g
            cp.get_default_memory_pool().free_all_blocks()
        elif name == "jax":
            import jax.numpy as jnp

            w_j, V_j = jnp.linalg.eigh(jnp.asarray(Ad))
            w = np.asarray(w_j)
            V = np.asarray(V_j)
        else:  # unknown GPU-ish backend → CPU
            return _scipy_eigsh_nearest(A, M, k, sigma=sigma)
    except Exception as exc:  # noqa: BLE001 - any device failure → CPU
        logger.debug("indefinite GPU eigh failed (%s) → CPU shift-invert", exc)
        return _scipy_eigsh_nearest(A, M, k, sigma=sigma)

    idx = np.argsort(np.abs(w - sigma))[:k]  # k nearest sigma
    idx = idx[np.argsort(w[idx])]  # then ascending
    evals = w[idx]
    evecs = dis[:, None] * V[:, idx]  # de-standardise → M-orthonormal
    return evals, evecs


def solve_eigsh(
    be: Any,
    A: SparseMatrix,
    M: MassMatrix | None,
    k: int,
    *,
    sigma: float = -0.01,
    which: str = "LM",
    clamp_nonneg: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised symmetric eigensolve ``A v = λ M v`` on any backend.

    Parameters
    ----------
    be : backend object
        A resolved backend (see :func:`resolve_backend`).
    A, M : sparse matrices
        Operator and (optional) mass matrix.
    k : int
        Number of eigenpairs.
    sigma : float
        Shift-invert target.  For ``clamp_nonneg=False`` this is the point
        the spectrum is sought *around* (0 for the Dirac operator).
    which : str
        Forwarded to the native PSD solver (ignored on the indefinite path).
    clamp_nonneg : bool
        ``True`` (default) → PSD path via the native ``be.eigsh`` (clamps
        tiny negatives to 0, returns the ``k`` smallest eigenvalues).
        ``False`` → indefinite path: full signed spectrum, ``k`` eigenvalues
        nearest ``sigma``, multi-backend (CPU ARPACK / GPU dense eigh).

    Returns
    -------
    (eigenvalues, eigenvectors) : ndarrays
        ``(k,)`` and ``(N, k)``, host arrays, ascending in eigenvalue,
        M-orthonormal eigenvectors.
    """
    if clamp_nonneg:
        return be.eigsh(A, M, k=k, sigma=sigma, which=which)

    name = getattr(be, "name", "numpy")
    if name in ("numpy", "scipy", "cpu"):
        return _scipy_eigsh_nearest(A, M, k, sigma=sigma)
    return _gpu_eigh_nearest(be, A, M, k, sigma=sigma)


# ======================================================================
# §3  OPERATOR → EIGENSYSTEM → DESCRIPTOR
# ======================================================================


def _validate_operator_pair(
    A: SparseMatrix,
    M: MassMatrix | None,
) -> tuple[sp.csc_matrix, sp.csc_matrix | None]:
    """Coerce ``(A, M)`` to CSC float64 and validate square/shape/finite."""
    A = sp.csc_matrix(A).astype(np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Operator A must be square, got shape {A.shape}.")
    if not np.all(np.isfinite(A.data)):
        raise ValueError("Operator A contains non-finite entries (NaN/Inf).")
    if M is not None:
        M = sp.csc_matrix(M).astype(np.float64)
        if M.shape != A.shape:
            raise ValueError(
                f"Mass matrix shape {M.shape} != operator shape {A.shape}."
            )
        if not np.all(np.isfinite(M.data)):
            raise ValueError("Mass matrix M contains non-finite entries.")
    return A, M


def operator_eigensystem(
    A: SparseMatrix,
    M: MassMatrix | None = None,
    *,
    k: int = 100,
    backend: BackendSpec | Any = "auto",
    sigma: float = -0.01,
    which: str = "LM",
    symmetrize: bool = True,
    clamp_nonneg: bool = True,
    surface_area: float | None = None,
    operator: str = "custom",
    metadata: dict[str, Any] | None = None,
) -> SpectralDecomposition:
    """Solve ``A v = λ M v`` and wrap the result as a SpectralDecomposition.

    This is the single assemble→solve→wrap entry point shared by every
    operator in :mod:`spectralbrain.expanded`.  The returned object is
    indistinguishable in type from an LBO decomposition, so all existing
    descriptors consume it directly.

    Parameters
    ----------
    A : sparse matrix, shape (N, N)
        The operator (stiffness-analogue).  May be indefinite for some
        operators (e.g. Dirac); set ``which``/``sigma`` accordingly.
    M : sparse matrix, shape (N, N), optional
        Mass matrix.  ``None`` → standard eigenproblem ``A v = λ v``.
    k : int
        Number of eigenpairs.  Internally clamped to ``min(k, N - 2)`` to
        stay within ARPACK's limit.
    backend : str or backend object
        See :func:`resolve_backend`.
    sigma : float
        Shift for shift-invert mode (passed to the backend).
    which : str
        Which eigenvalues to target (``"LM"`` of the shifted operator →
        smallest λ for PSD operators; use ``"SA"`` for indefinite ones).
    symmetrize : bool
        If ``True``, replace ``A`` by ``½(A + Aᵀ)`` to scrub floating-point
        asymmetry before the symmetric eigensolver.  Disable only when the
        operator is exactly symmetric by construction.
    clamp_nonneg : bool
        ``True`` (default) for positive-semidefinite operators (clamps tiny
        negative eigenvalues to 0 and returns the ``k`` smallest).  Set
        ``False`` for **indefinite** operators (e.g. Dirac) to keep the full
        signed spectrum and select the ``k`` eigenvalues nearest ``sigma``.
    surface_area : float, optional
        Stored on the decomposition for area-normalised ShapeDNA.
    operator : str
        Name tag recorded in ``metadata["operator"]`` (e.g. ``"hamiltonian"``).
    metadata : dict, optional
        Extra provenance merged into the decomposition metadata.

    Returns
    -------
    SpectralDecomposition
        With ``stiffness=A``, ``mass=M`` and
        ``metadata["operator"]=operator``.
    """
    A, M = _validate_operator_pair(A, M)
    n = A.shape[0]

    k_eff = int(min(k, max(1, n - 2)))
    if k_eff != k:
        logger.warning(
            "operator_eigensystem: clamped k=%d → %d for N=%d vertices.",
            k,
            k_eff,
            n,
        )

    if symmetrize:
        A = (A + A.T) * 0.5
        A = sp.csc_matrix(A)

    be = resolve_backend(backend)
    evals, evecs = solve_eigsh(
        be, A, M, k_eff, sigma=sigma, which=which, clamp_nonneg=clamp_nonneg
    )

    meta = {
        "operator": operator,
        "backend": getattr(be, "name", str(be)),
        "n_vertices": int(n),
        "k": int(k_eff),
    }
    if metadata:
        meta.update(metadata)

    return SpectralDecomposition(
        eigenvalues=evals,
        eigenvectors=evecs,
        stiffness=A,
        mass=M,
        surface_area=surface_area,
        metadata=meta,
    )


def operator_signature(
    decomp: SpectralDecomposition,
    kind: Literal["hks", "wks", "gps", "shapedna"] = "hks",
    **kwargs: Any,
) -> DescriptorMatrix:
    """Apply an LBO descriptor to *any* operator's eigenpairs.

    This is what makes non-LBO operators "free" once assembled: the heat
    kernel signature of a Dirac operator (the Dirac Kernel Signature), or
    of a Hamiltonian, is just :func:`spectralbrain.spectral.compute_hks`
    evaluated on the corresponding spectrum.

    Parameters
    ----------
    decomp : SpectralDecomposition
        Eigenpairs of any operator assembled via
        :func:`operator_eigensystem`.
    kind : str
        ``"hks"``, ``"wks"``, ``"gps"`` or ``"shapedna"``.
    **kwargs
        Forwarded to the underlying descriptor (e.g. ``n_times`` for HKS,
        ``n_energies`` for WKS).

    Returns
    -------
    ndarray
        Per-vertex descriptor ``(N, C)`` (``hks``/``wks``/``gps``) or the
        global ShapeDNA vector.

    Notes
    -----
    HKS/WKS assume a *non-negative* spectrum (heat/wave equations).  For
    indefinite operators (Dirac), prefer a signature built on
    ``exp(t·λ)`` of the squared/absolute spectrum — see
    :mod:`spectralbrain.expanded.extrinsic`, which provides the
    operator-appropriate kernel signature.
    """
    from spectralbrain.spectral.descriptors import (
        compute_gps,
        compute_hks,
        compute_shapedna,
        compute_wks,
    )

    dispatch: dict[str, Callable[..., Any]] = {
        "hks": compute_hks,
        "wks": compute_wks,
        "gps": compute_gps,
        "shapedna": compute_shapedna,
    }
    if kind not in dispatch:
        raise ValueError(f"Unknown signature kind {kind!r}; expected one of {list(dispatch)}.")
    return dispatch[kind](decomp, **kwargs)


# ======================================================================
# §4  SHARED GEOMETRY  (face areas, normals, principal curvatures)
# ======================================================================


def _validate_mesh(vertices: Vertices, faces: Faces) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a triangle mesh; return (vertices f64, faces i64)."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {v.shape}.")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3) triangles, got {f.shape}.")
    if f.size and f.max() >= v.shape[0]:
        raise ValueError(
            f"face index {int(f.max())} out of range for {v.shape[0]} vertices."
        )
    if f.size and f.min() < 0:
        raise ValueError("faces contain negative indices.")
    return v, f


def face_areas(vertices: Vertices, faces: Faces) -> np.ndarray:
    """Per-face triangle areas.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    ndarray, shape (F,)
        Triangle areas (mm²), non-negative.
    """
    v, f = _validate_mesh(vertices, faces)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)  # (F, 3)
    return 0.5 * np.linalg.norm(cross, axis=1)  # (F,)


def vertex_normals(vertices: Vertices, faces: Faces) -> Normals:
    """Area-weighted unit vertex normals.

    Delegates to the core library's implementation when available and
    falls back to a local area-weighted accumulation otherwise.

    Returns
    -------
    ndarray, shape (N, 3)
        Unit normals (rows of near-zero norm are returned as +z).
    """
    v, f = _validate_mesh(vertices, faces)
    try:
        from spectralbrain.core.meshes import _vertex_normals

        return np.asarray(_vertex_normals(v, f), dtype=np.float64)
    except Exception:  # noqa: BLE001 - fall back to local implementation
        pass

    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)  # area-weighted face normals (F, 3)
    vn = np.zeros_like(v)
    for c in range(3):
        np.add.at(vn, f[:, c], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    safe = norms[:, 0] < _EPS
    vn[~safe] /= norms[~safe]
    vn[safe] = np.array([0.0, 0.0, 1.0])
    return vn


def _rotate_coord_sys(
    up: np.ndarray,
    vp: np.ndarray,
    new_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an orthonormal tangent basis (up, vp) onto a new normal.

    Vectorised port of Rusinkiewicz's ``rotate_coord_sys``: rotate the
    frame about the axis (old_norm × new_norm) so that it becomes
    perpendicular to ``new_norm`` while staying as close as possible to
    the original frame.

    Parameters
    ----------
    up, vp : ndarray, shape (F, 3)
        Old orthonormal tangent vectors (old_norm = up × vp implicitly).
    new_norm : ndarray, shape (F, 3)
        Target unit normals.

    Returns
    -------
    (new_up, new_vp) : ndarrays, shape (F, 3)
    """
    old_norm = np.cross(up, vp)  # (F, 3)
    ndot = np.sum(old_norm * new_norm, axis=1, keepdims=True)  # (F, 1)

    # ndot ≤ -1 → antiparallel: flip the frame.
    flip = (ndot <= -1.0 + 1e-9)[:, 0]
    new_up = up.copy()
    new_vp = vp.copy()
    if np.any(flip):
        new_up[flip] = -up[flip]
        new_vp[flip] = -vp[flip]

    perp = old_norm - new_norm * ndot  # (F, 3)
    denom = 1.0 + ndot  # (F, 1)
    dperp = (old_norm + new_norm) / np.clip(denom, _EPS, None)  # (F, 3)

    proc = ~flip
    if np.any(proc):
        up_p, vp_p = up[proc], vp[proc]
        dp = dperp[proc]
        new_up[proc] = up_p - dp * np.sum(perp[proc] * up_p, axis=1, keepdims=True)
        new_vp[proc] = vp_p - dp * np.sum(perp[proc] * vp_p, axis=1, keepdims=True)
    return new_up, new_vp


def _project_curvature(
    old_u: np.ndarray,
    old_v: np.ndarray,
    ku: np.ndarray,
    kuv: np.ndarray,
    kv: np.ndarray,
    new_u: np.ndarray,
    new_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-express a 2nd-fundamental-form tensor in a new tangent basis.

    Vectorised port of Rusinkiewicz's ``proj_curv``.

    Parameters
    ----------
    old_u, old_v : ndarray, shape (F, 3)
        Basis in which (ku, kuv, kv) is expressed.
    ku, kuv, kv : ndarray, shape (F,)
        Components of the symmetric 2×2 form [[ku, kuv], [kuv, kv]].
    new_u, new_v : ndarray, shape (F, 3)
        Target basis (must be perpendicular to the same normal).

    Returns
    -------
    (new_ku, new_kuv, new_kv) : ndarrays, shape (F,)
    """
    r_up, r_vp = _rotate_coord_sys(new_u, new_v, np.cross(old_u, old_v))
    u1 = np.sum(r_up * old_u, axis=1)
    v1 = np.sum(r_up * old_v, axis=1)
    u2 = np.sum(r_vp * old_u, axis=1)
    v2 = np.sum(r_vp * old_v, axis=1)
    new_ku = ku * u1 * u1 + 2.0 * kuv * u1 * v1 + kv * v1 * v1
    new_kuv = ku * u1 * u2 + kuv * (u1 * v2 + u2 * v1) + kv * v1 * v2
    new_kv = ku * u2 * u2 + 2.0 * kuv * u2 * v2 + kv * v2 * v2
    return new_ku, new_kuv, new_kv


def principal_curvatures(
    vertices: Vertices,
    faces: Faces,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex principal curvatures and directions (Rusinkiewicz 2004).

    Estimates the second fundamental form per face from the variation of
    vertex normals along the triangle edges, then transports and averages
    the per-face tensors into per-vertex tangent frames and diagonalises.

    Sign convention: curvature is positive where the surface bends *toward*
    the outward normal (convex).  ``k1`` is the larger principal curvature.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    k1 : ndarray, shape (N,)
        Maximum principal curvature (κ₁ ≥ κ₂).
    k2 : ndarray, shape (N,)
        Minimum principal curvature.
    dir1 : ndarray, shape (N, 3)
        Principal direction associated with ``k1`` (unit, tangent).
    dir2 : ndarray, shape (N, 3)
        Principal direction associated with ``k2`` (unit, tangent).

    References
    ----------
    Rusinkiewicz S. "Estimating curvatures and their derivatives on
    triangle meshes." *3DPVT* 2004.
    """
    v, f = _validate_mesh(vertices, faces)
    n_v = v.shape[0]
    vn = vertex_normals(v, f)  # (N, 3)

    # ── per-vertex initial tangent frames ─────────────────────────────
    # Build pdir1 from an arbitrary world axis projected to the tangent
    # plane, robust to the normal being aligned with +x.
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (n_v, 1))
    aligned = np.abs(vn[:, 0]) > 0.9
    ref[aligned] = np.array([0.0, 1.0, 0.0])
    pdir1 = np.cross(ref, vn)
    pdir1 /= np.clip(np.linalg.norm(pdir1, axis=1, keepdims=True), _EPS, None)
    pdir2 = np.cross(vn, pdir1)  # (N, 3)

    # ── per-vertex barycentric area weights ───────────────────────────
    fa = face_areas(v, f)  # (F,)
    point_area = np.zeros(n_v, dtype=np.float64)
    for c in range(3):
        np.add.at(point_area, f[:, c], fa / 3.0)
    point_area = np.clip(point_area, _EPS, None)

    # ── per-face second fundamental form via edge normal differences ──
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    e0 = v[i2] - v[i1]  # opposite vertex 0
    e1 = v[i0] - v[i2]
    e2 = v[i1] - v[i0]

    fn = np.cross(e0, e1)
    fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), _EPS, None)  # (F, 3)
    uf = e0 / np.clip(np.linalg.norm(e0, axis=1, keepdims=True), _EPS, None)
    vf = np.cross(fn, uf)  # (F, 3)

    edges = (e0, e1, e2)
    dns = (vn[i2] - vn[i1], vn[i0] - vn[i2], vn[i1] - vn[i0])

    # Least squares for [[m0, m1], [m1, m2]] (3 unknowns) from 3 edges
    # (6 equations).  Accumulate the 3×3 normal matrix AᵀA and AᵀA·rhs.
    n_f = f.shape[0]
    ata = np.zeros((n_f, 3, 3), dtype=np.float64)
    atb = np.zeros((n_f, 3), dtype=np.float64)
    for e, dn in zip(edges, dns):
        ev_u = np.sum(e * uf, axis=1)  # (F,)
        ev_v = np.sum(e * vf, axis=1)
        dn_u = np.sum(dn * uf, axis=1)
        dn_v = np.sum(dn * vf, axis=1)
        # Row A: [ev_u, ev_v, 0]   (= dn_u)
        # Row B: [0, ev_u, ev_v]   (= dn_v)
        a0 = np.stack([ev_u, ev_v, np.zeros_like(ev_u)], axis=1)  # (F, 3)
        a1 = np.stack([np.zeros_like(ev_u), ev_u, ev_v], axis=1)  # (F, 3)
        ata += a0[:, :, None] * a0[:, None, :]
        ata += a1[:, :, None] * a1[:, None, :]
        atb += a0 * dn_u[:, None]
        atb += a1 * dn_v[:, None]

    # Solve per-face (regularise for degenerate faces).
    ata += np.eye(3)[None, :, :] * 1e-9
    m = np.linalg.solve(ata, atb[:, :, None])[:, :, 0]  # (F, 3) → [m0, m1, m2]
    fku, fkuv, fkv = m[:, 0], m[:, 1], m[:, 2]

    # ── transport face tensors into vertex frames and accumulate ──────
    acc_ku = np.zeros(n_v, dtype=np.float64)
    acc_kuv = np.zeros(n_v, dtype=np.float64)
    acc_kv = np.zeros(n_v, dtype=np.float64)
    acc_w = np.zeros(n_v, dtype=np.float64)

    for c in range(3):
        vidx = f[:, c]
        nku, nkuv, nkv = _project_curvature(
            uf, vf, fku, fkuv, fkv, pdir1[vidx], pdir2[vidx]
        )
        w = fa / (3.0 * point_area[vidx])  # corner weight (Rusinkiewicz)
        np.add.at(acc_ku, vidx, w * nku)
        np.add.at(acc_kuv, vidx, w * nkuv)
        np.add.at(acc_kv, vidx, w * nkv)
        np.add.at(acc_w, vidx, w)

    safe = acc_w > _EPS
    acc_ku[safe] /= acc_w[safe]
    acc_kuv[safe] /= acc_w[safe]
    acc_kv[safe] /= acc_w[safe]

    # ── diagonalise the per-vertex 2×2 symmetric form ─────────────────
    # eigenvalues of [[a, b], [b, c]]
    a, b, c = acc_ku, acc_kuv, acc_kv
    tr = a + c
    disc = np.sqrt(np.clip((a - c) ** 2 + 4.0 * b**2, 0.0, None))
    k1 = 0.5 * (tr + disc)  # larger
    k2 = 0.5 * (tr - disc)  # smaller

    # principal directions in the (pdir1, pdir2) basis
    # eigenvector for k1: [b, k1 - a] (or [k1 - c, b]); guard degeneracy
    ex = b.copy()
    ey = k1 - a
    deg = (np.abs(ex) < _EPS) & (np.abs(ey) < _EPS)
    ex[deg] = 1.0
    ey[deg] = 0.0
    enorm = np.sqrt(ex**2 + ey**2)
    enorm = np.clip(enorm, _EPS, None)
    ex /= enorm
    ey /= enorm
    dir1 = ex[:, None] * pdir1 + ey[:, None] * pdir2
    dir2 = np.cross(vn, dir1)
    dir1 /= np.clip(np.linalg.norm(dir1, axis=1, keepdims=True), _EPS, None)
    dir2 /= np.clip(np.linalg.norm(dir2, axis=1, keepdims=True), _EPS, None)

    return k1, k2, dir1, dir2


def mean_curvature(vertices: Vertices, faces: Faces) -> ScalarMap:
    """Per-vertex mean curvature ``H = (κ₁ + κ₂) / 2`` (extrinsic)."""
    k1, k2, _, _ = principal_curvatures(vertices, faces)
    return 0.5 * (k1 + k2)


def gaussian_curvature(vertices: Vertices, faces: Faces) -> ScalarMap:
    """Per-vertex Gaussian curvature ``K = κ₁·κ₂`` (intrinsic, Egregium).

    Estimated as the product of the Rusinkiewicz principal curvatures.  For
    the angle-deficit (purely intrinsic) estimator, compute it from face
    angles instead; this product form is convenient when ``k1``/``k2`` are
    already needed.
    """
    k1, k2, _, _ = principal_curvatures(vertices, faces)
    return k1 * k2


def casorati_curvature(vertices: Vertices, faces: Faces) -> ScalarMap:
    """Per-vertex Casorati curvature ``C = sqrt((κ₁² + κ₂²) / 2)``.

    A non-negative, rotation-invariant magnitude of bending (Koenderink),
    convenient as an extrinsic potential because it is always ≥ 0.
    """
    k1, k2, _, _ = principal_curvatures(vertices, faces)
    return np.sqrt(0.5 * (k1**2 + k2**2))


def shape_index(vertices: Vertices, faces: Faces) -> ScalarMap:
    """Per-vertex Koenderink shape index in [-1, 1].

    ``S = (2/π)·arctan((κ₁ + κ₂) / (κ₁ − κ₂))`` with ``κ₁ ≥ κ₂`` — a
    scale-invariant descriptor of *local shape* independent of curvature
    magnitude: spherical cap → +1, ridge → +½, saddle → 0, rut → −½,
    spherical cup → −1.  Returns 0 at umbilic/flat points (κ₁ = κ₂).

    Signs follow the mesh's normal orientation (outward for FreeSurfer /
    HippUnfold surfaces ⇒ convex regions read positive).
    """
    k1, k2, _, _ = principal_curvatures(vertices, faces)
    denom = k1 - k2  # ≥ 0 since k1 ≥ k2
    s = np.zeros_like(k1)
    nz = np.abs(denom) > _EPS
    s[nz] = (2.0 / np.pi) * np.arctan((k1[nz] + k2[nz]) / denom[nz])
    return s


__all__ = [
    "BackendSpec",
    "casorati_curvature",
    "face_areas",
    "free_gpu_memory",
    "gaussian_curvature",
    "gpu_memory_guard",
    "mean_curvature",
    "operator_eigensystem",
    "operator_signature",
    "principal_curvatures",
    "require_optional",
    "resolve_backend",
    "shape_index",
    "solve_eigsh",
    "vertex_normals",
]
