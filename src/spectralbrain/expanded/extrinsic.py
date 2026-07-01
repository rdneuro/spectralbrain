"""Extrinsic spectral operators: Dirac, Steklov, and the shape operator.

The Laplace–Beltrami operator is **intrinsic** — invariant under isometric
bending — and therefore blind to how a surface sits in space.  Cortical
gyrification, hippocampal infolding, and subcortical bossing are
*extrinsic* phenomena.  This module supplies the three classical extrinsic
spectral views:

Dirac operator
    A quaternionic first-order operator whose **real (scalar) part is
    exactly the cotangent Laplacian** and whose imaginary part encodes the
    surface normal via edge cross-products.  Its signed spectrum and the
    derived **Dirac Kernel Signature** capture extrinsic bending that the
    LBO cannot see (Liu, Jacobson & Crane, *SGP* 2017; Crane, Pinkall &
    Schröder, *SIGGRAPH* 2011).

Steklov / Dirichlet-to-Neumann
    For meshes **with boundary**, the DtN operator maps boundary values to
    their normal derivative of the harmonic extension; its eigenvalues (the
    Steklov spectrum) are a boundary-sensitive shape signature.  Built as
    the Schur complement of the cotangent Laplacian.

Shape operator (Weingarten map)
    Per-vertex principal curvatures and their rotation-invariants
    (mean/Gaussian/Casorati curvature, shape index) — the most direct,
    robust extrinsic descriptors, reused from
    :mod:`spectralbrain.expanded._base`.

All operators thread the multi-backend solver
(:func:`spectralbrain.expanded._base.solve_eigsh`); the Dirac operator is
**indefinite**, so its spectrum is solved without the non-negativity
clamping used for the Laplacian.

References
----------
Liu HTD, Jacobson A, Crane K. "A Dirac operator for extrinsic shape
analysis." *Computer Graphics Forum (SGP)* 36(5):139–149, 2017.
Crane K, Pinkall U, Schröder P. "Spin transformations of discrete
surfaces." *ACM TOG (SIGGRAPH)* 30(4):104, 2011.
Wang Y, Ben-Chen M, Polterovich I, Solomon J. "Steklov spectral geometry
for extrinsic shape analysis." *ACM TOG* 38(1):7, 2019.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    Faces,
    MassMatrix,
    SparseMatrix,
    Vertices,
    get_logger,
)

from spectralbrain.expanded._base import (
    BackendSpec,
    _validate_mesh,
    face_areas,
    principal_curvatures,
    resolve_backend,
    solve_eigsh,
)

logger = get_logger(__name__)

_EPS = 1e-12


# ======================================================================
# §1  SHAPE-OPERATOR (WEINGARTEN) DESCRIPTORS
# ======================================================================


def shape_operator_descriptor(
    vertices: Vertices,
    faces: Faces,
    *,
    channels: tuple[str, ...] = (
        "k1",
        "k2",
        "mean",
        "gaussian",
        "casorati",
        "shape_index",
    ),
) -> DescriptorMatrix:
    """Per-vertex extrinsic descriptor from the shape operator (Weingarten map).

    Stacks rotation-invariant curvature channels derived from the principal
    curvatures ``κ₁ ≥ κ₂`` (Rusinkiewicz estimation):

    ===========  ===================================================
    channel      definition
    ===========  ===================================================
    ``k1``       maximum principal curvature κ₁
    ``k2``       minimum principal curvature κ₂
    ``mean``     H = (κ₁ + κ₂) / 2
    ``gaussian`` K = κ₁ κ₂
    ``casorati`` C = √((κ₁² + κ₂²) / 2)  (≥ 0)
    ``shape_index`` Koenderink S ∈ [-1, 1]
    ``curvedness``  log-curvedness ln(C + ε)
    ===========  ===================================================

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    channels : tuple of str
        Which channels to include and in what order.

    Returns
    -------
    ndarray, shape (N, len(channels))
        The stacked extrinsic descriptor.
    """
    k1, k2, _, _ = principal_curvatures(vertices, faces)
    casor = np.sqrt(0.5 * (k1**2 + k2**2))
    denom = k1 - k2
    shape_idx = np.zeros_like(k1)
    nz = np.abs(denom) > _EPS
    shape_idx[nz] = (2.0 / np.pi) * np.arctan((k1[nz] + k2[nz]) / denom[nz])

    available: dict[str, np.ndarray] = {
        "k1": k1,
        "k2": k2,
        "mean": 0.5 * (k1 + k2),
        "gaussian": k1 * k2,
        "casorati": casor,
        "shape_index": shape_idx,
        "curvedness": np.log(casor + _EPS),
    }
    unknown = set(channels) - set(available)
    if unknown:
        raise ValueError(f"Unknown shape-operator channels: {sorted(unknown)}.")
    return np.column_stack([available[c] for c in channels])


# ======================================================================
# §2  DIRAC OPERATOR  (quaternionic, real 4N representation)
# ======================================================================
#
# A quaternion q = w + xi + yj + zk acts on H by left multiplication; its
# real 4×4 matrix representation L(q) is a ring homomorphism with
# L(q̄) = L(q)ᵀ.  We assemble the Hermitian quaternionic Dirac matrix
# D[a][b] = (e_a e_b) / (4 A_f) (summed over faces), where e_a, e_b are the
# triangle's opposite-edge vectors as pure-imaginary quaternions.  Because
# Re(e_a e_b) = -(e_a · e_b), the scalar channel of D is exactly the
# cotangent Laplacian; the imaginary channels carry e_a × e_b = 2 A_f n_f
# (the extrinsic normal information).


def _quat_left_blocks(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> list[list[np.ndarray]]:
    """Return the 4×4 left-multiplication matrix L(q) as nested arrays.

    L(q) = [[ w, -x, -y, -z],
            [ x,  w, -z,  y],
            [ y,  z,  w, -x],
            [ z, -y,  x,  w]]   (each entry is an (F,) array).
    """
    return [
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w],
    ]


def dirac_operator(
    vertices: Vertices,
    faces: Faces,
) -> tuple[SparseMatrix, MassMatrix]:
    """Assemble the extrinsic Dirac operator in real ``4N × 4N`` form.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    D4 : sparse matrix, shape (4N, 4N)
        Symmetric real representation of the Hermitian quaternionic Dirac
        operator.  Its scalar channel ``D4[::4, ::4]`` equals the cotangent
        Laplacian.
    M4 : sparse matrix, shape (4N, 4N)
        Block mass matrix ``diag(area) ⊗ I₄`` (lumped vertex areas).
    """
    v, f = _validate_mesh(vertices, faces)
    n = v.shape[0]

    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    # opposite-edge vectors (pure-imaginary quaternions)
    e = {
        0: v[i2] - v[i1],
        1: v[i0] - v[i2],
        2: v[i1] - v[i0],
    }
    fa = face_areas(v, f)  # (F,)
    inv4a = 1.0 / np.clip(4.0 * fa, _EPS, None)  # (F,)
    idx = {0: i0, 1: i1, 2: i2}

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for a in range(3):
        ua = e[a]  # (F, 3)
        for b in range(3):
            vb = e[b]  # (F, 3)
            # quaternion product of two pure-imaginary quaternions ua, vb:
            #   ua * vb = (-(ua·vb)) + (ua × vb)
            # The Dirac entry is D[a][b] = -(ua vb)/(4A) so that its scalar
            # part is +(ua·vb)/(4A) = L_cot[a][b] (cotangent Laplacian), i.e.
            # Re(D) = L exactly; the imaginary part carries -(ua×vb)/(4A).
            w = np.sum(ua * vb, axis=1) * inv4a  # scalar part = +L (F,)
            cr = -np.cross(ua, vb) * inv4a[:, None]  # vector part (F, 3)
            qx, qy, qz = cr[:, 0], cr[:, 1], cr[:, 2]

            blocks = _quat_left_blocks(w, qx, qy, qz)  # 4×4 of (F,)
            base_r = 4 * idx[a]  # (F,)
            base_c = 4 * idx[b]
            for r in range(4):
                for c in range(4):
                    rows.append(base_r + r)
                    cols.append(base_c + c)
                    vals.append(blocks[r][c])

    D4 = sp.coo_matrix(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(4 * n, 4 * n),
    ).tocsc()
    D4 = 0.5 * (D4 + D4.T)  # scrub fp asymmetry (Hermitian by construction)

    # lumped vertex areas → block mass diag(area) ⊗ I₄
    point_area = np.zeros(n, dtype=np.float64)
    for c in range(3):
        np.add.at(point_area, f[:, c], fa / 3.0)
    point_area = np.clip(point_area, _EPS, None)
    M4 = sp.diags(np.repeat(point_area, 4), format="csc")
    return D4, M4


def dirac_decompose(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 80,
    backend: BackendSpec | Any = "auto",
    sigma: float = 1e-4,
) -> SpectralDecomposition:
    """Eigendecompose the extrinsic Dirac operator near zero.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Number of Dirac modes nearest 0 (the spectrum clusters around 0).
    backend : str or backend object
        Multi-backend solver selector (the Dirac problem is indefinite).
    sigma : float
        Shift-invert target.

    Returns
    -------
    SpectralDecomposition
        ``eigenvalues`` are the **signed** Dirac eigenvalues (k,) — each
        physical eigenvalue carries the Kramers ×2 degeneracy of the real
        representation; ``eigenvectors`` hold the per-vertex spinor
        magnitudes ``|ψ(x)|`` (N, k), non-negative — **not** scalar LBO
        eigenvectors — suitable for the Dirac Kernel Signature.
        ``metadata["operator"] == "dirac"``.
    """
    v, f = _validate_mesh(vertices, faces)
    n = v.shape[0]
    D4, M4 = dirac_operator(v, f)

    be = resolve_backend(backend)
    k_solve = int(min(k, 4 * n - 2))
    evals4, evecs4 = solve_eigsh(
        be, D4, M4, k_solve, sigma=sigma, clamp_nonneg=False
    )

    # per-vertex quaternion magnitude |ψ_a|² = Σ_{c=0..3} ψ[4a+c]²
    comps = evecs4.reshape(n, 4, -1)  # (N, 4, k)
    psi_mag = np.sqrt(np.sum(comps**2, axis=1))  # (N, k)

    return SpectralDecomposition(
        eigenvalues=evals4,
        eigenvectors=psi_mag,
        stiffness=None,
        mass=None,
        surface_area=float(face_areas(v, f).sum()),
        metadata={
            "operator": "dirac",
            "backend": getattr(be, "name", str(be)),
            "n_vertices": int(n),
            "spinor_magnitude": True,
        },
    )


def compute_dks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 80,
    t_values: np.ndarray | None = None,
    n_times: int = 100,
    backend: BackendSpec | Any = "auto",
    normalize: bool = False,
) -> DescriptorMatrix:
    """Dirac Kernel Signature — extrinsic per-vertex spectral descriptor.

    Defines, for each vertex ``x`` and scale ``t``,

        DKS_t(x) = Σ_k e^{-t λ_k²} · |ψ_k(x)|²,

    where ``(λ_k, ψ_k)`` are the Dirac eigenpairs.  The ``λ²`` weighting
    handles the signed Dirac spectrum and makes the signature analogous to
    the heat kernel signature but **extrinsic** (sensitive to bending).

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Number of Dirac modes.
    t_values : ndarray (T,), optional
        Explicit time scales; auto-selected from the spectrum if ``None``.
    n_times : int
        Number of auto scales.
    backend : str or backend object
        Multi-backend selector.
    normalize : bool
        If ``True``, divide each vertex curve by its smallest-scale value.

    Returns
    -------
    ndarray, shape (N, T)
        Non-negative extrinsic signature.
    """
    decomp = dirac_decompose(vertices, faces, k=k, backend=backend)
    lam = decomp.eigenvalues  # (k,) signed
    mag = decomp.eigenvectors  # (N, k) magnitudes
    lam_sq = lam**2

    if t_values is None:
        nz = lam_sq[lam_sq > _EPS]
        if nz.size:
            t_min = 4.0 * np.log(10.0) / nz.max()
            t_max = 4.0 * np.log(10.0) / nz.min()
            t_values = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
        else:
            t_values = np.logspace(-2, 2, n_times)
    t_values = np.asarray(t_values, dtype=np.float64)  # (T,)

    decay = np.exp(-lam_sq[None, :] * t_values[:, None])  # (T, k)
    dks = (mag**2) @ decay.T  # (N, T)
    dks = np.clip(dks, 0.0, None)
    if normalize:
        dks = dks / np.clip(dks[:, :1], _EPS, None)
    return dks


# ======================================================================
# §3  STEKLOV / DIRICHLET-TO-NEUMANN  (meshes with boundary)
# ======================================================================


def _boundary_vertices(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Indices of boundary vertices (on edges incident to exactly one face)."""
    edges = np.vstack(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    return np.unique(boundary_edges.ravel())


def _boundary_mass(
    vertices: np.ndarray, faces: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    """1-D lumped boundary mass (half incident boundary-edge length)."""
    edges = np.vstack(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bedges = uniq[counts == 1]  # (E_b, 2)
    lengths = np.linalg.norm(vertices[bedges[:, 0]] - vertices[bedges[:, 1]], axis=1)

    pos = {int(vidx): i for i, vidx in enumerate(boundary)}
    mb = np.zeros(boundary.size, dtype=np.float64)
    for (a, b), L in zip(bedges, lengths):
        mb[pos[int(a)]] += 0.5 * L
        mb[pos[int(b)]] += 0.5 * L
    return np.clip(mb, _EPS, None)


def steklov_operator(
    vertices: Vertices,
    faces: Faces,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dirichlet-to-Neumann (Steklov) operator via the Schur complement.

    Partitions the cotangent Laplacian into boundary (B) and interior (I)
    blocks and forms the dense DtN operator on the boundary,

        S = L_BB − L_BI L_II⁻¹ L_IB,

    together with the 1-D boundary mass.  Requires a mesh **with boundary**.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    S : ndarray, shape (B, B)
        Dense symmetric DtN operator.
    M_B : ndarray, shape (B,)
        Boundary mass (diagonal).
    boundary : ndarray, shape (B,)
        Indices of the boundary vertices (into ``vertices``).

    Raises
    ------
    ValueError
        If the mesh is closed (no boundary).
    """
    from spectralbrain.core.meshes import _cotangent_laplacian

    v, f = _validate_mesh(vertices, faces)
    boundary = _boundary_vertices(f, v.shape[0])
    if boundary.size == 0:
        raise ValueError(
            "Steklov spectrum requires a mesh with boundary, but none was "
            "found (closed surface).  Use the Dirac or shape-operator "
            "descriptors for closed cortical/subcortical surfaces."
        )

    L, _ = _cotangent_laplacian(v, f)
    L = sp.csc_matrix(L)
    n = v.shape[0]
    is_b = np.zeros(n, dtype=bool)
    is_b[boundary] = True
    interior = np.where(~is_b)[0]

    L_BB = L[boundary][:, boundary].toarray()
    L_BI = L[boundary][:, interior]
    L_II = L[interior][:, interior].tocsc()
    L_IB = L[interior][:, boundary]

    from scipy.sparse.linalg import spsolve

    # X = L_II⁻¹ L_IB  (solve column-block)
    X = spsolve(L_II, L_IB.tocsc())
    if sp.issparse(X):
        X = X.toarray()
    schur = L_BB - (L_BI @ X)
    schur = 0.5 * (schur + schur.T)
    mb = _boundary_mass(v, f, boundary)
    return schur, mb, boundary


def _dense_geigh(
    S: np.ndarray,
    mb: np.ndarray,
    k: int,
    *,
    backend: BackendSpec | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense generalised symmetric eigensolve ``S φ = λ diag(mb) φ`` (k smallest).

    Multi-backend: standardises with ``D^{-1/2}`` and runs a dense ``eigh``
    on the requested device (torch/cupy/jax), falling back to SciPy.
    """
    dis = 1.0 / np.sqrt(np.clip(mb, _EPS, None))
    A = (S * dis[:, None]) * dis[None, :]
    A = 0.5 * (A + A.T)

    be = resolve_backend(backend)
    name = getattr(be, "name", "numpy")
    w = V = None
    try:
        if name == "torch":
            import torch

            t = torch.as_tensor(A, dtype=torch.float64, device=be.device)
            w_t, V_t = torch.linalg.eigh(t)
            w, V = w_t.cpu().numpy(), V_t.cpu().numpy()
        elif name == "cupy":
            import cupy as cp

            w_g, V_g = cp.linalg.eigh(cp.asarray(A))
            w, V = cp.asnumpy(w_g), cp.asnumpy(V_g)
        elif name == "jax":
            import jax.numpy as jnp

            w_j, V_j = jnp.linalg.eigh(jnp.asarray(A))
            w, V = np.asarray(w_j), np.asarray(V_j)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Steklov GPU eigh failed (%s) → SciPy", exc)
        w = None
    if w is None:
        from scipy.linalg import eigh

        w, V = eigh(A)

    order = np.argsort(w)[:k]
    evals = np.clip(w[order], 0.0, None)  # Steklov is PSD
    evecs = dis[:, None] * V[:, order]
    return evals, evecs


def steklov_spectrum(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 50,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Steklov eigenvalues/eigenfunctions on the mesh boundary.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Number of Steklov eigenpairs.
    backend : str or backend object
        Multi-backend selector for the dense boundary eigensolve.

    Returns
    -------
    SpectralDecomposition
        ``eigenvalues`` (k,) is the Steklov spectrum; ``eigenvectors``
        (B, k) are the boundary eigenfunctions; ``metadata`` records the
        boundary vertex indices and ``operator == "steklov"``.
    """
    S, mb, boundary = steklov_operator(vertices, faces)
    k_eff = int(min(k, max(1, S.shape[0] - 1)))
    evals, evecs = _dense_geigh(S, mb, k_eff, backend=backend)
    return SpectralDecomposition(
        eigenvalues=evals,
        eigenvectors=evecs,
        stiffness=None,
        mass=sp.diags(mb, format="csc"),
        surface_area=float(face_areas(*_validate_mesh(vertices, faces)).sum()),
        metadata={
            "operator": "steklov",
            "boundary_vertices": boundary,
            "n_boundary": int(boundary.size),
        },
    )


def compute_steklov_wks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 50,
    n_energies: int = 100,
    backend: BackendSpec | Any = "auto",
) -> DescriptorMatrix:
    """Wave Kernel Signature on the Steklov (boundary) spectrum.

    Returns
    -------
    ndarray, shape (B, n_energies)
        Per-boundary-vertex WKS over the Steklov eigensystem.
    """
    from spectralbrain.spectral.descriptors import compute_wks

    decomp = steklov_spectrum(vertices, faces, k=k, backend=backend)
    return compute_wks(decomp, n_energies=n_energies)


__all__ = [
    "compute_dks",
    "compute_steklov_wks",
    "dirac_decompose",
    "dirac_operator",
    "shape_operator_descriptor",
    "steklov_operator",
    "steklov_spectrum",
]
