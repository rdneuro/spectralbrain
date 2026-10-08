"""Extrinsic spectral operators: Dirac, Steklov, and the shape operator.

The Laplace-Beltrami operator is **intrinsic** -- invariant under isometric
bending -- and therefore blind to how a surface sits in space.  Cortical
gyrification, hippocampal infolding, and subcortical bossing are
*extrinsic* phenomena.  This module supplies the three classical extrinsic
spectral views:

Dirac operator
    The quaternionic (Crane-Pinkall-Schroder) Dirac operator assembled on
    vertices as ``D = D_f^H M_F D_f``, where ``D_f`` maps vertex spinors to
    faces via the edge quaternions.  Its **real (scalar) part is exactly
    the cotangent Laplacian** and its imaginary part encodes the surface
    normal via edge cross-products.  Being a Gram matrix (each face adds a
    rank-1 Hermitian term ``u u^H / 4A``) it is **positive semi-definite**:
    its eigenvalues ``mu >= 0`` are the *squared* first-order Dirac
    eigenvalues ``mu = lambda_D^2`` (the first-order operator, acting between
    vertices and faces, has the symmetric signed spectrum ``+/-sqrt mu``).  The
    derived **Dirac Kernel Signature** ``Sigma e^{-t mu}|psi|^2`` captures
    extrinsic bending that the LBO cannot see (Liu, Jacobson & Crane,
    *SGP* 2017; Crane, Pinkall & Schroder, *SIGGRAPH* 2011).

Steklov / Dirichlet-to-Neumann
    For meshes **with boundary**, the DtN operator maps boundary values to
    their normal derivative of the harmonic extension; its eigenvalues (the
    Steklov spectrum) are a boundary-sensitive shape signature.  Built as
    the Schur complement of the cotangent Laplacian.

Shape operator (Weingarten map)
    Per-vertex principal curvatures and their rotation-invariants
    (mean/Gaussian/Casorati curvature, shape index) -- the most direct,
    robust extrinsic descriptors, reused from
    :mod:`spectralbrain.spectral.operators._base`.

All operators thread the multi-backend solver
(:func:`spectralbrain.spectral.operators._base.solve_eigsh`).  The Dirac spectrum is
solved without the non-negativity clamping (round-off may give tiny
negative values around the 4-fold null space).

References
----------
Liu HTD, Jacobson A, Crane K. "A Dirac operator for extrinsic shape
analysis." *Computer Graphics Forum (SGP)* 36(5):139-149, 2017.
Crane K, Pinkall U, Schroder P. "Spin transformations of discrete
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
from spectralbrain.spectral.operators._base import (
    BackendSpec,
    _shape_index_from_k,
    _validate_mesh,
    face_areas,
    principal_curvatures,
    resolve_backend,
    solve_eigsh,
)

logger = get_logger(__name__)

_EPS = 1e-12


# ======================================================================
# S1  SHAPE-OPERATOR (WEINGARTEN) DESCRIPTORS
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
    curvatures ``kappa_1 >= kappa_2`` (Rusinkiewicz estimation):

    ===========  ===================================================
    channel      definition
    ===========  ===================================================
    ``k1``       maximum principal curvature kappa_1
    ``k2``       minimum principal curvature kappa_2
    ``mean``     H = (kappa_1 + kappa_2) / 2
    ``gaussian`` K = kappa_1 kappa_2
    ``casorati`` C = sqrt ((kappa_1^2 + kappa_2^2) / 2)  (>= 0)
    ``shape_index`` Koenderink S in [-1, 1]
    ``curvedness``  log-curvedness ln(C + eps)
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
    shape_idx = _shape_index_from_k(k1, k2)

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
# S2  DIRAC OPERATOR  (quaternionic, real 4N representation)
# ======================================================================
#
# A quaternion q = w + xi + yj + zk acts on H by left multiplication; its
# real 4x4 matrix representation L(q) is a ring homomorphism with
# L(qbar) = L(q)^T.  We assemble the Hermitian quaternionic Dirac matrix
# D[a][b] = (e_a e_b) / (4 A_f) (summed over faces), where e_a, e_b are the
# triangle's opposite-edge vectors as pure-imaginary quaternions.  Because
# Re(e_a e_b) = -(e_a * e_b), the scalar channel of D is exactly the
# cotangent Laplacian; the imaginary channels carry e_a x e_b = 2 A_f n_f
# (the extrinsic normal information).


def _quat_left_blocks(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> list[list[np.ndarray]]:
    """Return the 4x4 left-multiplication matrix L(q) as nested arrays.

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
    """Assemble the extrinsic Dirac operator in real ``4N x 4N`` form.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)

    Returns
    -------
    D4 : sparse matrix, shape (4N, 4N)
        Symmetric real representation of the Hermitian quaternionic Dirac
        operator ``D = D_f^H M_F D_f``.  Its scalar channel ``D4[::4, ::4]``
        equals the cotangent Laplacian.  It is positive semi-definite (its
        eigenvalues are the squared first-order Dirac eigenvalues).
    M4 : sparse matrix, shape (4N, 4N)
        Block mass matrix ``diag(area) (x) I_4`` (lumped vertex areas).
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
            #   ua * vb = (-(ua*vb)) + (ua x vb)
            # The Dirac entry is D[a][b] = -(ua vb)/(4A) so that its scalar
            # part is +(ua*vb)/(4A) = L_cot[a][b] (cotangent Laplacian), i.e.
            # Re(D) = L exactly; the imaginary part carries -(uaxvb)/(4A).
            w = np.sum(ua * vb, axis=1) * inv4a  # scalar part = +L (F,)
            cr = -np.cross(ua, vb) * inv4a[:, None]  # vector part (F, 3)
            qx, qy, qz = cr[:, 0], cr[:, 1], cr[:, 2]

            blocks = _quat_left_blocks(w, qx, qy, qz)  # 4x4 of (F,)
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

    # lumped vertex areas -> block mass diag(area) (x) I_4
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
        Number of Dirac modes nearest 0 -- i.e. the ``k`` smallest, since
        the assembled operator is positive semi-definite.
    backend : str or backend object
        Multi-backend solver selector (the Dirac problem is indefinite).
    sigma : float
        Shift-invert target.

    Returns
    -------
    SpectralDecomposition
        ``eigenvalues`` (k,) are the eigenvalues ``mu = lambda_D^2 >= 0`` of the
        assembled (squared) Dirac operator -- each carries the x4
        quaternionic degeneracy of the real representation;
        ``eigenvectors`` hold the per-vertex spinor magnitudes ``|psi(x)|``
        (N, k), non-negative -- **not** scalar LBO eigenvectors -- suitable
        for the Dirac Kernel Signature.  ``metadata["operator"] ==
        "dirac"`` and ``metadata["squared_dirac"] is True``.
    """
    v, f = _validate_mesh(vertices, faces)
    n = v.shape[0]
    D4, M4 = dirac_operator(v, f)

    be = resolve_backend(backend)
    k_solve = int(min(k, 4 * n - 2))
    evals4, evecs4 = solve_eigsh(be, D4, M4, k_solve, sigma=sigma, clamp_nonneg=False)

    # per-vertex quaternion magnitude |psi_a|^2 = Sigma_{c=0..3} psi[4a+c]^2
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
            "squared_dirac": True,
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
    """Dirac Kernel Signature -- extrinsic per-vertex spectral descriptor.

    Defines, for each vertex ``x`` and scale ``t``,

        DKS_t(x) = Sigma_k e^{-t lambda_k^2} * |psi_k(x)|^2 = Sigma_k e^{-t mu_k} * |psi_k(x)|^2,

    where ``lambda_k`` are the (signed) first-order Dirac eigenvalues and
    ``mu_k = lambda_k^2`` the eigenvalues returned by :func:`dirac_decompose`
    (the assembled operator is already the squared Dirac, so ``mu`` is used
    directly -- squaring it again would give ``e^{-t lambda^4}``).  The signature
    is the heat kernel signature of the squared Dirac operator, but
    **extrinsic** (sensitive to bending).

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
    # mu = lambda_D^2 (squared Dirac); round-off negatives near the null space -> 0
    lam_sq = np.clip(decomp.eigenvalues, 0.0, None)  # (k,)
    mag = decomp.eigenvectors  # (N, k) magnitudes

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
# S3  STEKLOV / DIRICHLET-TO-NEUMANN  (meshes with boundary)
# ======================================================================


def _boundary_vertices(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Indices of boundary vertices (on edges incident to exactly one face)."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    return np.unique(boundary_edges.ravel())


def _boundary_mass(vertices: np.ndarray, faces: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """1-D lumped boundary mass (half incident boundary-edge length)."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
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

        S = L_BB - L_BI L_II^-1 L_IB,

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

    # X = L_II^-1 L_IB  (solve column-block)
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
    """Dense generalised symmetric eigensolve ``S phi = lambda diag(mb) phi`` (k smallest).

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
    except Exception as exc:
        logger.debug("Steklov GPU eigh failed (%s) -> SciPy", exc)
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
    from spectralbrain.spectral.lbo.descriptors import compute_wks

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
