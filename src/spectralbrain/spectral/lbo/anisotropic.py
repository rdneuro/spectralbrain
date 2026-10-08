"""Anisotropic spectral descriptors.

These descriptors replace the isotropic Laplace-Beltrami operator
with directional (anisotropic) variants, enabling sensitivity to
geometry along specific directions (e.g. principal curvature axes,
sulcal ridges, cortical lamination gradients).

Implemented
-----------
- **Anisotropic HKS / WKS** -- HKS/WKS computed from an anisotropic
  Laplacian that weights diffusion by principal curvature direction.
- **ASMWD** -- Anisotropic Spectral Manifold Wavelet Descriptor
  (Li et al. CGF 2021).

The Finsler-LBO (Jadhav & Cremers, CVPR 2024) is provided as a
Laplacian *constructor* that feeds into existing HKS/WKS/SGW
pipelines.

.. note::
   Principal curvature directions are estimated here from a local PCA of
   each vertex neighbourhood. For the more reliable Rusinkiewicz estimate
   use :func:`spectralbrain.spectral.operators.finsler.finsler_laplacian`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

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
    progress_simple,
)

logger = get_logger(__name__)


# ======================================================================
# S1  ANISOTROPIC LAPLACIAN CONSTRUCTION
# ======================================================================


def anisotropic_laplacian(
    vertices: Vertices,
    faces: Faces,
    *,
    anisotropy: float = 1.0,
    direction: Literal["max_curvature", "min_curvature", "custom"] = "max_curvature",
    custom_directions: np.ndarray | None = None,
) -> tuple[SparseMatrix, MassMatrix]:
    """Build an anisotropic Laplacian weighted by curvature direction.

    Modifies the cotangent Laplacian by scaling diffusion along the
    principal curvature directions.  When ``anisotropy=0``, recovers
    the standard isotropic Laplacian.  When ``anisotropy=1``, diffusion
    is maximally biased along the chosen direction.

    This is a simplified implementation inspired by Andreux, Rodola,
    Aubry & Cremers (NORDIA 2014) and the Finsler-LBO of Jadhav &
    Cremers (CVPR 2024).

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    anisotropy : float
        Anisotropy strength in [0, 1].  0 = isotropic, 1 = fully
        directional.
    direction : str
        ``"max_curvature"`` -- bias along maximum curvature direction.
        ``"min_curvature"`` -- bias along minimum curvature direction.
        ``"custom"`` -- use *custom_directions*.
    custom_directions : ndarray, shape (N, 3), optional
        Per-vertex preferred directions (unit vectors).

    Returns
    -------
    L : SparseMatrix, shape (N, N)
    M : MassMatrix, shape (N, N)
    """
    from spectralbrain.core.meshes import _cotangent_laplacian

    N = vertices.shape[0]

    if direction == "custom" and custom_directions is None:
        raise ValueError("custom_directions required for direction='custom'.")

    # Base cotangent Laplacian.
    L_iso, M = _cotangent_laplacian(vertices, faces)

    if abs(anisotropy) < 1e-10:
        return L_iso, M

    # Estimate per-vertex directions.
    if direction in ("max_curvature", "min_curvature"):
        dirs = _estimate_curvature_directions(
            vertices,
            faces,
            which="max" if direction == "max_curvature" else "min",
        )
    else:
        dirs = custom_directions

    # Modify edge weights by directional bias.
    # For edge (i, j), the anisotropic weight is:
    #   w_ij' = w_ij * (1 + alpha * 1/2(|d_i * e_ij|^2 + |d_j * e_ij|^2))
    # where d_i is the preferred direction at vertex i and e_ij is the
    # unit edge vector.  Averaging the bias over both endpoints keeps the
    # operator symmetric (w_ij' = w_ji'), as required by the symmetric
    # eigensolver.
    L_coo = sp.coo_matrix(L_iso)
    rows, cols, vals = L_coo.row, L_coo.col, L_coo.data.copy()

    # Only modify off-diagonal entries.
    off_diag = rows != cols
    r_off = rows[off_diag]
    c_off = cols[off_diag]

    edge_vecs = vertices[c_off] - vertices[r_off]  # (E, 3)
    edge_len = np.linalg.norm(edge_vecs, axis=1, keepdims=True)
    edge_unit = edge_vecs / np.clip(edge_len, 1e-12, None)

    dirs = np.asarray(dirs, dtype=np.float64)
    cos_sq_src = np.sum(dirs[r_off] * edge_unit, axis=1) ** 2  # (E,)
    cos_sq_dst = np.sum(dirs[c_off] * edge_unit, axis=1) ** 2  # (E,)
    cos_sq = 0.5 * (cos_sq_src + cos_sq_dst)

    # Scale off-diagonal weights.
    scale = 1.0 + anisotropy * cos_sq
    off_vals = vals[off_diag] * scale

    # Rebuild: off-diagonal part, then diagonal = -(off-diagonal row sum)
    # so that every row sums to zero.
    L_off = sp.coo_matrix((off_vals, (r_off, c_off)), shape=(N, N)).tocsc()
    L_off = 0.5 * (L_off + L_off.T)  # scrub fp asymmetry
    diag_vals = -np.asarray(L_off.sum(axis=1)).ravel()
    L_aniso = sp.csc_matrix(L_off + sp.diags(diag_vals, 0, format="csc"))

    logger.info(
        "Anisotropic Laplacian: alpha=%.2f, direction=%s",
        anisotropy,
        direction,
    )
    return L_aniso, M


def _estimate_curvature_directions(
    vertices: Vertices,
    faces: Faces,
    which: str = "max",
) -> np.ndarray:
    """Estimate per-vertex principal curvature directions via
    local quadric fitting.

    Simplified implementation: uses PCA of the 1-ring neighbourhood
    projected onto the tangent plane as a proxy for curvature directions.
    """
    from spectralbrain.core.base import knn_search
    from spectralbrain.core.meshes import _vertex_normals

    N = vertices.shape[0]
    normals = _vertex_normals(vertices, faces)
    _, indices = knn_search(vertices, k=15)

    directions = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        nbrs = vertices[indices[i]]  # (k, 3)
        n_i = normals[i]

        # Project onto tangent plane.
        centered = nbrs - vertices[i]
        proj = centered - np.outer(centered @ n_i, n_i)  # (k, 3)

        # PCA of projected neighbours.
        if np.linalg.norm(proj) < 1e-12:
            directions[i] = np.array([1, 0, 0])
            continue

        cov = proj.T @ proj
        _eigvals, eigvecs = np.linalg.eigh(cov)

        # eigvecs[:, 0] is (~) the normal: the projected neighbours have
        # no variance along it.  The two tangent axes are columns 1 and 2.
        if which == "max":
            directions[i] = eigvecs[:, -1]  # largest in-plane variance
        else:
            directions[i] = eigvecs[:, 1]  # smallest in-plane variance

    # Normalise.
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.clip(norms, 1e-12, None)
    return directions


# ======================================================================
# S2  ANISOTROPIC DESCRIPTORS
# ======================================================================


def compute_anisotropic_hks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 50,
    n_times: int = 50,
    anisotropy: float = 0.5,
    direction: str = "max_curvature",
) -> DescriptorMatrix:
    """HKS computed from an anisotropic Laplacian.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Number of eigenpairs.
    n_times : int
        Time samples.
    anisotropy : float
        Anisotropy strength [0, 1].
    direction : str
        Curvature direction bias.

    Returns
    -------
    ndarray, shape (N, T)
    """
    from spectralbrain.core.backends import NumpyBackend
    from spectralbrain.spectral.lbo.descriptors import compute_hks

    L, M = anisotropic_laplacian(
        vertices,
        faces,
        anisotropy=anisotropy,
        direction=direction,
    )
    be = NumpyBackend()
    evals, evecs = be.eigsh(L, M, k=k)
    decomp = SpectralDecomposition(evals, evecs, stiffness=L, mass=M)
    return compute_hks(decomp, n_times=n_times)


def compute_anisotropic_wks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 50,
    n_energies: int = 50,
    anisotropy: float = 0.5,
    direction: str = "max_curvature",
) -> DescriptorMatrix:
    """WKS computed from an anisotropic Laplacian.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
    n_energies : int
    anisotropy : float
    direction : str

    Returns
    -------
    ndarray, shape (N, E)
    """
    from spectralbrain.core.backends import NumpyBackend
    from spectralbrain.spectral.lbo.descriptors import compute_wks

    L, M = anisotropic_laplacian(
        vertices,
        faces,
        anisotropy=anisotropy,
        direction=direction,
    )
    be = NumpyBackend()
    evals, evecs = be.eigsh(L, M, k=k)
    decomp = SpectralDecomposition(evals, evecs, stiffness=L, mass=M)
    return compute_wks(decomp, n_energies=n_energies)


def compute_asmwd(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 50,
    n_scales: int = 5,
    n_directions: int = 4,
    anisotropy: float = 0.5,
    kernel: Callable | None = None,
) -> DescriptorMatrix:
    """Anisotropic Spectral Manifold Wavelet Descriptor (ASMWD).

    Computes wavelet descriptors along multiple anisotropic
    directions, concatenating the results.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Eigenpairs per direction.
    n_scales : int
        Wavelet scales.
    n_directions : int
        Number of interpolated directions between max and min
        curvature.
    anisotropy : float
    kernel : callable, optional
        Wavelet kernel.  Default: Mexican hat.

    Returns
    -------
    ndarray, shape (N, n_directions x n_scales)

    References
    ----------
    Li Q et al. Anisotropic spectral manifold wavelet descriptor.
    *Computer Graphics Forum* 40(7):261-272, 2021.
    """
    from spectralbrain.core.backends import NumpyBackend
    from spectralbrain.spectral.lbo.wavelets import mexican_hat_kernel, sgw_descriptor

    if kernel is None:
        kernel = mexican_hat_kernel

    # Estimate both curvature directions.
    dir_max = _estimate_curvature_directions(vertices, faces, "max")
    dir_min = _estimate_curvature_directions(vertices, faces, "min")
    # Principal directions are line fields (sign-ambiguous).  Fix a
    # right-handed tangent frame (dir_max, dir_min, n) so the interpolated
    # directions rotate consistently from vertex to vertex.
    from spectralbrain.core.meshes import _vertex_normals

    normals = _vertex_normals(vertices, faces)
    handed = np.sign(np.sum(np.cross(dir_max, dir_min) * normals, axis=1, keepdims=True))
    handed[handed == 0] = 1.0
    dir_min = dir_min * handed

    all_descs: list[np.ndarray] = []
    be = NumpyBackend()

    with progress_simple("ASMWD directions", total=n_directions) as tick:
        for d_idx in range(n_directions):
            # Rotate from the max- to the min-curvature axis in the
            # tangent plane (angle interpolation keeps unit length).
            alpha = d_idx / max(1, n_directions - 1)
            ang = 0.5 * np.pi * alpha
            custom_dir = np.cos(ang) * dir_max + np.sin(ang) * dir_min
            norms = np.linalg.norm(custom_dir, axis=1, keepdims=True)
            custom_dir /= np.clip(norms, 1e-12, None)

            L, M = anisotropic_laplacian(
                vertices,
                faces,
                anisotropy=anisotropy,
                direction="custom",
                custom_directions=custom_dir,
            )
            evals, evecs = be.eigsh(L, M, k=k)
            decomp = SpectralDecomposition(evals, evecs, stiffness=L, mass=M)
            desc = sgw_descriptor(decomp, n_scales=n_scales, kernel=kernel)
            all_descs.append(desc)
            tick(1)

    return np.hstack(all_descs)  # (N, n_dir x n_scales)


# ======================================================================

__all__: list[str] = [
    "anisotropic_laplacian",
    "compute_anisotropic_hks",
    "compute_anisotropic_wks",
    "compute_asmwd",
]
