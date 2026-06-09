"""Anisotropic spectral descriptors.

These descriptors replace the isotropic Laplace–Beltrami operator
with directional (anisotropic) variants, enabling sensitivity to
geometry along specific directions (e.g. principal curvature axes,
sulcal ridges, cortical lamination gradients).

Implemented
-----------
- **Anisotropic HKS / WKS** — HKS/WKS computed from an anisotropic
  Laplacian that weights diffusion by principal curvature direction.
- **ASMWD** — Anisotropic Spectral Manifold Wavelet Descriptor
  (Li et al. CGF 2021).

The Finsler-LBO (Jadhav & Cremers, CVPR 2024) is provided as a
Laplacian *constructor* that feeds into existing HKS/WKS/SGW
pipelines.

.. note::
   Anisotropic descriptors require **mesh** topology (faces) to
   estimate principal curvature directions.  Point cloud support
   is experimental via local PCA axes.
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
# §1  ANISOTROPIC LAPLACIAN CONSTRUCTION
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

    This is a simplified implementation inspired by Andreux, Rodolà,
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
        ``"max_curvature"`` — bias along maximum curvature direction.
        ``"min_curvature"`` — bias along minimum curvature direction.
        ``"custom"`` — use *custom_directions*.
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
    #   w_ij' = w_ij · (1 + α · |d_i · e_ij|²)
    # where d_i is the preferred direction at vertex i and
    # e_ij is the unit edge vector.
    L_coo = sp.coo_matrix(L_iso)
    rows, cols, vals = L_coo.row, L_coo.col, L_coo.data.copy()

    # Only modify off-diagonal entries.
    off_diag = rows != cols
    r_off = rows[off_diag]
    c_off = cols[off_diag]

    edge_vecs = vertices[c_off] - vertices[r_off]  # (E, 3)
    edge_len = np.linalg.norm(edge_vecs, axis=1, keepdims=True)
    edge_unit = edge_vecs / np.clip(edge_len, 1e-12, None)

    # Directional bias at source vertex.
    dir_at_src = dirs[r_off]  # (E, 3)
    cos_sq = np.sum(dir_at_src * edge_unit, axis=1) ** 2  # (E,)

    # Scale off-diagonal weights.
    scale = 1.0 + anisotropy * cos_sq
    vals[off_diag] *= scale

    # Rebuild and fix diagonal (row sum = 0).
    L_aniso = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    # Reset diagonal.
    diag_vals = -np.asarray(L_aniso.sum(axis=1)).ravel()
    L_aniso.setdiag(0)
    L_aniso = L_aniso + sp.diags(diag_vals, 0, format="csc")

    logger.info(
        "Anisotropic Laplacian: α=%.2f, direction=%s",
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

        if which == "max":
            directions[i] = eigvecs[:, -1]  # largest variance
        else:
            directions[i] = eigvecs[:, 0]  # smallest variance

    # Normalise.
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.clip(norms, 1e-12, None)
    return directions


# ======================================================================
# §2  ANISOTROPIC DESCRIPTORS
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
    from spectralbrain.backends.cpu import NumpyBackend
    from spectralbrain.spectral.descriptors import compute_hks

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
    from spectralbrain.backends.cpu import NumpyBackend
    from spectralbrain.spectral.descriptors import compute_wks

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
    ndarray, shape (N, n_directions × n_scales)

    References
    ----------
    Li Q et al. Anisotropic spectral manifold wavelet descriptor.
    *Computer Graphics Forum* 40(7):261–272, 2021.
    """
    from spectralbrain.backends.cpu import NumpyBackend
    from spectralbrain.spectral.wavelets import (
        mexican_hat_kernel,
        sgw_descriptor,
    )

    if kernel is None:
        kernel = mexican_hat_kernel

    # Estimate both curvature directions.
    dir_max = _estimate_curvature_directions(vertices, faces, "max")
    dir_min = _estimate_curvature_directions(vertices, faces, "min")

    all_descs: list[np.ndarray] = []
    be = NumpyBackend()

    with progress_simple("ASMWD directions", total=n_directions) as tick:
        for d_idx in range(n_directions):
            # Interpolate between max and min curvature.
            alpha = d_idx / max(1, n_directions - 1)
            custom_dir = (1 - alpha) * dir_max + alpha * dir_min
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

    return np.hstack(all_descs)  # (N, n_dir × n_scales)


# ======================================================================

__all__: list[str] = [
    "anisotropic_laplacian",
    "compute_anisotropic_hks",
    "compute_anisotropic_wks",
    "compute_asmwd",
]
