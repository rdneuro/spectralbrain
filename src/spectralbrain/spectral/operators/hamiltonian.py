"""Hamiltonian / Schrodinger spectral operators ``H = Delta + V``.

The Hamiltonian is the cheapest non-LBO operator: it is the
Laplace-Beltrami operator plus a per-vertex potential ``V`` that acts
only on the **diagonal** of the stiffness matrix.  Sparsity, assembly
cost, and the generalised eigenproblem are therefore identical to the
LBO -- making this a true drop-in extension of the existing pipeline.
Choosing ``V`` lets the eigenbasis be *steered* or *localised*:

- ``V = 0``                     -> recovers the standard LBO.
- ``V = curvature`` (extrinsic) -> injects bending/folding sensitivity
  into an otherwise intrinsic basis (the key for cortical gyrification).
- ``V = region mask``           -> focuses the spectrum on a sub-region
  (e.g. a hippocampal subfield).
- ``V = 1 / (2|phi|)`` style       -> compressed / localised manifold modes
  (Neumann et al. 2014; Melzi et al. 2018).

Discretisation
--------------
With ``W`` the cotangent stiffness and ``A`` the (lumped) mass matrix,
the discrete Hamiltonian generalised eigenproblem is

    (W + A*diag(V)) phi  =  lambda A phi.

Because ``V`` only perturbs the diagonal of ``W`` (weighted by the mass),
the operator stays sparse with the same nonzero pattern as the LBO.

Descriptors
-----------
Any LBO descriptor applies to a Hamiltonian decomposition.  We expose:

- :func:`hamiltonian_operator`     -- assemble ``(H, A)``.
- :func:`hamiltonian_decompose`    -- assemble + eigensolve -> decomposition.
- :func:`compute_siwks`            -- Scale-Invariant WKS on the Hamiltonian
  spectrum (Fan et al., *Medical Image Analysis* 2021).
- :func:`compute_localized_hks` / :func:`compute_localized_wks`
  -- HKS/WKS from a region- or curvature-steered Hamiltonian.
- :func:`compute_compressed_modes` -- localised manifold harmonics.

References
----------
Choukroun Y, Shtern A, Bronstein A, Kimmel R. "Hamiltonian operator for
spectral shape analysis." *IEEE TVCG* 26(2):1320-1331, 2020.
Fan Y et al. "Tetrahedral spectral feature-based Bayesian manifold
learning for grey matter morphometry." *Medical Image Analysis*
73:102014, 2021.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from spectralbrain.core.base import SpectralDecomposition
from spectralbrain.runtime import (
    DescriptorMatrix,
    Faces,
    MassMatrix,
    ScalarMap,
    SparseMatrix,
    Vertices,
    get_logger,
)
from spectralbrain.spectral.operators._base import (
    BackendSpec,
    casorati_curvature,
    mean_curvature,
    operator_eigensystem,
    principal_curvatures,
)

logger = get_logger(__name__)

PotentialSpec = Literal["zero", "curvature", "abs_mean_curvature", "casorati", "gaussian"]
"""Built-in potential generators for :func:`build_potential`."""

_EPS = 1e-12


# ======================================================================
# S1  POTENTIALS
# ======================================================================


def build_potential(
    vertices: Vertices,
    faces: Faces,
    *,
    kind: PotentialSpec = "casorati",
    scale: float = 1.0,
    normalize: bool = True,
) -> ScalarMap:
    """Construct a per-vertex potential ``V`` from mesh geometry.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    kind : str
        ``"zero"`` -- V == 0 (recovers LBO).
        ``"curvature"`` -- signed mean curvature H (extrinsic, can be +/-).
        ``"abs_mean_curvature"`` -- |H| (extrinsic, >= 0).
        ``"casorati"`` -- Casorati curvature sqrt ((kappa_1^2+kappa_2^2)/2) (extrinsic, >= 0).
        ``"gaussian"`` -- Gaussian curvature K = kappa_1kappa_2 (intrinsic).
    scale : float
        Multiplies the potential after optional normalisation.
    normalize : bool
        If ``True``, divide the (non-zero) potential by its mean absolute
        value so that ``scale`` has a comparable meaning across meshes.

    Returns
    -------
    ndarray, shape (N,)
        The per-vertex potential V.

    Notes
    -----
    A non-negative potential (``casorati``, ``abs_mean_curvature``) keeps
    the Hamiltonian positive semi-definite, which is numerically the safest
    choice for the shift-invert eigensolver and gives a clean heat/wave
    interpretation.  Signed potentials localise eigenfunctions toward
    curvature extrema of one sign; :func:`hamiltonian_decompose` then
    solves the unclamped spectrum (eigenvalues may be negative).

    With ``normalize=True`` the potential is dimensionless (mean ``|V|``
    = 1) while the LBO eigenvalues carry units of 1/length^2; choose
    ``scale`` relative to the spectrum (e.g. a multiple of lambda_1) when the
    mesh is in physical units such as mm.
    """
    n = vertices.shape[0]
    if kind == "zero":
        return np.zeros(n, dtype=np.float64)
    if kind == "curvature":
        v = mean_curvature(vertices, faces)
    elif kind == "abs_mean_curvature":
        v = np.abs(mean_curvature(vertices, faces))
    elif kind == "casorati":
        v = casorati_curvature(vertices, faces)
    elif kind == "gaussian":
        k1, k2, _, _ = principal_curvatures(vertices, faces)
        v = k1 * k2
    else:
        raise ValueError(f"Unknown potential kind {kind!r}.")

    v = np.asarray(v, dtype=np.float64)
    if normalize:
        denom = np.mean(np.abs(v))
        if denom > _EPS:
            v = v / denom
    return scale * v


def region_potential(
    n_vertices: int,
    region_mask: np.ndarray,
    *,
    barrier: float = 100.0,
    inside_value: float = 0.0,
) -> ScalarMap:
    """Build a binary 'well' potential that confines modes to a region.

    Vertices **outside** ``region_mask`` receive a large potential
    (``barrier``), so Hamiltonian eigenfunctions with ``lambda < barrier`` decay
    there and concentrate inside the region -- a soft Dirichlet well useful
    for subfield-localised spectra.

    Parameters
    ----------
    n_vertices : int
        Total number of vertices.
    region_mask : ndarray of bool, shape (N,)
        ``True`` for vertices *inside* the region of interest.
    barrier : float
        Potential height outside the region (the confinement strength).
    inside_value : float
        Potential inside the region (usually 0).

    Returns
    -------
    ndarray, shape (N,)
    """
    mask = np.asarray(region_mask, dtype=bool)
    if mask.shape != (n_vertices,):
        raise ValueError(f"region_mask shape {mask.shape} != (n_vertices={n_vertices},).")
    v = np.full(n_vertices, barrier, dtype=np.float64)
    v[mask] = inside_value
    return v


# ======================================================================
# S2  OPERATOR ASSEMBLY
# ======================================================================


def hamiltonian_operator(
    stiffness: SparseMatrix,
    mass: MassMatrix,
    potential: ScalarMap,
) -> SparseMatrix:
    """Assemble the discrete Hamiltonian ``H = W + A*diag(V)``.

    Parameters
    ----------
    stiffness : sparse matrix, shape (N, N)
        Cotangent stiffness ``W`` (the LBO numerator).
    mass : sparse matrix, shape (N, N)
        Lumped mass matrix ``A`` (diagonal).
    potential : ndarray, shape (N,)
        Per-vertex potential ``V``.

    Returns
    -------
    sparse matrix, shape (N, N)
        ``H`` in CSC format, same sparsity pattern as ``W`` (the potential
        only adds to the diagonal, weighted by the mass).
    """
    n = stiffness.shape[0]
    v = np.asarray(potential, dtype=np.float64)
    if v.shape != (n,):
        raise ValueError(f"potential shape {v.shape} != ({n},).")
    mass_diag = np.asarray(mass.diagonal(), dtype=np.float64)
    h = sp.csc_matrix(stiffness, dtype=np.float64) + sp.diags(mass_diag * v, format="csc")
    return h


def hamiltonian_decompose(
    vertices: Vertices,
    faces: Faces,
    *,
    potential: ScalarMap | PotentialSpec = "casorati",
    k: int = 100,
    potential_scale: float = 1.0,
    laplacian_method: Literal["cotangent", "robust"] = "cotangent",
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Assemble and eigendecompose the Hamiltonian ``H = Delta + V``.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    potential : ndarray (N,) or str
        Either an explicit per-vertex potential or a :data:`PotentialSpec`
        built via :func:`build_potential`.
    k : int
        Number of eigenpairs.
    potential_scale : float
        Scale applied when ``potential`` is a built-in spec.
    laplacian_method : str
        ``"cotangent"`` (default) or ``"robust"`` (Sharp-Crane tufted
        Laplacian, requires ``robust_laplacian``).
    backend : str or backend object
        Compute backend (see :func:`spectralbrain.spectral.operators.resolve_backend`).

    Returns
    -------
    SpectralDecomposition
        With ``metadata["operator"] == "hamiltonian"``.
    """
    from spectralbrain.core.meshes import BrainMesh

    mesh = BrainMesh(vertices, faces)
    W, A = mesh.compute_laplacian(method=laplacian_method)

    if isinstance(potential, str):
        v = build_potential(vertices, faces, kind=potential, scale=potential_scale)
        pot_tag = potential
    else:
        v = np.asarray(potential, dtype=np.float64)
        pot_tag = "custom"

    H = hamiltonian_operator(W, A, v)
    sa = mesh.surface_area()

    sigma, clamp = _hamiltonian_shift(v)
    return operator_eigensystem(
        H,
        A,
        k=k,
        backend=backend,
        sigma=sigma,
        which="LM",
        clamp_nonneg=clamp,
        surface_area=sa,
        operator="hamiltonian",
        metadata={"potential": pot_tag, "potential_scale": potential_scale},
    )


def _hamiltonian_shift(v: np.ndarray) -> tuple[float, bool]:
    """Shift-invert target and clamping mode for ``H = W + A*diag(V)``.

    The Rayleigh quotient of ``H`` w.r.t. the lumped mass is bounded below
    by ``min(V)`` (``W`` is PSD).  For ``V >= 0`` the operator is PSD and the
    standard clamped path (sigma = -0.01) applies.  For a **signed** potential
    the spectrum can be negative: we must neither clamp it to 0 nor target
    sigma = -0.01 (which would return the eigenvalues nearest -0.01 and miss
    the lowest ones).  Placing sigma strictly below ``min(V)`` makes the
    ``k`` eigenvalues nearest sigma exactly the ``k`` smallest.
    """
    vmin = float(np.min(v)) if v.size else 0.0
    if vmin >= 0.0:
        return -0.01, True
    return vmin - 0.01 * (abs(vmin) + 1.0), False


# ======================================================================
# S3  DESCRIPTORS
# ======================================================================


def compute_siwks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 100,
    n_energies: int = 100,
    potential: ScalarMap | PotentialSpec = "casorati",
    potential_scale: float = 1.0,
    backend: BackendSpec | Any = "auto",
) -> DescriptorMatrix:
    """Scale-Invariant Wave Kernel Signature on the Hamiltonian spectrum.

    Computes the WKS from the Hamiltonian eigensystem and renders it
    scale-invariant by logarithmically rescaling the energy axis with the
    spectrum's own scale, following Fan et al. (2021) for grey-matter
    morphometry.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Number of eigenpairs.
    n_energies : int
        Number of WKS energy samples.
    potential : ndarray (N,) or str
        Hamiltonian potential.
    potential_scale : float
    backend : str or backend object

    Returns
    -------
    ndarray, shape (N, n_energies)

    References
    ----------
    Fan Y et al. *Medical Image Analysis* 73:102014, 2021.
    """
    from spectralbrain.spectral.lbo.descriptors import compute_wks

    decomp = hamiltonian_decompose(
        vertices,
        faces,
        potential=potential,
        k=k,
        potential_scale=potential_scale,
        backend=backend,
    )

    # Scale-invariance.  Under a uniform scaling x -> betax the eigenvalues
    # scale as beta^-2 and the M-orthonormal eigenfunctions as phi^2 propto 1/area.
    # (i) Normalise the eigenvalues by the spectrum's own scale (largest
    #     positive eigenvalue).  Note that WKS with *auto* energies is
    #     already invariant to a uniform eigenvalue rescale, so this step
    #     matters for the recorded ``siwks_scale`` and for any caller-fixed
    #     energy grid; it is kept for that reason.
    # (ii) Normalise the amplitude by the surface area (phi~ = sqrt area * phi) so
    #     the WKS values themselves -- not only their column-normalised
    #     profile -- are comparable across shapes of different size.
    evals = decomp.eigenvalues.copy()
    pos = evals[evals > _EPS]
    if pos.size:
        scale = float(pos.max())
        area = float(decomp.surface_area) if decomp.surface_area else 1.0
        decomp = SpectralDecomposition(
            eigenvalues=evals / scale,
            eigenvectors=decomp.eigenvectors * np.sqrt(max(area, _EPS)),
            stiffness=decomp.stiffness,
            mass=decomp.mass,
            surface_area=decomp.surface_area,
            metadata={**decomp.metadata, "siwks_scale": scale, "siwks_area": area},
        )
    return compute_wks(decomp, n_energies=n_energies)


def compute_localized_hks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 100,
    n_times: int = 100,
    potential: ScalarMap | PotentialSpec = "casorati",
    potential_scale: float = 1.0,
    backend: BackendSpec | Any = "auto",
) -> DescriptorMatrix:
    """Heat Kernel Signature from a potential-steered Hamiltonian basis.

    When ``potential`` encodes curvature, the resulting HKS is sensitive to
    extrinsic bending -- a property the intrinsic LBO-HKS lacks.

    Returns
    -------
    ndarray, shape (N, n_times)
    """
    from spectralbrain.spectral.lbo.descriptors import compute_hks

    decomp = hamiltonian_decompose(
        vertices,
        faces,
        potential=potential,
        k=k,
        potential_scale=potential_scale,
        backend=backend,
    )
    return compute_hks(decomp, n_times=n_times)


def compute_localized_wks(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 100,
    n_energies: int = 100,
    potential: ScalarMap | PotentialSpec = "casorati",
    potential_scale: float = 1.0,
    backend: BackendSpec | Any = "auto",
) -> DescriptorMatrix:
    """Wave Kernel Signature from a potential-steered Hamiltonian basis.

    Returns
    -------
    ndarray, shape (N, n_energies)
    """
    from spectralbrain.spectral.lbo.descriptors import compute_wks

    decomp = hamiltonian_decompose(
        vertices,
        faces,
        potential=potential,
        k=k,
        potential_scale=potential_scale,
        backend=backend,
    )
    return compute_wks(decomp, n_energies=n_energies)


def compute_compressed_modes(
    vertices: Vertices,
    faces: Faces,
    *,
    k: int = 50,
    mu: float = 1.0,
    n_iter: int = 8,
    backend: BackendSpec | Any = "auto",
) -> SpectralDecomposition:
    """Localised manifold harmonics via a self-consistent Hamiltonian.

    Approximates *compressed modes* (Ozolins et al. 2013; Neumann et al.
    2014) by iterating a potential that penalises spatial spread:
    starting from the LBO basis, the potential ``V propto 1 / (mu^-1 + |phi|)`` is
    rebuilt from the current eigenfunctions, sharpening their localisation.

    This is a lightweight, eigensolver-only surrogate for the full
    L^1-regularised optimisation, suitable as a localised descriptor basis.

    Parameters
    ----------
    vertices, faces : arrays
    k : int
        Number of localised modes.
    mu : float
        Localisation strength (larger -> more localised).
    n_iter : int
        Self-consistent iterations.
    backend : str or backend object

    Returns
    -------
    SpectralDecomposition
        With ``metadata["operator"] == "compressed_modes"``.

    References
    ----------
    Ozolins V, Lai R, Caflisch R, Osher S. "Compressed modes for
    variational problems in mathematics and physics." *PNAS* 110(46), 2013.
    Neumann T et al. "Compressed manifold modes for mesh processing."
    *Computer Graphics Forum* 33(5), 2014.
    """
    from spectralbrain.core.meshes import BrainMesh
    from spectralbrain.runtime import progress_simple

    mesh = BrainMesh(vertices, faces)
    W, A = mesh.compute_laplacian(method="cotangent")
    sa = mesh.surface_area()
    W.shape[0]

    # Start from the LBO basis (V = 0).
    decomp = operator_eigensystem(W, A, k=k, backend=backend, surface_area=sa, operator="lbo")

    # Spectral scale of the LBO: the potential must be commensurate with the
    # eigenvalues (units 1/length^2), otherwise it is either negligible or a
    # pure constant shift that leaves the eigenvectors unchanged.
    pos = decomp.eigenvalues[decomp.eigenvalues > _EPS]
    lam_scale = float(np.median(pos)) if pos.size else 1.0

    with progress_simple("Compressed modes", total=n_iter) as tick:
        for _ in range(n_iter):
            phi = decomp.eigenvectors  # (N, k)
            # Reweighting potential favouring spatial concentration.  The
            # eigenfunction magnitude (propto 1/sqrt area) is made dimensionless by
            # its mean so that mu has a mesh-independent meaning.
            spread = np.mean(np.abs(phi), axis=1)  # (N,)
            spread = spread / max(float(spread.mean()), _EPS)
            v = lam_scale * mu / (1.0 / max(mu, _EPS) + spread)  # (N,)
            H = hamiltonian_operator(W, A, v)
            decomp = operator_eigensystem(
                H,
                A,
                k=k,
                backend=backend,
                surface_area=sa,
                operator="compressed_modes",
                metadata={"mu": mu, "lambda_scale": lam_scale},
            )
            tick(1)

    return decomp


__all__ = [
    "PotentialSpec",
    "build_potential",
    "compute_compressed_modes",
    "compute_localized_hks",
    "compute_localized_wks",
    "compute_siwks",
    "hamiltonian_decompose",
    "hamiltonian_operator",
    "region_potential",
]
