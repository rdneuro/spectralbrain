"""Non-Laplace–Beltrami spectral operators and descriptors.

:mod:`spectralbrain.expanded` extends SpectralBrain beyond the
Laplace–Beltrami operator (LBO) with a family of *complementary* spectral
operators.  The LBO trunk is never replaced: every operator here plugs
into the **same** assemble→eigensolve→descriptor pipeline and produces the
**same** :class:`~spectralbrain.core.base.SpectralDecomposition`, so all
existing descriptors (HKS, WKS, GPS, ShapeDNA, SGW, …) apply unchanged.

Motivation
----------
The LBO is intrinsic and isometry-invariant — blind, by construction, to
*extrinsic* bending such as cortical gyrification.  The operators in this
subpackage fill that blind spot (Hamiltonian, Dirac, Steklov, shape
operator), add topology- and graph-spectral views (Hodge, persistent,
Ricci, NetLSD/FGSD), and provide classic morphometric spectra (EFD,
eigenshape, SPHARM).

Submodules
----------
operators
    Hamiltonian / Schrödinger ``H = Δ + V`` (curvature- or region-steered
    bases, SIWKS, localised and compressed modes).  *Available.*
biharmonic
    Biharmonic / polyharmonic spectra and multiscale biharmonic kernels
    via LBO eigenpair reuse.  *Available.*
extrinsic
    Dirac operator (+ Dirac Kernel Signature), Steklov / Dirichlet-to-
    Neumann, and shape-operator descriptors.  *Available.*
anisotropic
    Finsler-LBO and curvature-aligned anisotropy banks (extends
    :mod:`spectralbrain.spectral.anisotropic`).  *Available.*
topologic
    Hodge ``L_k`` / p-form Laplacians, Ollivier/Forman Ricci, Reeb–Morse,
    connection / magnetic / sheaf Laplacians.  *Available.*
persistent
    Persistent homology, persistent Laplacian, combinatorial/persistent
    Hodge–Dirac.  *Available.*
graphs
    Graph-Laplacian spectra, NetLSD, FGSD, effective-resistance / Estrada,
    Fiedler / normalised-cut features.  *Available.*
misc
    Elliptic Fourier descriptors, eigenshape, principal warps, SPHARM.
    *Available.*

Optional dependencies
---------------------
This subpackage uses optional, lazily-imported third-party libraries
(``gudhi``, ``ripser``, ``persim``, ``GraphRicciCurvature``, ``pyefd``,
``robust_laplacian``, ``potpourri3d``, ``pot``, ``networkx``).  None are
imported at ``import spectralbrain`` time.  Install the full set with::

    pip install 'spectralbrain[expanded]'

Importing this subpackage is cheap: submodules and symbols are resolved
lazily on first access via the module ``__getattr__`` hook.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# ── available submodules (shipped) ────────────────────────────────────
_SUBMODULES: dict[str, str] = {
    "operators": ".operators",
    "biharmonic": ".biharmonic",
    "extrinsic": ".extrinsic",
    "anisotropic": ".anisotropic",
    "topologic": ".topologic",
    "persistent": ".persistent",
    "graphs": ".graphs",
    "misc": ".misc",
}

# ── submodules scheduled for later deliveries ─────────────────────────
# (the expanded subpackage is now feature-complete)
_FORTHCOMING: dict[str, str] = {}

# ── public symbols → owning submodule (lazy) ──────────────────────────
_SYMBOL_MODULE: dict[str, str] = {
    # _base (foundation)
    "resolve_backend": "._base",
    "operator_eigensystem": "._base",
    "operator_signature": "._base",
    "solve_eigsh": "._base",
    "require_optional": "._base",
    "principal_curvatures": "._base",
    "mean_curvature": "._base",
    "gaussian_curvature": "._base",
    "casorati_curvature": "._base",
    "shape_index": "._base",
    "vertex_normals": "._base",
    "face_areas": "._base",
    # operators
    "hamiltonian_operator": ".operators",
    "hamiltonian_decompose": ".operators",
    "build_potential": ".operators",
    "region_potential": ".operators",
    "compute_siwks": ".operators",
    "compute_localized_hks": ".operators",
    "compute_localized_wks": ".operators",
    "compute_compressed_modes": ".operators",
    # biharmonic
    "polyharmonic_decompose": ".biharmonic",
    "biharmonic_kernel_signature": ".biharmonic",
    "multiscale_biharmonic_kernel": ".biharmonic",
    "compute_bhks": ".biharmonic",
    # extrinsic
    "shape_operator_descriptor": ".extrinsic",
    "dirac_operator": ".extrinsic",
    "dirac_decompose": ".extrinsic",
    "compute_dks": ".extrinsic",
    "steklov_operator": ".extrinsic",
    "steklov_spectrum": ".extrinsic",
    "compute_steklov_wks": ".extrinsic",
    # anisotropic
    "finsler_laplacian": ".anisotropic",
    "finsler_decompose": ".anisotropic",
    "anisotropic_bank_descriptor": ".anisotropic",
    # topologic
    "build_incidence": ".topologic",
    "hodge_laplacian": ".topologic",
    "hodge_decompose": ".topologic",
    "betti_numbers": ".topologic",
    "forman_ricci_curvature": ".topologic",
    "ollivier_ricci_curvature": ".topologic",
    "ricci_curvature_mesh": ".topologic",
    "magnetic_laplacian": ".topologic",
    "magnetic_decompose": ".topologic",
    "connection_laplacian": ".topologic",
    "connection_decompose": ".topologic",
    "sheaf_laplacian": ".topologic",
    "reeb_graph": ".topologic",
    "reeb_graph_features": ".topologic",
    # persistent
    "graph_persistence_h0": ".persistent",
    "vietoris_rips_diagram": ".persistent",
    "betti_curve": ".persistent",
    "persistence_landscape": ".persistent",
    "persistence_statistics": ".persistent",
    "persistence_image": ".persistent",
    "persistent_laplacian": ".persistent",
    "persistent_laplacian_spectrum": ".persistent",
    "combinatorial_dirac": ".persistent",
    "combinatorial_dirac_spectrum": ".persistent",
    # graphs
    "graph_laplacian_decompose": ".graphs",
    "netlsd": ".graphs",
    "fgsd": ".graphs",
    "effective_resistance": ".graphs",
    "kirchhoff_index": ".graphs",
    "estrada_index": ".graphs",
    "fiedler_value": ".graphs",
    "fiedler_vector": ".graphs",
    "spectral_features": ".graphs",
    # misc
    "elliptic_fourier_descriptors": ".misc",
    "eigenshape_analysis": ".misc",
    "principal_warps": ".misc",
    "spharm_coefficients": ".misc",
    "spharm_power_spectrum": ".misc",
    "spharm_surface": ".misc",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve submodules and public symbols on first access."""
    if name in _SUBMODULES:
        mod = importlib.import_module(_SUBMODULES[name], __name__)
        globals()[name] = mod
        return mod
    if name in _SYMBOL_MODULE:
        mod = importlib.import_module(_SYMBOL_MODULE[name], __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    if name in _FORTHCOMING:
        raise AttributeError(
            f"spectralbrain.expanded.{name} ({_FORTHCOMING[name]}) is not "
            f"yet available in this build — it ships in a later delivery."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy names to tab-completion and dir()."""
    return sorted(set(__all__) | set(_FORTHCOMING))


__all__ = [
    # submodules
    "anisotropic",
    "biharmonic",
    "extrinsic",
    "graphs",
    "misc",
    "operators",
    "persistent",
    "topologic",
    # _base
    "casorati_curvature",
    "face_areas",
    "gaussian_curvature",
    "mean_curvature",
    "operator_eigensystem",
    "operator_signature",
    "principal_curvatures",
    "require_optional",
    "resolve_backend",
    "shape_index",
    "solve_eigsh",
    "vertex_normals",
    # operators
    "build_potential",
    "compute_compressed_modes",
    "compute_localized_hks",
    "compute_localized_wks",
    "compute_siwks",
    "hamiltonian_decompose",
    "hamiltonian_operator",
    "region_potential",
    # biharmonic
    "biharmonic_kernel_signature",
    "compute_bhks",
    "multiscale_biharmonic_kernel",
    "polyharmonic_decompose",
    # extrinsic
    "compute_dks",
    "compute_steklov_wks",
    "dirac_decompose",
    "dirac_operator",
    "shape_operator_descriptor",
    "steklov_operator",
    "steklov_spectrum",
    # anisotropic
    "anisotropic_bank_descriptor",
    "finsler_decompose",
    "finsler_laplacian",
    # topologic
    "betti_numbers",
    "build_incidence",
    "connection_decompose",
    "connection_laplacian",
    "forman_ricci_curvature",
    "hodge_decompose",
    "hodge_laplacian",
    "magnetic_decompose",
    "magnetic_laplacian",
    "ollivier_ricci_curvature",
    "reeb_graph",
    "reeb_graph_features",
    "ricci_curvature_mesh",
    "sheaf_laplacian",
    # persistent
    "betti_curve",
    "combinatorial_dirac",
    "combinatorial_dirac_spectrum",
    "graph_persistence_h0",
    "persistence_image",
    "persistence_landscape",
    "persistence_statistics",
    "persistent_laplacian",
    "persistent_laplacian_spectrum",
    "vietoris_rips_diagram",
    # graphs
    "effective_resistance",
    "estrada_index",
    "fgsd",
    "fiedler_value",
    "fiedler_vector",
    "graph_laplacian_decompose",
    "kirchhoff_index",
    "netlsd",
    "spectral_features",
    # misc
    "eigenshape_analysis",
    "elliptic_fourier_descriptors",
    "principal_warps",
    "spharm_coefficients",
    "spharm_power_spectrum",
    "spharm_surface",
]


if TYPE_CHECKING:  # static type-checkers see the real modules
    from spectralbrain.expanded import (  # noqa: F401
        anisotropic,
        biharmonic,
        extrinsic,
        graphs,
        misc,
        operators,
        persistent,
        topologic,
    )
    from spectralbrain.expanded._base import (  # noqa: F401
        casorati_curvature,
        face_areas,
        gaussian_curvature,
        mean_curvature,
        operator_eigensystem,
        operator_signature,
        principal_curvatures,
        require_optional,
        resolve_backend,
        shape_index,
        solve_eigsh,
        vertex_normals,
    )
