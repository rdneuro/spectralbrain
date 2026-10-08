"""SpectralBrain spectral analysis.

- :mod:`spectralbrain.spectral.lbo` -- the Laplace-Beltrami trunk (descriptors,
  distances, wavelets, collections, anisotropic variants, eigenmodes). Its
  public API is re-exported here, so ``from spectralbrain.spectral import
  compute_hks`` keeps working.
- :mod:`spectralbrain.spectral.operators` -- every other spectral operator
  (Dirac, Steklov, Hamiltonian, Hodge, magnetic, sheaf, Finsler, biharmonic,
  persistent Laplacian, graph spectra, classical morphometric spectra). It
  produces the same ``SpectralDecomposition``; submodules load lazily.
"""

from spectralbrain.spectral import lbo, operators  # noqa: F401
from spectralbrain.spectral.lbo.anisotropic import (  # noqa: F401
    anisotropic_laplacian,
    compute_anisotropic_hks,
    compute_anisotropic_wks,
    compute_asmwd,
)
from spectralbrain.spectral.lbo.collections import (  # noqa: F401
    compute_dwks,
    compute_dwks_collection,
    compute_functional_map,
    lateralization_map,
    shape_difference_operator,
    spectral_deformation,
)
from spectralbrain.spectral.lbo.descriptors import (  # noqa: F401
    MGPLandmarkResult,
    compute_all_descriptors,
    compute_bates_signatures,
    compute_bks,
    compute_gps,
    compute_hks,
    compute_ibks,
    compute_shapedna,
    compute_si_hks,
    compute_si_wks,
    compute_wks,
    heat_flow_entropy,
    mgp_landmarks,
    shared_hks_times,
)
from spectralbrain.spectral.lbo.distances import (  # noqa: F401
    aggregate_to_networks,
    biharmonic_distance,
    build_geometric_connectome,
    commute_time_distance,
    descriptor_distance,
    diffusion_distance,
    diffusion_distance_multiscale,
    shapedna_distance,
    wesd,
    wesd_matrix,
)
from spectralbrain.spectral.lbo.modes import (  # noqa: F401
    compute_geometric_eigenmodes,
    eigenmode_wavelength,
)
from spectralbrain.spectral.lbo.wavelets import (  # noqa: F401
    heat_kernel,
    mexican_hat_kernel,
    meyer_kernel,
    sgw_descriptor,
    sgw_transform,
)
