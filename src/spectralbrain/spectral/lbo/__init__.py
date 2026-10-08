"""Laplace-Beltrami spectral analysis -- the trunk of SpectralBrain.

Everything here consumes a :class:`~spectralbrain.core.base.SpectralDecomposition`
of the cotangent LBO (``BrainMesh.decompose()``):

- :mod:`.descriptors` -- ShapeDNA, HKS, SI-HKS, WKS, GPS, Bates, BKS, IBKS,
  SI-WKS, heat-flow entropy and M-GP landmarks.
- :mod:`.distances` -- WESD, ShapeDNA / biharmonic / commute-time / diffusion
  distances and geometric connectomes.
- :mod:`.wavelets` -- spectral graph wavelets.
- :mod:`.collections` -- pairs and cohorts of shapes: functional maps, shape
  difference operators, DWKS and correspondence-free spectral deformation.
- :mod:`.anisotropic` -- anisotropic HKS / WKS / ASMWD.
- :mod:`.modes` -- geometric eigenmodes on the render mesh and their
  wavelengths.
"""

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
