# Wave / band-pass descriptors

Where the heat signature *averages* over scales (a low-pass view), the **Wave
Kernel Signature** localizes in frequency — a band-pass view of the same
spectral basis. The two are complementary: HKS is better at coarse localization,
WKS at separating features that live at similar scales but different
"frequencies".

## WKS — the Wave Kernel Signature

The WKS evaluates the probability of a quantum particle, with a log-energy
distribution centered at $e$, being measured at point $x$:

$$
\text{WKS}(x, e) = C_e \sum_i \phi_i(x)^2 \,
\exp\!\left( -\frac{(e - \log \lambda_i)^2}{2\sigma^2} \right).
$$

By scanning the energy $e$ across the spectrum, each vertex gets a signature that
responds to a narrow band of eigenfrequencies at a time. This frequency
selectivity makes the WKS sharper than the HKS for fine feature matching.

```python
wks = sb.compute_wks(decomp, n_energies=100)   # (n_vert, n_energies)
```

## Anisotropic WKS

As with the heat family, an **anisotropic** WKS replaces the isotropic LBO with a
curvature-aligned {func}`~spectralbrain.anisotropic_laplacian`, so the band-pass
response becomes direction-sensitive.

```python
from spectralbrain.spectral.anisotropic import compute_anisotropic_wks
a_wks = compute_anisotropic_wks(mesh, ...)
```

:::{tip}
HKS vs. WKS in one line: **HKS = how much heat stays** (multiscale, low-pass),
**WKS = which frequencies live here** (band-pass). Many studies stack both.
:::

:::{seealso}
Tutorial `05_wave_kernel_and_gps`. API: {func}`~spectralbrain.compute_wks`.
A time-derivative, collection-aware variant (DWKS) is covered in
{doc}`functional_maps_and_distances`.
:::
