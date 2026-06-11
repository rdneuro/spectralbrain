# Spectral wavelets

Kernel signatures (HKS, WKS, BKS) summarize the spectrum with a single weighting
per scale. **Spectral wavelets** instead build a *bank* of band-pass filters,
giving an explicit multi-resolution decomposition of functions on the surface —
the spectral-geometry analogue of a classical wavelet transform.

## SGW — Spectral Graph Wavelets

SGW applies a filter kernel $g(\cdot)$ dilated by a set of scales $\{t_j\}$ to
the spectrum, then maps each filtered response back onto the surface. With
$\hat g$ a band-pass generator (mexican-hat, heat, or Meyer kernels are
provided), the wavelet at scale $t_j$ and vertex $x$ is

$$
\psi_{t_j}(x) = \sum_i g(t_j \lambda_i)\, \phi_i(x)\, \phi_i(\cdot).
$$

For efficiency the transform is computed with a **Chebyshev polynomial
approximation** of the kernel, avoiding a full eigendecomposition when only the
wavelet coefficients are needed.

```python
coeffs = sb.sgw_transform(decomp, scales=[...], kernel="mexican_hat")
desc = sb.sgw_descriptor(decomp, ...)   # per-vertex multi-scale descriptor
```

Available kernels: `mexican_hat`, `heat`, and `meyer`.

## ASMWD — Anisotropic Spectral Mesh Wavelet Descriptor

ASMWD combines the wavelet idea with the anisotropic (curvature-aligned)
Laplacian, producing a multi-resolution descriptor that is also
direction-sensitive — useful where both *scale* and *orientation* of a feature
matter.

```python
from spectralbrain.spectral.anisotropic import compute_asmwd
asmwd = compute_asmwd(mesh, ...)
```

:::{seealso}
API: {func}`~spectralbrain.sgw_transform`,
{func}`~spectralbrain.sgw_descriptor`. ASMWD lives in
`spectralbrain.spectral.anisotropic` alongside the anisotropic HKS/WKS.
:::
