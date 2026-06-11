# Heat-diffusion descriptors

This family describes local geometry through **heat flow**: drop a unit of heat
at a point and watch how much stays put over time. Slow time scales see broad,
global shape; fast time scales see fine, local detail.

## HKS — the Heat Kernel Signature

The HKS records the amount of heat remaining at a point $x$ after time $t$,
summed over the spectrum:

$$
\text{HKS}(x, t) = \sum_{i} e^{-\lambda_i t}\, \phi_i(x)^2 .
$$

Sampling a range of $t$ gives each vertex a multiscale signature: small $t$
emphasizes curvature and ridges, large $t$ emphasizes global position within the
structure. The HKS is intrinsic, isometry-invariant, and stable under noise.

```python
hks = sb.compute_hks(decomp, t_values=[1, 10, 100])   # (n_vert, n_t)
```

## SI-HKS — Scale-Invariant HKS

The plain HKS is invariant to isometry but *not* to scale: a larger copy of the
same shape diffuses heat differently. SI-HKS removes the global scale factor
(via a logarithmic time sampling and a Fourier transform of the signature), so
two structures that differ only in size match.

```python
si_hks = sb.compute_si_hks(decomp)
```

## Anisotropic HKS

Standard heat flow is isotropic — it spreads equally in all directions. The
**anisotropic** variant biases diffusion along principal curvature directions,
sharpening sensitivity to directional features such as sulcal banks or the
long axis of the hippocampus. It is built on an
{func}`~spectralbrain.anisotropic_laplacian` rather than the isotropic LBO.

```python
from spectralbrain.spectral.anisotropic import compute_anisotropic_hks
a_hks = compute_anisotropic_hks(mesh, ...)
```

:::{seealso}
Tutorial `04_heat_kernel_signature`. API:
{func}`~spectralbrain.compute_hks`, {func}`~spectralbrain.compute_si_hks`,
{func}`~spectralbrain.anisotropic_laplacian`.
:::
