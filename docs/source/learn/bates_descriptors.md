# Bates–Kornfeld signatures

This family generalizes the kernel-signature idea (HKS/WKS) with a different
spectral weighting, giving descriptors that emphasize a complementary part of the
spectrum.

## Bates signatures and BKS

The **Bates–Kornfeld Signature** (BKS) follows the same per-vertex template as
the heat and wave signatures —

$$
S(x) = \sum_i g(\lambda_i)\, \phi_i(x)^2
$$

— but with a kernel $g(\lambda)$ chosen to weight the spectrum differently from
the exponential (heat) or Gaussian-in-log-energy (wave) kernels. The
`compute_bates_signatures` entry point produces the family; `compute_bks` is the
signature itself.

```python
bates = sb.compute_bates_signatures(decomp, ...)
bks = sb.compute_bks(decomp, ...)
```

## IBKS — the inverse signature

The **inverse BKS** reweights toward the opposite end of the spectrum, trading
emphasis between coarse and fine geometry. Using BKS and IBKS together brackets
the spectral range a single kernel would otherwise sample only partially.

```python
ibks = sb.compute_ibks(decomp, ...)
```

:::{seealso}
API: {func}`~spectralbrain.compute_bates_signatures`,
{func}`~spectralbrain.compute_bks`, {func}`~spectralbrain.compute_ibks`.
Compare against the {doc}`heat <heat_descriptors>` and
{doc}`wave <wave_descriptors>` kernels to see how the spectral weighting differs.
:::
