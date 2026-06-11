# Learn

The methods behind SpectralBrain — the *why*, not just the call signature. Each
page introduces a family of descriptors, the intuition for what it measures, the
mathematics in brief, and the functions that implement it. Reference-level
detail lives in the {doc}`API <../api/index>`; full derivations are reserved for
the forthcoming book.

```{toctree}
:maxdepth: 1

laplace_beltrami
spectral_descriptors
global_embeddings
heat_descriptors
wave_descriptors
bates_descriptors
wavelet_descriptors
functional_maps_and_distances
normative_modeling
```

## The descriptor families at a glance

| Family | Descriptors | Captures |
| --- | --- | --- |
| {doc}`Foundations <laplace_beltrami>` | LBO eigenpairs (`SpectralDecomposition`) | the spectral basis everything else is built on |
| {doc}`Descriptors in general <spectral_descriptors>` | the spectral-filter framework | how a spectrum becomes a shape measurement |
| {doc}`Global & embeddings <global_embeddings>` | ShapeDNA, GPS | whole-shape fingerprint; intrinsic point embedding |
| {doc}`Heat diffusion <heat_descriptors>` | HKS, SI-HKS, anisotropic HKS | multiscale local geometry via heat flow |
| {doc}`Wave / band-pass <wave_descriptors>` | WKS, anisotropic WKS | frequency-localized geometry |
| {doc}`Bates–Kornfeld <bates_descriptors>` | Bates signatures, BKS, IBKS | spectral-kernel signatures and their inverse |
| {doc}`Spectral wavelets <wavelet_descriptors>` | SGW, ASMWD | multi-resolution, optionally anisotropic |
| {doc}`Functional maps & distances <functional_maps_and_distances>` | functional maps, SDO, DWKS, WESD & co. | cross-shape correspondence and shape distance |

Every descriptor in the table reads the same `SpectralDecomposition`, so the
{doc}`Foundations page <laplace_beltrami>` is the natural place to begin.
