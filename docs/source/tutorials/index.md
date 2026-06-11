# Tutorials

End-to-end, narrative walkthroughs on real data — the place to *learn by doing*.
Each notebook is rendered with its committed output (figures and all), so you can
read the whole story without running anything; clone the repo to execute them
yourself.

:::{admonition} How these are built
:class: note
The notebooks live in the repository's top-level `tutorials/` folder. The doc
build copies them into this section and renders them **without re-executing**
(`nb_execution_mode = "off"`), because several need real surfaces or a GPU. See
the {doc}`../about/contributing` page to rebuild them from their generators.
:::

```{toctree}
:maxdepth: 1

01_laplace_beltrami_operator
02_reading_real_brains_io
03_shapedna_global_fingerprint
04_heat_kernel_signature
05_wave_kernel_and_gps
06_pointclouds_and_tracts
07_functional_maps_and_distances
08_cohorts_and_vertexwise_stats
09_effectsizes_classification_harmonization
10_bayesian_and_visualization
```

## The arc

1. **Laplace–Beltrami operator** — the eigenproblem, hands-on.
2. **Reading real brains (I/O)** — FreeSurfer, GIfTI, NIfTI, HippUnfold, TractSeg.
3. **ShapeDNA** — the global fingerprint and shape distance.
4. **Heat Kernel Signature** — multiscale per-vertex geometry.
5. **Wave Kernel & GPS** — band-pass signatures and spectral embedding.
6. **Point clouds & tracts** — volumetric segmentations and white-matter bundles.
7. **Functional maps & distances** — cross-shape correspondence and metrics.
8. **Cohorts & vertex-wise stats** — group loading, FWE permutation, FDR, TFCE.
9. **Effect sizes, classification & harmonization** — ComBat/ComBat-GAM, AUC.
10. **Bayesian & visualization** — PyMC models and publication figures.
