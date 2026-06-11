# Global fingerprints & embeddings

Two descriptors read the spectral basis at the *whole-shape* level rather than
per vertex: **ShapeDNA**, which is the spectrum itself, and **GPS**, which embeds
every point into a spectral coordinate system.

## ShapeDNA — the shape's fingerprint

ShapeDNA is simply the (normalized) sequence of LBO eigenvalues:

$$
\text{ShapeDNA} = (\lambda_1, \lambda_2, \dots, \lambda_k).
$$

Because the spectrum is intrinsic and isometry-invariant, two surfaces with the
same ShapeDNA have, to that truncation, the same intrinsic geometry. It is a
compact, fixed-length fingerprint — ideal for comparing whole structures (left
vs. right hippocampus, patient vs. normative reference) with a single distance.

```python
sdna = sb.compute_shapedna(decomp)              # (k,) eigenvalue fingerprint
d = sb.shapedna_distance(sdna_a, sdna_b)        # compare two shapes
```

## GPS — the Global Point Signature

GPS turns each vertex into a point in an infinite-dimensional space whose
coordinates are the eigenfunctions scaled by the inverse square-root of the
eigenvalues:

$$
\text{GPS}(x) = \left( \frac{\phi_1(x)}{\sqrt{\lambda_1}},
\frac{\phi_2(x)}{\sqrt{\lambda_2}}, \dots \right).
$$

In this embedding, intrinsic (geodesic-like) proximity on the surface becomes
ordinary Euclidean proximity. It is the bridge from "shape" to "feature vector"
for clustering or correspondence.

```python
gps = sb.compute_gps(decomp, n_components=50)   # (n_vert, n_components)
```

:::{seealso}
Tutorial `03_shapedna_global_fingerprint` (ShapeDNA) and
`05_wave_kernel_and_gps` (GPS). API: {func}`~spectralbrain.compute_shapedna`,
{func}`~spectralbrain.compute_gps`, {func}`~spectralbrain.shapedna_distance`.
:::
