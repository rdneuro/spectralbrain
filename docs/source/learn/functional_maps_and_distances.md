# Functional maps & distances

The descriptors so far live on a *single* shape. This page covers the tools that
relate *two or more* shapes: functional maps and the shape difference operator
for correspondence, the collection-aware DWKS, and the spectral distance metrics
that turn shape comparison into a number.

## Functional maps and the shape difference operator

A **functional map** represents a correspondence between two surfaces not as a
vertex-to-vertex matching but as a small matrix $\mathbf{C}$ that translates
functions from one shape's spectral basis to the other's. It is compact, robust,
and the natural language for comparing structures of slightly different
tessellation.

The **shape difference operator** (SDO) then localizes *where* and *how* two
shapes differ, given their functional map — separating area-based from
conformal (angle-based) change.

```python
C = sb.compute_functional_map(decomp_a, decomp_b, descriptors_a, descriptors_b)
sdo = sb.shape_difference_operator(decomp_a, decomp_b, C)
```

## DWKS — collection-aware wave signature

The **Derivative WKS** extends the wave signature across a *collection* of shapes
related by functional maps, exposing how band-pass geometry varies through the
collection rather than within a single surface.

```python
from spectralbrain.spectral.collections import compute_dwks, compute_dwks_collection
```

## Spectral distances

Once shapes are described, these metrics quantify how far apart they are:

| Function | Distance |
| --- | --- |
| {func}`~spectralbrain.wesd` / {func}`~spectralbrain.wesd_matrix` | Weighted Spectral Distance — eigenvalue-based, whole-shape |
| {func}`~spectralbrain.shapedna_distance` | distance between ShapeDNA fingerprints |
| {func}`~spectralbrain.biharmonic_distance` | smooth, globally-aware point-to-point distance |
| {func}`~spectralbrain.commute_time_distance` | expected random-walk commute time between points |
| {func}`~spectralbrain.diffusion_distance` | heat-diffusion distance (with a multiscale variant) |
| {func}`~spectralbrain.descriptor_distance` | generic distance between two descriptor fields |

From point-to-point spectral distances you can also assemble a
**geometric connectome** —
{func}`~spectralbrain.build_geometric_connectome` — and aggregate it to networks
for graph-level analysis.

```python
W = sb.wesd_matrix(decompositions)            # pairwise whole-shape distances
conn = sb.build_geometric_connectome(decomp)  # intrinsic connectome
```

:::{seealso}
Tutorial `07_functional_maps_and_distances`. Point-cloud distances
(`chamfer_distance`, `hausdorff_distance`, `procrustes_align`) are documented in
the {doc}`API <../api/index>` under the core module.
:::
