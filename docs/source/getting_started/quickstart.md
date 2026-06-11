# Quickstart

## Your first analysis

Every workflow follows the same three beats: **load → decompose → describe**
(then analyze or render).

```python
import spectralbrain as sb

# 1. Load — any supported format; the loader auto-detects it.
vertices, faces = sb.load_freesurfer_surface("lh.pial")

# 2. Decompose — solve the Laplace–Beltrami eigenproblem once.
mesh = sb.BrainMesh(vertices, faces)
decomp = mesh.decompose(k=300)            # -> SpectralDecomposition

# 3. Describe — every descriptor reads the same decomposition.
shapedna = sb.compute_shapedna(decomp)                 # global fingerprint
hks = sb.compute_hks(decomp, t_values=[1, 10, 100])    # multiscale, per-vertex
wks = sb.compute_wks(decomp, n_energies=100)           # band-pass, per-vertex

# 4. Render.
sb.plot_brain(data=hks[:, 0], atlas="schaefer_400")
```

The single `SpectralDecomposition` is the hub: compute it once, then derive as
many descriptors as you like from it without re-solving the eigenproblem.

## Pick your path

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb;1.2em` New to spectral geometry?
Start with the concepts. The {doc}`Learn section <../learn/index>` explains the
Laplace–Beltrami operator and what each descriptor *measures*, before you wire
it into a study.

+++
{doc}`Go to Learn → <../learn/index>`
:::

:::{grid-item-card} {octicon}`zap;1.2em` Already know the methods?
Jump to worked, end-to-end examples on real data, or straight to the function
signatures.

+++
{doc}`Tutorials → <../tutorials/index>` · {doc}`API → <../api/index>`
:::

::::
