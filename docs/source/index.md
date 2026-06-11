---
sd_hide_title: true
---

# SpectralBrain

```{div} sd-text-center sd-fs-1 sd-font-weight-bold sd-text-primary
SpectralBrain
```

```{div} sd-text-center sd-fs-4 sd-text-muted
Spectral shape analysis for brain structures
```

```{div} sd-text-center sd-mb-4
Compute, analyze, and visualize intrinsic spectral shape descriptors of
cortical surfaces, subcortical and hippocampal meshes, white-matter tracts, and
point clouds — from the **Laplace–Beltrami operator** straight through to
rigorous statistics and publication-ready figures.
```

```{code-block} bash
:caption: Install
pip install spectralbrain
```

A five-line taste — the Heat Kernel Signature of a surface:

```python
import spectralbrain as sb

vertices, faces = sb.load_freesurfer_surface("lh.pial")
mesh = sb.BrainMesh(vertices, faces)
decomp = mesh.decompose(k=300)                 # Laplace–Beltrami eigenpairs
hks = sb.compute_hks(decomp, t_values=[1, 10, 100])
sb.plot_brain(data=hks[:, 0], atlas="schaefer_400")
```

---

## Where to go next

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Getting Started
:link: getting_started/index
:link-type: doc

Install, run your first analysis, and find the learning path that matches what
you already know.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Learn
:link: learn/index
:link-type: doc

The *why* behind the methods — one page per descriptor family, the distances,
and normative modeling.
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc

Ten end-to-end notebooks, from the Laplace–Beltrami operator to Bayesian models
on real cohorts.
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Example Gallery
:link: auto_examples/index
:link-type: doc

Short, runnable recipes — each one is a figure you can reproduce in seconds.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-to Guides
:link: howto/index
:link-type: doc

Task-focused recipes: the GPU backend, multi-site harmonization, vertex-wise
statistics, the MTLE-HS hippocampus workflow.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Reference
:link: api/index
:link-type: doc

Every public class and function, generated from the source docstrings.
:::

::::

---

## Why spectral shape?

Volume and thickness collapse a structure's shape to a few scalars and are
sensitive to registration and voxel size. Intrinsic spectral descriptors derived
from the Laplace–Beltrami operator characterize shape *independently of pose and
parameterization*, capturing geometry that volume alone misses. SpectralBrain
packages those descriptors together with the I/O, multi-site harmonization,
correct multiple-comparison statistics, and rendering that a neuroimaging study
needs end to end — with a primary focus on the hippocampus in mesial temporal
lobe epilepsy, while staying general to any brain surface or point cloud.

```{toctree}
:hidden:
:caption: Get started

getting_started/index
```

```{toctree}
:hidden:
:caption: Documentation

learn/index
tutorials/index
auto_examples/index
howto/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
about/citing
about/contributing
about/changelog
```
