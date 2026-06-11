# Foundations: the Laplace–Beltrami operator

Everything in SpectralBrain is built on one object: the spectral decomposition of
the **Laplace–Beltrami operator** (LBO) of a surface or point cloud. Understand
this page and the rest of the library is just a matter of which descriptor you
read off the same basis.

## Intuition

The LBO is the curved-surface generalization of the familiar Laplacian
$\Delta = \partial_x^2 + \partial_y^2 + \partial_z^2$. On a mesh it measures how
much a function at a vertex differs from the average of its neighbors, weighted
by the local geometry. Its eigenfunctions are the natural "vibration modes" of
the shape — the harmonics of a drum, but for a hippocampus or a cortical surface.

Crucially, the LBO is **intrinsic**: it depends only on the surface metric, not
on how the shape is positioned, rotated, or scaled in space. That is exactly the
property that makes spectral descriptors robust to registration and pose, where
volume and thickness are not.

## The eigenproblem

We solve the generalized eigenvalue problem

$$
\mathbf{W}\,\boldsymbol{\phi}_i = \lambda_i \,\mathbf{A}\,\boldsymbol{\phi}_i ,
\qquad 0 = \lambda_0 \le \lambda_1 \le \lambda_2 \le \dots
$$

where $\mathbf{W}$ is the cotangent stiffness matrix, $\mathbf{A}$ the
mass (area) matrix, $\lambda_i$ the eigenvalues, and $\boldsymbol{\phi}_i$ the
eigenfunctions. The eigenvalues $\{\lambda_i\}$ are the **spectrum**; the
eigenfunctions $\{\boldsymbol{\phi}_i\}$ form an orthonormal basis for functions
on the surface. Low $\lambda$ describes coarse, global shape; high $\lambda$
describes fine, local detail.

## In SpectralBrain

You compute the decomposition once and reuse it for every descriptor:

```python
import spectralbrain as sb

vertices, faces = sb.load_freesurfer_surface("lh.pial")
mesh = sb.BrainMesh(vertices, faces)

decomp = mesh.decompose(k=300)   # -> SpectralDecomposition

decomp.eigenvalues      # (k,)        the spectrum  {λ_i}
decomp.eigenvectors     # (n_vert, k) the basis     {φ_i}
```

Choosing `k` (the number of eigenpairs) trades detail for cost: a few hundred
modes is plenty for whole-structure descriptors like ShapeDNA, while per-vertex
signatures benefit from more.

## Where to next

- {doc}`global_embeddings` — read the spectrum directly (ShapeDNA) or embed
  points with it (GPS).
- {doc}`heat_descriptors` and {doc}`wave_descriptors` — multiscale per-vertex
  signatures built from $\{\lambda_i, \boldsymbol{\phi}_i\}$.

:::{seealso}
Tutorial `01_laplace_beltrami_operator` walks through the eigenproblem on a real
surface step by step. API: {func}`~spectralbrain.BrainMesh`,
{class}`~spectralbrain.SpectralDecomposition`.
:::
