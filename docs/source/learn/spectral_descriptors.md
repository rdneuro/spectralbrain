# Spectral descriptors, in general

The {doc}`Foundations page <laplace_beltrami>` ended with a single object: the
spectral decomposition $\{\lambda_i, \boldsymbol{\phi}_i\}$ of a shape's
Laplace–Beltrami operator. A **spectral descriptor** is anything you read off
that decomposition to characterize the shape. Every descriptor in SpectralBrain
— ShapeDNA, HKS, WKS, BKS, the wavelets, GPS — is a particular way of reading the
same spectrum. This page is about what that reading *is*, where it comes from,
what it actually means geometrically, and how a list of eigenvalues and
eigenfunctions ends up encoding morphometry.

## What a descriptor is

A descriptor turns the spectral basis into a number, a vector, or a field that
you can compare, cluster, or correlate with a clinical variable. They come in a
few flavors:

- **Global descriptors** summarize the *whole* shape as a single vector — the
  spectrum itself (ShapeDNA). One vector per surface.
- **Point signatures** assign a feature vector to *every vertex* — a field over
  the surface (HKS, WKS, BKS/IBKS, the wavelet descriptors).
- **Embeddings** re-coordinate the surface into a space where intrinsic geometry
  becomes ordinary geometry (GPS).
- **Cross-shape operators** relate *two or more* shapes (functional maps, the
  shape difference operator, the spectral distances).

Different as they look, they are all instances of one idea: applying a function
to the Laplace–Beltrami operator and looking at the result.

## The common recipe: filtering the spectrum

The Laplace–Beltrami operator $\Delta$ is self-adjoint, so the spectral theorem
lets us apply any scalar function $g$ to it through its eigenpairs:

$$
g(\Delta) \;=\; \sum_i g(\lambda_i)\, \boldsymbol{\phi}_i \boldsymbol{\phi}_i^{\top}.
$$

This **functional calculus** is the engine behind every descriptor. The choice of
$g$ is the only thing that changes from one descriptor family to the next — it is
a *spectral filter* that decides which frequencies of the shape you listen to.

### The prototype: the heat kernel

The cleanest derivation starts with the heat equation on the surface. Let heat
diffuse from an initial distribution $u_0$:

$$
\frac{\partial u}{\partial t} = -\Delta u,
\qquad\Longrightarrow\qquad
u(t) = e^{-t\Delta}\,u_0 .
$$

The operator $e^{-t\Delta}$ is the **heat operator**; its kernel $k_t(x, y)$ tells
you how much heat travels from $y$ to $x$ in time $t$. Expanded in the spectral
basis,

$$
k_t(x, y) = \sum_i e^{-\lambda_i t}\, \phi_i(x)\,\phi_i(y).
$$

Two readings of this single kernel give the two most important descriptors:

- Its **diagonal** — how much heat stays put at each point — is the **Heat Kernel
  Signature**:

  $$
  \mathrm{HKS}(x, t) = k_t(x, x) = \sum_i e^{-\lambda_i t}\, \phi_i(x)^2 .
  $$

- Its **trace** — integrated over the whole surface, since
  $\int_M \phi_i^2 = 1$ — is a global quantity, the **heat trace**:

  $$
  Z(t) = \int_M k_t(x,x)\,dx = \sum_i e^{-\lambda_i t},
  $$

  which is a smooth re-encoding of the spectrum, i.e. of ShapeDNA.

So the global fingerprint (ShapeDNA, via the trace) and the local signature (HKS,
via the diagonal) are *the same object* seen at two granularities. That is the
template for the whole library.

### The general point signature

Replace the heat filter $g(\lambda)=e^{-\lambda t}$ with any other filter $f$ and
take the diagonal, and you get the general **point signature**:

$$
S_f(x) = \sum_i f(\lambda_i)\, \phi_i(x)^2 .
$$

The descriptor families are simply different choices of $f$:

| Filter $f(\lambda)$ | Behavior | Descriptor |
| --- | --- | --- |
| $e^{-\lambda t}$, varying $t$ | low-pass | {doc}`HKS <heat_descriptors>` |
| Gaussian in $\log\lambda$, varying center | band-pass | {doc}`WKS <wave_descriptors>` |
| alternative spectral kernels | re-weighted | {doc}`BKS / IBKS <bates_descriptors>` |
| a bank of dilated band-pass kernels | multi-resolution | {doc}`SGW / ASMWD <wavelet_descriptors>` |
| identity on $\{\lambda_i\}$ | the raw spectrum | {doc}`ShapeDNA <global_embeddings>` |

Seeing the families this way demystifies the zoo: choosing a descriptor *is*
choosing a spectral filter, and the trade-offs (multiscale vs. frequency-sharp,
isotropic vs. anisotropic) are trade-offs between filters.

:::{note}
Why $\phi_i(x)^2$ rather than $\phi_i(x)$? Eigenfunctions are defined only up to
sign (and up to rotation within a degenerate eigenspace), so the raw value
$\phi_i(x)$ is not a well-defined feature of the shape. The square removes the
sign ambiguity and, by analogy with quantum mechanics, reads as the *energy
density* of mode $i$ at point $x$.
:::

## Interpretation: the harmonics of a shape

The eigenfunctions are the **standing waves** of the surface — the patterns it
would vibrate in if it were a drumhead. The eigenvalue $\lambda_i$ is the squared
frequency of mode $i$: small $\lambda$ means a slow, smooth, global undulation;
large $\lambda$ means a fast, wiggly, local one. The zeros of an eigenfunction
(its *nodal lines*) partition the surface the way the nodal lines of a vibrating
plate do.

This is the single most useful mental model. A spectral filter that keeps small
$\lambda$ (a low-pass filter) describes a point by its place in the *global*
shape; a filter that keeps large $\lambda$ describes it by its *local* geometry.
Sweeping the filter — over time $t$ in the HKS, over energy in the WKS — is
sweeping from local to global, which is exactly why these signatures are
**multiscale**.

## What they actually mean — and their limits

Two properties give spectral descriptors their power:

- **Intrinsic.** They depend only on the surface metric, not on the embedding in
  space. Translate, rotate, or reflect the shape and the descriptors do not
  move. There is no registration step and no pose confound.
- **Isometry-invariant.** Bend the surface without stretching it and the
  descriptors are unchanged, because $\Delta$ itself is unchanged.

The natural question — *how much* of the shape does the spectrum capture? — is the
famous one Mark Kac posed in 1966: **"Can one hear the shape of a drum?"** The
answer is a precise *almost*. The spectrum determines a great deal (area,
topology, total curvature — see below), but not everything: there exist
**isospectral, non-isometric** shapes — geometrically different surfaces with
identical spectra. So ShapeDNA is an extremely informative fingerprint, not a
perfect one.

This limitation is exactly why point signatures and embeddings exist. Where the
global spectrum can confuse two distinct shapes, the *spatial* pattern of
$\phi_i(x)^2$ across the surface — what HKS, WKS, and the wavelets encode — adds
the localized information the bare spectrum throws away.

## How they encode morphometry

The bridge from "a list of eigenvalues" to "a measurement of shape" is the
small-time behavior of the heat trace. For a closed surface, the
Minakshisundaram–Pleijel expansion gives

$$
Z(t) = \sum_i e^{-\lambda_i t}
\;\sim\; \frac{\mathrm{Area}(M)}{4\pi t}
\;+\; \frac{\chi(M)}{6}
\;+\; O(t)
\qquad (t \to 0^+),
$$

where $\chi(M)$ is the Euler characteristic. Read term by term, this is a
morphometric statement hiding inside the spectrum:

- The **leading term** is proportional to **surface area** — equivalently,
  Weyl's law $N(\lambda) \sim \tfrac{\mathrm{Area}}{4\pi}\,\lambda$ says the
  density of eigenvalues counts area. The spectrum *measures size*.
- The **constant term** is the **Euler characteristic**, and by the
  Gauss–Bonnet theorem $\int_M K\,dA = 2\pi\chi(M)$, so this term is the
  **total Gaussian curvature** and the **topology** of the structure. The
  spectrum *measures curvature and connectivity*.

So even the global descriptor already carries area, integrated curvature, and
topology. The point signatures then *localize* this: the high-frequency content
of $S_f(x)$ concentrates where curvature is high — protrusions, ridges, the necks
between hippocampal subfields, the banks of a sulcus — so the field
$x \mapsto S_f(x)$ is a curvature- and scale-sensitive map of the structure.

Contrast this with classical morphometry. Volume and thickness compress a
structure to one or two scalars and inherit the registration and voxel-size
sensitivities of the segmentation. Spectral descriptors instead provide a
*pose-free, parameterization-robust* characterization of the full geometry — they
see the difference between two hippocampi of equal volume but different shape,
which is precisely the regime where conditions like mesial temporal lobe epilepsy
leave their signature, and where left/right asymmetries become measurable.

## From the continuous theory to a mesh

In practice the surface is a triangle mesh, and the operator is discretized with
the cotangent stiffness matrix $\mathbf{W}$ and the mass matrix $\mathbf{A}$
(the {doc}`Foundations page <laplace_beltrami>`). Two practical consequences
follow from the theory above:

- **Truncation is band-limiting.** You compute only the first $k$ eigenpairs, so
  every descriptor is implicitly low-passed at mode $k$. Choosing $k$ sets the
  finest geometric scale a descriptor can resolve.
- **Scale matters — sometimes.** Because the leading heat-trace term scales with
  area, raw descriptors respond to overall size. When you want shape *independent*
  of size, use the scale-normalized variants (e.g.
  {doc}`SI-HKS <heat_descriptors>`) or normalize the spectrum, exactly as
  `shapedna_distance` does internally.

```python
import spectralbrain as sb

decomp = sb.BrainMesh(vertices, faces).decompose(k=300)

# Same decomposition, different spectral filters → different descriptors:
sdna = sb.compute_shapedna(decomp)                  # the spectrum (global)
hks  = sb.compute_hks(decomp, t_values=[1, 10, 100])  # low-pass field
wks  = sb.compute_wks(decomp, n_energies=100)         # band-pass field
```

## The family map

Everything in the {doc}`Learn section <index>` is now one statement: each family
is a choice within this single spectral framework.

- **The spectrum, read directly** → {doc}`ShapeDNA & GPS <global_embeddings>`.
- **The diagonal, low-pass** → {doc}`heat descriptors <heat_descriptors>`.
- **The diagonal, band-pass** → {doc}`wave descriptors <wave_descriptors>`.
- **Alternative spectral kernels** → {doc}`Bates–Kornfeld <bates_descriptors>`.
- **A bank of band-pass filters** → {doc}`spectral wavelets <wavelet_descriptors>`.
- **The operator compared across shapes** →
  {doc}`functional maps & distances <functional_maps_and_distances>`.

:::{seealso}
Prerequisite: {doc}`laplace_beltrami`. For the inverse problem and isospectral
shapes, Kac (1966) and Gordon–Webb–Wolpert (1992) are the classic references;
for spectral shape analysis in neuroimaging, Reuter et al. (2006) introduced
ShapeDNA. Hands-on: tutorials `01`–`05`.
:::
