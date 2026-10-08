<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rdneuro/spectralbrain/main/assets/sb_logo_banner_dark.png">
    <img src="https://raw.githubusercontent.com/rdneuro/spectralbrain/main/assets/sb_logo_banner_light.png" alt="spectralbrain: spectral shape analysis for brain structures" width="560">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/spectralbrain/"><img src="https://img.shields.io/pypi/v/spectralbrain.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/spectralbrain/"><img src="https://img.shields.io/pypi/pyversions/spectralbrain.svg" alt="Python"></a>
  <a href="https://github.com/rdneuro/spectralbrain/actions"><img src="https://github.com/rdneuro/spectralbrain/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.21090748"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21090748.svg" alt="DOI"></a>
</p>

<p align="center">
  <a href="#installation">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-fits-together">Workflow</a> ·
  <a href="#the-pipeline-in-practice">Demo</a> ·
  <a href="#documentation-map">Docs map</a> ·
  <a href="#citing">Cite</a>
</p>

---

**SpectralBrain** computes, analyzes and visualizes spectral shape descriptors of
brain structures: cortical surfaces, subcortical and hippocampal meshes, and
white-matter tract surfaces, all handled as triangle meshes. It takes you from
FreeSurfer / HippUnfold / TractSeg output, through the Laplace–Beltrami operator
and statistically rigorous group inference, to publication-ready figures, in one
tested library.

<p align="center">
  <img src="https://raw.githubusercontent.com/rdneuro/spectralbrain/main/assets/spectralbrain_concept.png" alt="SpectralBrain concept: cortical, hippocampal and tract meshes pass through the Laplace-Beltrami operator; its eigenpairs are read out as ShapeDNA, HKS, SI-HKS, WKS, GPS, BKS, functional maps and wavelets" width="720">
</p>

<p align="center">
  <em>The core idea. Any triangle mesh goes in; the Laplace–Beltrami operator (or
  another member of the operator family) yields eigenpairs {λ<sub>k</sub>, φ<sub>k</sub>};
  isometry-invariant spectral descriptors come out.</em>
</p>

## Why spectral shape?

Volume and thickness collapse a structure's shape to a few scalars and are
sensitive to registration and voxel size. **Intrinsic spectral descriptors**
built from the Laplace–Beltrami operator (LBO), such as ShapeDNA and the heat and
wave kernel signatures, describe shape *independently of pose and
parameterization* and capture geometry that volume misses. They are well
established in geometry processing but scattered across research code, and
rarely packaged with the I/O, multi-site harmonization, correct
multiple-comparison statistics and rendering that a neuroimaging study needs end
to end. SpectralBrain fills that gap. Its primary focus is the hippocampus in
mesial temporal lobe epilepsy, but it works on any brain surface mesh.

## How it fits together

Every analysis follows the same five steps: **load geometry → build the operator
and decompose → compute descriptors → group statistics → assess and visualize**.
Each step is owned by one subpackage, and each subpackage can be used on its own.

<p align="center">
  <img src="https://raw.githubusercontent.com/rdneuro/spectralbrain/main/assets/spectralbrain_workflow.png" alt="SpectralBrain workflow: inputs, the five-step main workflow, and the six subpackages core, io, spectral, statistics, viz and utils" width="900">
</p>

<p align="center">
  <em>(a) The ingredients of an analysis: geometry, operator, cohort, atlas,
  descriptors and inference. (b) The five-step workflow, ending in an effect-size
  read-out. (c) The six subpackages:
  <code>core · io · spectral · statistics · viz · utils</code>.</em>
</p>

| Step | Subpackage | What you get |
|---|---|---|
| 1 · Load geometry | `io` | FreeSurfer surfaces and morphometry, GIfTI, NIfTI / MGZ volumes and labels, HippUnfold v1 & v2, `.ply / .obj / .stl / .vtk`, HDF5, with format auto-detection; volume → mesh; TractSeg bundle masks as meshes; BIDS and `SUBJECTS_DIR` cohorts loaded in parallel |
| 2 · Build operator & decompose | `core`, `spectral.operators` | `BrainMesh.decompose()` on CPU, Torch, CuPy or JAX; the same `SpectralDecomposition` from eight operator families beyond the LBO |
| 3 · Compute descriptors | `spectral.lbo` | ShapeDNA, HKS, SI-HKS, WKS, GPS, BKS and its inverse, functional maps, spectral graph wavelets, shape distances |
| 4 · Group statistics | `statistics` | vertex-wise tests with FWER control, TFCE, DeLong, BCa bootstrap, ComBat(-GAM), normative models, Bayesian models, contiguous ddCRP clustering |
| 5 · Assess & visualize | `viz` | template-free six-view renderer, flat maps, posterior plots, 3D tractography, parcellation-vs-clustering grids |

A few things worth knowing:

- **Statistics done right.** Vertex-wise inference uses genuine family-wise error
  control (max-statistic permutation), with FDR and TFCE alongside; partial
  correlations use the correct degrees of freedom; AUCs are compared with the
  analytic DeLong test; six PyMC Bayesian models (horseshoe, hierarchical,
  Gaussian-process normative, spatial, connectome, group comparison) run on
  NUTS, nutpie, NumPyro or BlackJAX.
- **Contiguous clustering, honestly benchmarked.** A distance-dependent Chinese
  Restaurant Process (ddCRP) with a Normal-Inverse-Wishart collapsed likelihood
  (spatial and fPCA-functional variants), consensus clustering and data-driven
  autotuning. Partitions are compared with atlases (ARI / AMI / NMI / VI,
  Dice / Jaccard, within-parcel homogeneity) against **size-matched random
  parcellations**, with spatially-aware ARI (spARI) and eigenstrapping /
  BrainSMASH nulls for bounded surfaces.
- **Beyond the LBO.** Dirac, Steklov, Hamiltonian, Hodge, magnetic, connection and
  sheaf Laplacians, Finsler, biharmonic, persistent Laplacian, graph spectra and
  classical morphometric spectra all return a `SpectralDecomposition`, so every
  descriptor, statistic and figure works on them unchanged.
- **Meshes only, by design.** Point clouds live in the sibling library
  [**pointsbrain**](https://github.com/rdneuro/pointsbrain), which builds a
  point-cloud `SpectralDecomposition` and then reuses all of SpectralBrain.

## Installation

```bash
pip install spectralbrain
```

Optional feature sets (extras):

```bash
pip install "spectralbrain[bayesian]"   # PyMC, nutpie, NumPyro, BlackJAX, ArviZ
pip install "spectralbrain[viz]"        # vedo, fury, trimesh, cmcrameri, ...
pip install "spectralbrain[gpu]"        # torch, CuPy, JAX (CUDA)
pip install "spectralbrain[neuro]"      # nilearn, dipy, pybids, templateflow, ...
pip install "spectralbrain[operators]"  # gudhi, ripser, POT, networkx, ... (non-LBO operators)
pip install "spectralbrain[tuning]"     # optuna (ddCRP autotuning; random fallback)
pip install "spectralbrain[full]"       # everything above
```

Requires Python 3.11–3.12.

## Quick start

The core API is on the top-level package; statistics and visualization live in
submodules you import explicitly (as with `scipy.stats`):

```python
import spectralbrain as sb                   # meshes, descriptors, I/O, cohorts
import spectralbrain.statistics as sbstats   # frequentist + Bayesian
import spectralbrain.viz as sbviz            # 3D / 2D figures
```

### 1 · Mesh → eigenpairs → descriptors

```python
import spectralbrain as sb

# A BrainMesh from vertices (N, 3) and faces (M, 3).
vertices, faces = sb.io.load_gifti_surface("path/to/surf/gii")
mesh   = sb.BrainMesh(vertices, faces)
decomp = mesh.decompose(k=100)                       # 100 LBO eigenpairs

hks = sb.compute_hks(decomp, t_values=[1.0, 10.0, 100.0])   # (N, 3)
wks = sb.compute_wks(decomp, n_energies=50)                 # (N, 50)
dna = sb.compute_shapedna(decomp)                           # (k-1,) global
```

### 2 · Compare two shapes

```python
d = sb.shapedna_distance(dna_a, dna_b)   # pose-invariant spectral distance
```

### 3 · Vertex-wise group statistics with FWER control

```python
import spectralbrain.statistics as sbstats

# controls, patients : (n_subjects, n_vertices) descriptor fields
res = sbstats.vertexwise_permutation(
    controls, patients,
    n_permutations=5000,
    correction="max",      # family-wise error via the max-statistic null
    seed=0,
)
significant = res.significant          # boolean mask, FWER-controlled
```

`correction="fdr"` and `"none"` are also available; `vertexwise_ttest`
defaults to Welch's t-test.

### 4 · Compare two classifiers (analytic DeLong)

```python
auc_new, auc_ref, p = sbstats.auc_comparison_delong(y_true, scores_new, scores_ref)
```

### More recipes

<details>
<summary><strong>Six-view 3D render, template-free</strong></summary>

```python
import spectralbrain.viz as sbviz

fig = sbviz.plot_hippocampus_sixview(
    mesh, scalars=hks[:, 1],
    cmap="plasma", scalar_bar_title="HKS(t=10)",
    save="hipp_sixview.png",
)
# Pick any subset/order of the six canonical views:
fig = sbviz.plot_hippocampus_sixview(mesh, scalars=hks[:, 1],
                                     views=("superior", "left_lateral"))
```

Views: `anterior, posterior, inferior, superior, left_lateral, right_lateral`.
It renders *any* surface (HippUnfold v2 `den-8k`, an `aseg` ROI mesh, a whole
cortical hemisphere) with no bundled template, so scalar ↔ vertex correspondence
is guaranteed.

</details>

<details>
<summary><strong>Bayesian sparse regression</strong> (extra: <code>[bayesian]</code>)</summary>

```python
from spectralbrain.statistics import HorseshoeRegression

model = HorseshoeRegression(tau_prior=0.5).fit(X, y, sampler="nuts")
importance = model.feature_importance()   # sparse posterior shrinkage
```

Bayesian models accept `sampler="auto" | "nuts" | "nutpie" | "numpyro" | "blackjax"`.

</details>

<details>
<summary><strong>Contiguous ddCRP clustering + atlas comparison</strong></summary>

The sampler keeps **incremental per-component sufficient statistics** (the
cluster scatter is never recomputed inside the candidate loop) and caches the NIW
marginal per cluster, so it scales to dense surfaces (~1–3 s per draw at ~7k
vertices). Pass `n_components` to PCA-whiten the descriptors for a further
speed-up and better conditioning.

```python
import spectralbrain.statistics as sbstats

# H : (V, d) per-vertex descriptors (e.g. fused HKS/WKS); vertices/faces from the mesh.
res = sbstats.cluster_ddcrp(H, faces=faces, vertices=vertices,
                            decay_kind="exponential")   # decay uses real edge distances
print(res)                                              # ClusterResult(method='ddcrp', n_clusters=...)

# Tune hyperparameters from the data (optuna if installed, else random search):
tuned = sbstats.autotune_ddcrp(H, faces=faces, vertices=vertices, n_trials=40)
final = tuned.cluster_result                            # refit ClusterResult

# Is the partition more atlas-like than a size-matched random parcellation?
report = sbstats.cluster_atlas_concordance(final.labels, atlas_labels,
                                           faces=faces, coords=vertices, n_null=1000)
print(report["metrics"]["ari"], report["ari_null"]["z"], report["spARI"]["spARI"])
```

</details>

<details>
<summary><strong>Tracts in 3D and parcellation-vs-clustering grids</strong> (extra: <code>[viz]</code>)</summary>

```python
import spectralbrain.viz as sbviz

# (a) 3D tractography from several points of view (FURY; DEC orientation colours)
sl, _ = sbviz.load_streamlines("CST_left.trk", to_space="world")   # needs dipy
fig, _ = sbviz.streamlines_multiview(sl, views=("left", "anterior", "superior", "oblique"),
                                     out_path="cst_multiview.pdf")

# bundle surface from a TractSeg mask, with an HKS overlay
V, F = sbviz.mask_to_mesh(mask, affine=affine)
hks = sbviz.spectral_overlay(V, F, kind="hks")
sbviz.render_bundle_surface(V, F, scalars=hks, view="oblique", out_path="cst_hks.png")

# (b) 3D grid: columns = views, rows = a reference parcellation then each clustering
fig, meta = sbviz.plot_parcellation_vs_clusters(
    vertices, faces, parcellation=atlas_labels,
    clusterings={"ddCRP": res, "Leiden": leiden_res},   # ClusterResult or label arrays
    views=["left_lateral", "anterior", "superior"],
    save="parcellation_vs_clusters.png",
)
```

The grid works the same way for hippocampi, whole brains and bundle surfaces: it
takes any `(vertices, faces)` mesh plus a dict of per-vertex labelings.

</details>

<details>
<summary><strong>Loading a cohort</strong></summary>

```python
import spectralbrain as sb

# BIDS / derivatives (one file per subject):
files = sb.discover_bids("/data/derivatives/hippunfold",
                         "sub-{sub}/surf/sub-{sub}_hemi-L_*thickness.shape.gii")
group = sb.load_group(files, mode="maps", n_jobs=8)
res   = sb.group_comparison(group, group.covariate("group"), test="ttest")

# FreeSurfer SUBJECTS_DIR, resampled to a common template:
group = sb.load_group_freesurfer("/data/fs", measure="thickness",
                                 template="fsaverage", n_jobs=8)

# TractSeg bundle masks -> meshes ready for .decompose():
bundles = sb.load_tractseg("/data/sub-01/tractseg_output", output="mesh")
decomp  = bundles["CST_left"].decompose(k=80)
```

`mode="pipeline"` runs load → decompose → descriptor per subject (with an
optional GPU `backend=`); `mode="maps"` stacks vertex-corresponded fields.

</details>

<details>
<summary><strong>GPU eigensolvers</strong> (extra: <code>[gpu]</code>)</summary>

```python
from spectralbrain.core.backends import TorchBackend       # or CupyBackend, JaxBackend

decomp = mesh.decompose(k=200, backend=TorchBackend())     # GPU eigensolve
```

</details>

## The pipeline in practice

The five steps run end to end on real study designs. The figure below carries
two analyses through the whole pipeline side by side and adds a parallel
**Bayesian lane**, so the same data are assessed with frequentist tools
(max-statistic permutation, TFCE, ROC / DeLong) and Bayesian ones (hierarchical
models, horseshoe priors, HDI + ROPE, posterior-predictive checks, LOO).

<p align="center">
  <img src="https://raw.githubusercontent.com/rdneuro/spectralbrain/main/assets/spectralbrain_demo.png" alt="SpectralBrain end-to-end demonstration: hippocampal lateralization, cortical morphometry and a Bayesian lane carried across the five pipeline steps" width="900">
</p>

<p align="center">
  <em><strong>Columns</strong> are worked analyses: hippocampal lateralization in
  MTLE-HS (HippUnfold <code>den-8k</code> surfaces, L vs R), cortical morphometry
  (patients vs controls, FreeSurfer / Schaefer-200) and a Bayesian lane.
  <strong>Rows</strong> are the five pipeline steps, each tagged with the
  subpackage that implements it. Meshes, eigenmodes, spectra and HKS / WKS fields
  are computed; the statistical panels are illustrative, generated from synthetic
  example data to show the shape of an analysis, not empirical results.</em>
</p>

## Documentation map

```
spectralbrain/
├── runtime.py            types, logging, seeds, progress, containers, joblib, RAM/VRAM guards
├── core/                 BrainMesh, SpectralDecomposition, GeometricObject, compute backends
├── io/                   loaders/savers, meshing, parcellation, cohorts, TractSeg, preprocessing
├── spectral/
│   ├── lbo/              descriptors, distances, wavelets, collections, anisotropic, eigenmodes
│   └── operators/        Dirac/Steklov, Hamiltonian, Hodge/Ricci/sheaf, persistent, Finsler, graphs, ...
├── statistics/
│   ├── analysis.py       vertex-wise inference, effect sizes, RSA, networks, EDA/QC, recommendations
│   ├── bayesian.py       Bayesian models + MCMC samplers
│   ├── normative.py      harmonisation, normative models, method comparison
│   ├── surrogates.py     bootstrap, null models, synthetic data
│   └── clustering/       clustering methods, ddCRP, cluster-vs-atlas statistics
├── utils/                atlases, example data, helpers
└── viz/                  graphics · bayes · clusters · spectral · hipp · render3d
```

`spectralbrain.spectral.operators` is lazy-loaded (extra `[operators]`). Each
`viz` module mirrors one analysis module and shares the 3D engine in `render3d`.
Upgrading from 0.0.x? The old → new import table is in
[`CHANGELOG.md`](CHANGELOG.md).

## Validation

- `examples/example_clustering_tracts.py` runs the whole pipeline end to end
  (HippUnfold hippocampus → HKS/WKS → ddCRP → atlas statistics → 3D figures),
  with a synthetic fallback when no data are present.
- `validation/validate_spari.py` checks the spatially-aware Rand index: its
  analytical properties (identity, label invariance, symmetry, **reduction to the
  exact ARI as the spatial scale vanishes**, chance level, locality monotonicity)
  run anywhere, and an R cell-for-cell bridge (`--with-r`) against the published
  `spARI` package justifies `validated_against_R=True`.

## Development

```bash
git clone https://github.com/rdneuro/spectralbrain
cd spectralbrain
uv sync --group dev          # or: pip install -e ".[full]" + dev tools
uv run pytest                # run the test suite
uv run ruff check src/ tests/
```

## Citing

If SpectralBrain contributes to your work, please cite it. Someday, maybe, if we
feel lucky, a JOSS paper will be submitted; until then, cite the archived release
on Zenodo:

> Debona, R. *SpectralBrain: Spectral Shape Analysis for Brain Structures*.
> Zenodo. https://doi.org/10.5281/zenodo.21090748

The DOI [10.5281/zenodo.21090748](https://doi.org/10.5281/zenodo.21090748)
always resolves to the latest release. [`CITATION.cff`](CITATION.cff) holds a
machine-readable citation, which GitHub's **"Cite this repository"** button reads.

## License

MIT, see [`LICENSE`](LICENSE).
