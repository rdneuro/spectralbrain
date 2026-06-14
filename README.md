<p align="center">
  <strong>SpectralBrain</strong><br>
  <em>Spectral Shape Analysis for Brain Structures</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/spectralbrain/"><img src="https://img.shields.io/pypi/v/spectralbrain.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/spectralbrain/"><img src="https://img.shields.io/pypi/pyversions/spectralbrain.svg" alt="Python"></a>
  <a href="https://github.com/rdneuro/spectralbrain/actions"><img src="https://github.com/rdneuro/spectralbrain/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

**SpectralBrain** computes, analyzes, and visualizes spectral shape descriptors
of brain structures — cortical surfaces, subcortical meshes, hippocampal
subfields, white-matter tracts, and point clouds from volumetric segmentations.
It connects spectral geometry (the Laplace–Beltrami operator) to clinical
neuroimaging, with one pipeline from FreeSurfer / HippUnfold output through
statistically rigorous analysis to publication-ready figures.

## Statement of need

Volumetric and thickness measures collapse a structure's shape to a few scalars
and are sensitive to registration and voxel size. **Intrinsic spectral
descriptors** derived from the Laplace–Beltrami operator (LBO) — ShapeDNA, the
Heat/Wave Kernel Signatures, and relatives — characterize shape *independently
of pose and parameterization*, capturing geometry that volume alone misses. They
are well established in geometry processing but scattered across
research code, rarely packaged with the I/O, multi-site harmonization,
correct multiple-comparison statistics, and rendering that a neuroimaging study
needs end to end. SpectralBrain fills that gap as a single, tested library, with
a primary focus on the hippocampus in mesial temporal lobe epilepsy, while
remaining general to any brain surface or point cloud.

## Key capabilities

- **Spectral descriptors** — ShapeDNA, Heat Kernel Signature (HKS),
  Scale-Invariant HKS, Wave Kernel Signature (WKS), Global Point Signature
  (GPS), Bates–Kornfeld Signature (BKS) and its inverse, functional maps, and
  more — all from the LBO eigenpairs of a mesh *or* point cloud.
- **Input-agnostic I/O** — FreeSurfer surfaces and morphometry, GIfTI
  (`.surf.gii` / `.func.gii` / `.shape.gii`), NIfTI / MGZ volumes and labels,
  HippUnfold v1 & v2 outputs, `.ply / .obj / .stl / .vtk`, HDF5, and point
  clouds, with automatic format detection.
- **Cohort loading** — BIDS / derivatives, FreeSurfer `SUBJECTS_DIR`, or an
  explicit list, loaded in parallel and stacked for group analysis; FreeSurfer
  measures can be resampled onto a common template; TractSeg bundle masks import
  directly as point clouds or isosurface meshes.
- **Statistics done right** — vertex-wise tests with genuine family-wise error
  control (max-statistic permutation), FDR, partial correlations with correct
  degrees of freedom, TFCE, the analytic DeLong AUC test, BCa bootstrap, ComBat /
  ComBat-GAM harmonization, and six PyMC Bayesian models.
- **Publication figures** — a template-free six-view 3D renderer (vedo), plus
  unfolded flat-maps, cluster overlays, and Bayesian-posterior plots.

## Installation

```bash
pip install spectralbrain
```

Optional feature sets (extras):

```bash
pip install "spectralbrain[bayesian]"   # PyMC, nutpie, NumPyro, BlackJAX, ArviZ
pip install "spectralbrain[viz]"        # vedo, scienceplots, hippunfold_plot, …
pip install "spectralbrain[gpu]"        # torch, CuPy, JAX (CUDA)
pip install "spectralbrain[neuro]"      # nilearn, dipy, pybids, templateflow, …
pip install "spectralbrain[full]"       # everything above
```

Requires Python 3.11–3.12.

## API at a glance

The core API is on the top-level package; heavier statistics and visualization
live in submodules you import explicitly (mirroring `scipy.stats`):

```python
import spectralbrain as sb               # meshes, descriptors, I/O
import spectralbrain.statistics as sbstats   # frequentist + Bayesian
import spectralbrain.viz as sbviz             # 3D / 2D figures
```

## Quick start

### 1 — Mesh → eigenpairs → descriptors

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

Point clouds work identically — `sb.BrainPointCloud(points).decompose(k=...)`.

### 2 — Compare two shapes

```python
d = sb.shapedna_distance(dna_a, dna_b)   # pose-invariant spectral distance
```

### 3 — Vertex-wise group statistics with FWER control

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

### 4 — Compare two classifiers (analytic DeLong)

```python
auc_new, auc_ref, p = sbstats.auc_comparison_delong(y_true, scores_new, scores_ref)
```

### 5 — Six-view 3D render (template-free)

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
It renders *any* surface — HippUnfold v2 `den-8k`, an `aseg` ROI mesh, or a whole
cortical hemisphere — with no bundled template, so scalar↔vertex correspondence
is guaranteed.

### 6 — Bayesian sparse regression (extra: `[bayesian]`)

```python
from spectralbrain.statistics import HorseshoeRegression

model = HorseshoeRegression(tau_prior=0.5).fit(X, y, sampler="nuts")
importance = model.feature_importance()   # sparse posterior shrinkage
```

## Loading a cohort

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

# TractSeg bundle masks → meshes ready for .decompose():
bundles = sb.load_tractseg("/data/sub-01/tractseg_output", output="mesh")
decomp  = bundles["CST_left"].decompose(k=80)
```

`mode="pipeline"` runs load → decompose → descriptor per subject (with an
optional GPU `backend=`); `mode="maps"` stacks vertex-corresponded fields.

## Compute backends

Eigen-decomposition and Bayesian sampling run on pluggable backends:

```python
from spectralbrain.backends import TorchBackend       # or CupyBackend, JaxBackend
decomp = mesh.decompose(k=200, backend=TorchBackend())  # GPU eigsolve
```

Bayesian models accept `sampler="auto" | "nuts" | "nutpie" | "numpyro" |
"blackjax"`.

## Documentation map

| Subpackage | What it provides |
|---|---|
| `spectralbrain` (top level) | `BrainMesh`, `BrainPointCloud`, `decompose`, all `compute_*` descriptors, distances, I/O, cohort loading |
| `spectralbrain.io` | loaders/savers, BIDS & FreeSurfer discovery, `load_group`, template resampling, TractSeg import, parcellation |
| `spectralbrain.statistics` | vertex-wise tests, TFCE, effect sizes, RSA, classification, ComBat(-GAM), normative models, bootstrap & null models, six Bayesian models |
| `spectralbrain.backends` | CPU / Torch / CuPy / JAX eigensolvers; PyMC / nutpie / NumPyro / BlackJAX samplers |
| `spectralbrain.viz` | six-view 3D renderer, unfolded flat-maps, cluster overlays, Bayesian-posterior and general scientific plots |

## Development

```bash
git clone https://github.com/rdneuro/spectralbrain
cd spectralbrain
uv sync --group dev          # or: pip install -e ".[full]" + dev tools
uv run pytest                # run the test suite
uv run ruff check src/ tests/
```

## Citing

If SpectralBrain contributes to your work, please cite it (a JOSS paper is in
preparation; until then cite the repository and release DOI). See
[`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
