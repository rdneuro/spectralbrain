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

**SpectralBrain** is a Python library for computing, analyzing, and visualizing
spectral shape descriptors of brain structures — cortical surfaces, subcortical
meshes, hippocampal subfields, white-matter tracts, and point clouds derived
from volumetric segmentations.

It bridges spectral geometry (Laplace–Beltrami eigenpairs) with clinical
neuroimaging, providing a unified pipeline from FreeSurfer/HippUnfold output to
publication-ready statistical analysis and visualization.

## Key Capabilities

**Spectral Descriptors** — 12+ descriptors computed from the Laplace–Beltrami
operator eigenpairs: ShapeDNA, Heat Kernel Signature (HKS), Scale-Invariant HKS,
Wave Kernel Signature (WKS), Global Point Signature (GPS), Bates–Kornfeld
Signature (BKS), Inverse BKS, Spectral Graph Wavelets, and more.

**Input Agnostic** — Load FreeSurfer surfaces (`.pial`, `.white`, `.inflated`),
GIfTI (`.surf.gii`, `.func.gii`), NIfTI volumetric labels, HippUnfold outputs,
generic meshes (`.ply`, `.obj`, `.stl`), or point clouds. Automatic format
detection.

**Multi-site Harmonization** — Built-in ComBat and ComBat-GAM for removing batch
effects across multiple datasets before normative modeling.

**Bayesian Statistics** — Six PyMC-based models: Horseshoe Regression, Bayesian
Group Comparison (BEST), Hierarchical Linear Model, Gaussian Process Normative,
Bayesian Spatial Model (GMRF), and Bayesian Connectome Comparison. Multiple
sampler backends (NUTS, nutpie, NumPyro).

**Normative Modeling** — Build age- and sex-conditioned normative distributions
from healthy reference cohorts; score individual patients as z-score deviation
maps.

**Clustering** — 16 clustering methods for spatial, temporal, and joint
spatio-temporal analysis of descriptor fields, including HDBSCAN, Leiden,
graph-regularized NMF, persistence-based, Mapper (TDA), tensor decomposition,
and scale-space blob tracking.

**Visualization** — 7 visualization modules: 2D statistical graphics, cortical
and subcortical brain plots (via `yabplot`), hippocampal surface rendering
(via `hippunfold-plot`), Bayesian posterior plots, 3D mesh and point cloud
rendering (via `vedo`/`open3d`/`pyvista`), and cluster visualization.

**GPU Acceleration** — Optional CUDA backends (PyTorch, CuPy) for
eigenpair computation, descriptor calculation, and batch processing. CPU
parallelization via `joblib` for multi-subject pipelines.

## Installation

### From PyPI (minimal)

```bash
pip install spectralbrain
```

### With optional extras

```bash
# Bayesian models (PyMC + ArviZ)
pip install "spectralbrain[bayesian]"

# GPU acceleration (PyTorch CUDA)
pip install "spectralbrain[gpu]"

# Full visualization suite (vedo, open3d, yabplot, hippunfold-plot)
pip install "spectralbrain[viz]"

# Neuroimaging I/O (nilearn, dipy, templateflow)
pip install "spectralbrain[neuro]"

# Everything
pip install "spectralbrain[full]"
```

### From source (development)

```bash
git clone https://github.com/rdneuro/spectralbrain.git
cd spectralbrain
pip install -e ".[full]"

# Or with uv (faster)
uv sync --all-extras --group dev
```

### Conda environment

```bash
conda env create -f environment.yml
conda activate spectralbrain
pip install -e ".[full]"
```

## Quick Start

```python
import spectralbrain as sb

# Load a FreeSurfer surface
vertices, faces = sb.load_freesurfer_surface("lh.pial")

# Build a mesh object and compute eigenpairs
mesh = sb.BrainMesh(vertices, faces)
mesh.compute_eigenpairs(k=300)

# Compute spectral descriptors
hks = sb.compute_hks(mesh.eigenvalues, mesh.eigenvectors,
                     t_values=[1, 10, 100])
wks = sb.compute_wks(mesh.eigenvalues, mesh.eigenvectors)
shapedna = sb.compute_shapedna(mesh.eigenvalues, n=50)
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `spectralbrain.core` | `BrainMesh`, `BrainPointCloud`, `SpectralDecomposition` |
| `spectralbrain.io` | Format detection, loaders, exporters, parcellation |
| `spectralbrain.spectral` | All spectral descriptors, distances, wavelets |
| `spectralbrain.statistics` | EDA, frequentist tests, Bayesian models, normative, clustering, surrogates |
| `spectralbrain.backends` | CPU (NumPy/SciPy) and GPU (PyTorch/CuPy) backends |
| `spectralbrain.viz` | 7 visualization submodules |
| `spectralbrain.utils` | Atlases, datasets, helpers |

### Descriptors

```python
# Heat Kernel Signature — multiscale shape descriptor
hks = sb.compute_hks(eigenvalues, eigenvectors, t_values=[1, 10, 100])

# Wave Kernel Signature — frequency-localized shape descriptor
wks = sb.compute_wks(eigenvalues, eigenvectors, n_scales=100)

# ShapeDNA — global shape fingerprint
dna = sb.compute_shapedna(eigenvalues, n=100)

# Global Point Signature
gps = sb.compute_gps(eigenvalues, eigenvectors, n=50)

# All descriptors at once
all_desc = sb.compute_all_descriptors(eigenvalues, eigenvectors)
```

### Multi-site Harmonization

```python
from spectralbrain.statistics import harmonize_combat, harmonize_combat_gam

# Standard ComBat
result = harmonize_combat(
    data, sites=site_labels,
    covariates=np.column_stack([ages, sex]),
)
harmonized = result.data_harmonized

# ComBat-GAM (nonlinear age effects)
result = harmonize_combat_gam(
    data, sites=site_labels,
    continuous_covariates=ages[:, None],
    continuous_names=["age"],
    smooth_terms=["age"],
)
```

### Normative Modeling

```python
from spectralbrain.statistics import NormativeModel

# Build normative from healthy controls
norm = NormativeModel(method="gaussian")
norm.fit(descriptors_hc, ages=ages_hc, sex=sex_hc,
         sites=sites_hc, harmonize_method="combat")

# Score a patient
z_scores = norm.score(descriptor_patient, age=45, sex=1)
```

### Bayesian Models

```python
from spectralbrain.statistics import HorseshoeRegression

model = HorseshoeRegression(tau_prior=0.1)
model.fit(descriptors, clinical_scores, sampler="auto")
model.summary(var_names=["beta"])
importance = model.feature_importance()
```

### Visualization

```python
from spectralbrain.viz import (
    plot_brain, plot_hippocampus, plot_posterior,
    plot_forest, plot_mesh, plot_point_cloud,
)

# Cortical surface plot
plot_brain(data=hks[:, 0], atlas="schaefer_400")

# Hippocampal subfield rendering
plot_hippocampus(values=z_scores, density="0p5mm")

# Bayesian posterior
plot_posterior(samples, hdi_prob=0.94, rope=(-0.1, 0.1))

# Forest plot
plot_forest(var_names=["HKS", "WKS", "GPS"], posteriors=[...])
```

## Supported Formats

| Origin | Formats |
|--------|---------|
| FreeSurfer | `.pial`, `.white`, `.inflated`, `.sphere`, `.annot`, `.label`, `.curv`, `.thickness`, `.sulc` |
| GIfTI | `.surf.gii`, `.func.gii`, `.label.gii` |
| NIfTI | `.nii`, `.nii.gz` (volumetric labels → point clouds) |
| HippUnfold | Hippocampal subfield surfaces and labels |
| TractSeg | White-matter tract surfaces |
| Generic | `.ply`, `.obj`, `.stl`, `.off`, `.vtk` |

## Supported Atlases

Parcellation is supported for Schaefer (100–1000 parcels), Brainnetome,
Desikan–Killiany, Destrieux, HCP MMP, aseg, hippocampal subfields
(FreeSurfer `segmentHA_T1`), and thalamic nuclei.

## Development

```bash
# Run tests
make test

# Lint
make lint

# Format
make format

# Build
make build
```

## Citation

If you use SpectralBrain in your research, please cite:

```bibtex
@software{debona2024spectralbrain,
  author  = {Debona, Rodrigo},
  title   = {SpectralBrain: Spectral Shape Analysis for Brain Structures},
  year    = {2024},
  url     = {https://github.com/rdneuro/spectralbrain},
}
```

## License

[MIT](LICENSE)

## Acknowledgments

SpectralBrain builds on the work of many open-source projects, including
[lapy](https://github.com/Deep-MI/LaPy),
[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/),
[HippUnfold](https://github.com/khanlab/hippunfold),
[PyMC](https://www.pymc.io/),
[yabplot](https://github.com/yabplot/yabplot),
[vedo](https://vedo.embl.es/), and
[ArviZ](https://www.arviz.org/).

Developed at the Instituto Nacional de Neurociência Translacional (INNT)
and Universidade Federal do Rio de Janeiro (UFRJ), under the supervision
of Dr. Roger Walz, MD, PhD.
