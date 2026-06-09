# SpectralBrain tutorials

A ten-notebook, hands-on introduction to spectral shape analysis of brain
structures with SpectralBrain. The series runs end to end on a small bundled
dataset of five real subjects (no downloads), and is designed to be worked
through in order: each notebook builds on the previous one, from the
Laplace-Beltrami operator to a full Bayesian, publication-ready workflow.

## The series

| # | Notebook | Topic | Key functions |
|---|----------|-------|---------------|
| 01 | `01_laplace_beltrami_operator.ipynb` | The LBO: eigenvalues, eigenfunctions, Weyl's law, isometry invariance | `BrainMesh`, `decompose` |
| 02 | `02_reading_real_brains_io.ipynb` | Loading FreeSurfer, GIfTI, segmentations, HippUnfold, TractSeg | `load_freesurfer_surface`, `load_gifti_surface`, `marching_cubes`, `load_tractseg` |
| 03 | `03_shapedna_global_fingerprint.ipynb` | ShapeDNA, heat-trace asymptotics, scale invariance, shape distance | `compute_shapedna`, `shapedna_distance` |
| 04 | `04_heat_kernel_signature.ipynb` | HKS, multi-scale diffusion, curvature link, SI-HKS | `compute_hks`, `compute_si_hks` |
| 05 | `05_wave_kernel_and_gps.ipynb` | WKS (band-pass) and the GPS embedding | `compute_wks`, `compute_gps` |
| 06 | `06_pointclouds_and_tracts.ipynb` | Point clouds, the robust Laplacian, BKS / iBKS, white-matter tracts | `BrainPointCloud`, `compute_bks`, `compute_ibks` |
| 07 | `07_functional_maps_and_distances.ipynb` | Functional maps and intrinsic / extrinsic distances | `compute_functional_map`, `biharmonic_distance`, `chamfer_distance` |
| 08 | `08_cohorts_and_vertexwise_stats.ipynb` | Cohorts and vertex-wise statistics with FWE / FDR / TFCE | `load_group`, `vertexwise_permutation`, `tfce`, `cohens_d_map` |
| 09 | `09_effectsizes_classification_harmonization.ipynb` | AUC / DeLong, bootstrap, ICC, classification, ComBat | `auc_comparison_delong`, `bootstrap_ci`, `classify`, `harmonize_combat` |
| 10 | `10_bayesian_and_visualization.ipynb` | Bayesian models and the visualization capstone | `HorseshoeRegression`, `BayesianGroupComparison`, `GaussianProcessNormative` |

Each notebook ends with five exercises and a pointer to the next.

## Running them

From the repository root:

```bash
pip install -e ".[bayesian,viz,neuro,notebooks]"
jupyter lab tutorials/
```

Point-cloud notebooks (06, and the tract parts of 02) require
[`robust_laplacian`](https://pypi.org/project/robust-laplacian/), which is
included in the `neuro` and `notebooks` extras above.

## The bundled dataset (`tutorials/data/`)

Five real subjects, each chosen to exercise one ingestion path:

| Subject | Modality | Contents |
|---------|----------|----------|
| sub01, sub02 | FreeSurfer | pial surfaces, `aseg`, hippocampal subfields, morphometry |
| sub03, sub04 | HippUnfold v2 | `den-8k` hippocampal midthickness surfaces (L/R) |
| sub05 | TractSeg | 15 white-matter bundles (binary masks derived from TOMs) + one example TOM |

The TractSeg masks were derived from Tract Orientation Maps by collapsing the
orientation vectors to a binary mask; notebook 02 shows the conversion.

## A note on sample size

Five subjects are enough to demonstrate every function and concept, but far too
few for real group inference. Notebooks 08–10 therefore use small, **clearly
labelled synthetic cohorts built on the real template geometry** for the
statistical and Bayesian demonstrations. These illustrate the methods only and
make no scientific claim. To scale to a real study, swap the single-subject
loaders in notebook 02 for `discover_bids` / `discover_freesurfer` and let
`load_group` stream the cohort.
