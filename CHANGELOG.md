# Changelog

All notable changes to **SpectralBrain** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Released versions are tagged in git; the package version is derived from the
git tag at build time via `hatch-vcs`.

## [Unreleased]

### Documentation / packaging
- Rewrote `README.md` as a comprehensive guide whose every code example is
  executed against the real API before shipping (statement of need,
  install/extras, quick start, cohort loading, backends, subpackage map).
- Added a headless **viz + Bayesian CI job** (xvfb + offscreen VTK) so the
  six-view renderer and the corrected statistics paths are exercised on CI,
  not only locally.

### Fixed
- **Statistics audit — corrected several methods (correctness-first):**
  - `vertexwise_permutation` no longer mislabels per-vertex p-values as
    "already corrected." It now offers genuine multiple-comparison
    control: `correction="max"` (default) applies family-wise error rate
    control via the **maximum-statistic** null (Nichols & Holmes 2002);
    `"fdr"` and `"none"` are also available. Empirically the max-stat
    correction holds FWER at the nominal level under the null (was ≈1.0
    with the old labelling).
  - `auc_comparison_delong` now implements the **actual analytic DeLong
    test** (fast midrank algorithm, Sun & Xu 2014) — deterministic and
    exact — instead of the previous bootstrap with a hard-coded seed that
    merely borrowed the name.
  - `vertexwise_correlation` partial correlation uses the correct degrees
    of freedom (`S − 2 − n_covariates`); the previous `S − 2` made partial
    correlations anti-conservative.
  - `non_inferiority_test(paired=False)` now runs a Welch two-sample test
    (Satterthwaite df); previously the `paired` flag was silently ignored
    and the test was always paired.
  - `bootstrap_ci(n_jobs=...)` now actually parallelises (joblib) with
    per-replicate child seeds, so results are identical for any `n_jobs`;
    the parameter was documented but ignored.
  - `BayesianModel.fit`: the `auto` sampler path no longer swallows every
    exception (`except (ImportError, Exception): pass`) when trying
    nutpie — a real model-specification error was silently hidden and the
    run fell through to PyMC. It now distinguishes a missing nutpie
    (`ImportError`, quiet fallback) from a genuine failure (logged
    warning), and `sampler="nuts"` goes straight to PyMC NUTS instead of
    trying nutpie first. Sampler `**kwargs` no longer leak into
    `_build_model` (they were dead there; model data flows via instance
    attributes set in each subclass's `fit`).
  - `compute_icc`: removed a dead `MS_within` expression (the ICC formulae
    themselves were already correct and are unchanged).
- **`parcellate_batch(n_jobs=...)` was ignored** — the parameter was
  documented as the number of parallel workers but the body was a plain
  sequential loop. It now dispatches through `parallel_map` (fail-soft per
  subject preserved) so `n_jobs` actually parallelises.
- **`parallel_map` / `parallel_batch` under the `loky` backend**: the
  progress-bar path captured the Rich progress object (which holds a
  thread lock) inside the worker closure, raising a `PicklingError` when
  run with `n_jobs != 1`. Results are now streamed back and the bar is
  advanced in the parent process, so parallel execution works with
  progress enabled.
- **CupyBackend eigensolver**: the canonical backend (`backends/gpu.py`)
  carried a broken `eigsh` that called CuPy's sparse solver with a mass
  matrix and shift-invert — neither of which CuPy supports. Replaced with
  the validated standardisation `Ã = D^{-1/2} L D^{-1/2}` + dense
  `cupy.linalg.eigh` on the GPU, recovering M-orthonormal eigenvectors,
  with a CPU sparse shift-invert fallback for meshes above `dense_max`
  (20 000 vertices). Verified to machine precision against the analytic
  sphere spectrum and SciPy.
- **JaxBackend on CPU-only installs**: `__init__` called
  `jax.devices("gpu")`, which raises (rather than returning an empty
  list) in modern JAX when no GPU is present, crashing instantiation.
  Now probes defensively and falls back to CPU.
- Generic mesh I/O (`.ply`, `.obj`, `.stl`, `.vtk`, `.vtp`) now goes
  through PyVista, which is a core dependency. Previously these formats
  were routed through `trimesh` — an undeclared optional package that
  additionally could not read or write `.vtk`/`.vtp` at all, leaving
  those declared formats broken on a default install. Both `io.load` /
  `io.load_mesh` (read) and `io.save_mesh` (write) are affected.

### Changed
- **Vertex-wise tests vectorised and made NaN-safe.** `vertexwise_ttest`,
  `vertexwise_mannwhitney`, `vertexwise_permutation`, and
  `vertexwise_correlation` now compute across vertices in one vectorised
  call instead of a Python loop (large speed-up; identical statistics).
  `vertexwise_ttest` defaults to **Welch's t-test** (`equal_var=False`) —
  safer with unequal group sizes/variances; pass `equal_var=True` for the
  previous Student's pooled-variance behaviour.
- Declared `scikit-image` and `joblib` explicitly as core dependencies.
  Both were imported directly (`skimage` by `marching_cubes` and others;
  `joblib` for CPU parallelism) but only present transitively, so a clean
  install could miss them.
- `BayesianModel.fit` now accepts `sampler="blackjax"` (routed through
  PyMC's JAX bridge `sample_blackjax_nuts`) alongside the existing
  `nuts`/`nutpie`/`numpyro` options, and raises `ValueError` on an unknown
  sampler instead of silently doing nothing.
- `null_edge_rewiring` and `null_spin_permutation` gained an `n_jobs`
  parameter for joblib parallelism. Each surrogate is keyed to an
  independent `SeedSequence`-spawned child RNG, so results are **identical
  whether run on one core or many** — only the wall-clock time changes.
- Removed the orphaned top-level `gpu.py`. It was an unreferenced,
  pre-refactor duplicate of `backends/gpu.py` that had diverged from the
  wired-in module; its only newer content (the CupyBackend fix above)
  has been merged into the canonical location.
- Removed the `trimesh` dependency entirely; PyVista (already required)
  covers every generic mesh format for both reading and writing.
- Declared `joblib` explicitly as a core dependency (it was used
  directly for CPU parallelism but only present transitively via
  scikit-learn).

### Added
- **Template-free six-view 3D surface renderer** (`viz.hipp3d`):
  `plot_hippocampus_sixview` / `plot_surface_sixview` render any surface —
  HippUnfold v2 `den-8k`, separate `hipp`/`dentate`, an `aseg` ROI mesh,
  or a whole cortical hemisphere — in the six canonical anatomical views
  (anterior, posterior, inferior, superior, left-lateral, right-lateral).
  Built on **vedo** (offscreen VTK), composited into a matplotlib grid
  with a shared colorbar and italic view labels per the HipPlots
  conventions. Unlike the `hippunfold_plot`/`hippomaps`-backed functions
  in `viz.hipp`, it needs no bundled template, so it works on meshes whose
  vertex count does not match a v1 template (the silent-mismatch trap).
  Accepts a `BrainMesh`, a `(coords, faces)` tuple, or a GIFTI/FreeSurfer
  path; scalar↔vertex correspondence is guaranteed because the field is
  rendered on the very mesh it was computed on. Renders use orthographic
  projection with per-view exact framing (no perspective distortion, no
  cropping, no wasted whitespace) and accept any subset/ordering of views
  via ``views=`` — e.g. ``views=("left_lateral",)`` for a single panel.
  Validated on real HippUnfold den-8k, `aparc+aseg` ROI-17, and cortical
  pial inputs.
- **Template resampling + FreeSurfer group loader** (`io.group`):
  `resample_to_template` brings a native-space per-vertex overlay onto a
  template surface (e.g. fsaverage) via spherical registration
  (`{hemi}.sphere.reg`), with `"nearest"` or inverse-distance `"linear"`
  interpolation. `load_group_freesurfer` loads a morphometry *measure*
  across a cohort, resampling each subject to a common template so the
  group stacks into a vertex-corresponded `(S, N)` array for analysis.
- **TractSeg import** (`io.tractseg`): `discover_tractseg_bundles` /
  `discover_tractseg_subjects` locate bundle-segmentation masks, and
  `load_tractseg` / `load_tractseg_bundle` turn each binary mask into a
  `BrainPointCloud` (mask voxels in world space) or a `BrainMesh`
  isosurface (marching cubes) — both ready for `.decompose()` and the
  spectral descriptors. Empty masks are skipped fail-soft.
- `tests/test_tractseg.py` and additional `tests/test_group.py` cases
  cover bundle discovery, point-cloud/mesh loading and decomposition,
  template-resampling identity, and cross-vertex-count stacking.
- **Group loading** (`spectralbrain.io.group`): an end-to-end path from a
  cohort on disk to the vertex-wise group statistics. `discover_bids`
  (glob with a `{sub}` placeholder), `discover_freesurfer`
  (`SUBJECTS_DIR` + surface/measure), or an explicit list/dict feed
  `load_group`, which loads every subject in parallel (joblib, fail-soft)
  and stacks them into a `GroupData` carrying subject IDs and parsed BIDS
  entities. Two modes: `"maps"` (vertex-corresponded overlays/descriptor
  fields) and `"pipeline"` (load → decompose → descriptor per subject,
  with an optional GPU `backend`). `group_comparison` splits a `GroupData`
  by a covariate and dispatches to `vertexwise_ttest` /
  `vertexwise_mannwhitney` / `vertexwise_permutation`.
- `tests/test_group.py`: covers all three discovery modes, both load
  modes, parallel/sequential invariance, and the analysis glue.
- **`TorchBackend`** — a PyTorch GPU compute backend mirroring the
  `NumpyBackend`/`CupyBackend` interface. Its `eigsh` uses the same
  diagonal-mass standardisation (`Ã = D^{-1/2} L D^{-1/2}` + dense
  `torch.linalg.eigh` on device, CPU sparse fallback above `dense_max`)
  and is selectable via `get_gpu_backend("torch")` and
  `BackendName.TORCH`. Includes `jit` (`torch.compile`) and `vmap`
  (`torch.func.vmap`) helpers for batch descriptor computation.
- **`BlackjaxSampler`** — a BlackJAX GPU Bayesian sampler running the
  window-adaptation → NUTS pipeline on a log-density function, with
  multi-chain support via `jax.vmap` and an `to_arviz` converter that
  works across ArviZ 0.x (`InferenceData`) and 1.x (`DataTree`).
  Selectable via `get_gpu_bayesian_sampler("blackjax")`.
- `tests/test_backends.py`: numerical regression tests for the general
  backends (sphere-spectrum recovery, Torch-vs-NumPy agreement) and the
  Bayesian samplers (mean recovery, multi-chain shapes, factories).
- `tests/test_io.py`: regression tests covering all generic mesh formats
  (read and write round-trips), the `.npz`/`.h5` containers, and
  extension-based format auto-detection.

## [0.1.0] — 2026-06-05

First public release.

### Added
- Core geometric objects: `BrainMesh`, `BrainPointCloud`, and
  `SpectralDecomposition`, with cotangent and robust Laplace–Beltrami
  operators and a `decompose()` entry point.
- Spectral descriptors computed from a `SpectralDecomposition`: ShapeDNA,
  Heat Kernel Signature (HKS), Scale-Invariant HKS, Wave Kernel Signature
  (WKS), Global Point Signature (GPS), Bates signatures, BKS and inverse
  BKS, plus `compute_all_descriptors()`.
- Spectral distances: WESD, ShapeDNA distance, biharmonic, commute-time,
  and diffusion distances; geometric connectome construction; spectral
  graph wavelets; anisotropic Laplacian; functional maps and shape
  difference operators.
- I/O for FreeSurfer surfaces/annotations/morphometry, GIfTI, NIfTI
  volumetric labels, and generic meshes, with automatic format detection
  and a parcellation pipeline.
- Statistics: exploratory QC, vertex-wise frequentist tests with TFCE,
  PyMC-based Bayesian models (including horseshoe regression), normative
  modeling with ComBat / ComBat-GAM harmonization, surrogate generation,
  and a clustering suite.
- CPU (NumPy/SciPy) and optional GPU (PyTorch/CuPy) backends with lazy
  imports so heavy optional dependencies are only loaded on demand.
- Visualization modules for 2D statistical graphics, cortical/subcortical
  brain plots, hippocampal surfaces, Bayesian posteriors, and 3D meshes
  and point clouds.
- Packaging: `py.typed` marker, PEP 621 metadata, hatch-vcs versioning,
  CI (ruff + pytest across Python 3.11/3.12, build check) and trusted
  PyPI publishing.

[Unreleased]: https://github.com/rdneuro/spectralbrain/compare/0.1.0...HEAD
[0.1.0]: https://github.com/rdneuro/spectralbrain/releases/tag/0.1.0
