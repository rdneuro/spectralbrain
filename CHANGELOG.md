# Changelog

All notable changes to SpectralBrain are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); the project follows
[SemVer](https://semver.org/) once it reaches 1.0.

## [Unreleased]

## [0.1.0] - Unreleased

Structural release: a leaner taxonomy, one rendering module per analysis
module, and a mesh-only identity. **Breaking**: import paths change (table
below); there are no compatibility shims. Numerical results are unchanged.

### Removed
- **Point clouds.** SpectralBrain now handles triangle meshes only. The
  point-cloud material moved to the sibling library **pointsbrain**, which
  produces a `SpectralDecomposition` and then reuses every descriptor,
  statistic and figure of SpectralBrain: `BrainPointCloud`,
  `farthest_point_sampling`, `compute_adjacency_from_knn`,
  `estimate_point_density`, `detect_density_outliers`, `labels_to_pointcloud`,
  `raw_to_pointcloud`, `SyntheticPointCloud`, `example_point_cloud`, the
  point-cloud renderers of `viz.geometry.points`, the `output="pointcloud"`
  path of the TractSeg loaders (now raises with a pointer to pointsbrain) and
  tutorial 06. `open3d` left the `viz` extra.
- Private duplicates `rsa_compare` / `mantel_test` / `energy_distance` in the
  vendored clustering core (the public ones in `statistics.analysis` remain).
- `cotangent_stiffness` / `lumped_mass` (former `viz.spectral_deformation`):
  spectral deformation now uses the library's own cotangent LBO
  (`BrainMesh(V, F).compute_laplacian()`); outputs agree to 1e-14 on generic
  meshes.
- The lazy `sb.expanded` attribute; use `sb.spectral.operators`.

### Changed
- `spectral/` is the parent of two subpackages: `spectral.lbo` (the
  Laplace-Beltrami trunk) and `spectral.operators` (former `expanded`; its
  `operators.py` is now `hamiltonian.py`, `anisotropic.py` is `finsler.py`,
  `misc.py` is `classical.py`). `from spectralbrain.spectral import compute_hks`
  still works.
- `backends/` was dissolved: array backends -> `core.backends`; samplers ->
  `statistics.bayesian`; joblib helpers and RAM/VRAM guards -> `runtime`.
- `statistics.clustering` is a subpackage (`methods`, `ddcrp`, `atlas_stats`,
  private `_core`); `statistics.eda` merged into `statistics.analysis`;
  `statistics.landmarking` merged into `spectral.lbo.descriptors`.
- `viz` reduced from 10 modules (+ `geometry/`) to six: `graphics`, `bayes`,
  `clusters` (+ parcellation grids), `spectral` (eigenmode panels, deformation
  maps), `hipp` (+ hippocampal six-view) and the shared engine `render3d`
  (cortex/subcortex, tracts, generic meshes, six-view).
- `recommend_descriptor(mesh, ...)` takes a `BrainMesh` or `(vertices, faces)`
  and decomposes surrogates with the cotangent LBO (it used a kNN point-cloud
  Laplacian); rankings may shift slightly.
- `load_tractseg` / `load_tractseg_bundle` default to `output="mesh"`; the
  `jitter*` / `seed` keywords were point-cloud only and are gone.
- The `expanded` extra is now `operators` (`expanded` kept as an alias).
- All `.py` files are plain ASCII (comments, docstrings, log and printed
  messages); Greek symbols in figure labels are rendered with mathtext.
  Eigenmode / deformation code and messages translated to English.

### Added
- `spectralbrain.io.raw_to_mesh` -- raw T1w -> skull-strip -> segment ->
  LBO-ready `BrainMesh` (replaces `raw_to_pointcloud`).
- `spectralbrain.spectral.lbo.modes` -- `compute_geometric_eigenmodes` now
  accepts a path, a `(vertices, faces)` pair or a `BrainMesh` (no yabplot needed
  to compute), plus `eigenmode_wavelength`.
- Explicit `__all__` for `statistics.clustering.methods`.

### Fixed
- `batch_effect_scan` reported `has_batch_effect=False` for constant data with
  SciPy >= 1.11 (NaN p-value); it now reports `None` with a warning.
- The union-find helper `find2` was public by accident (now `_uf_find`).
- Test configuration ignores the PyVista deprecation raised inside `pyacvd`.

### Migration (0.0.x -> 0.1.0)

| 0.0.x | 0.1.0 |
|---|---|
| `spectralbrain.backends.NumpyBackend`, `CupyBackend`, `JaxBackend`, `TorchBackend`, `get_gpu_backend` | `spectralbrain.core.backends` (also `spectralbrain.core`) |
| `spectralbrain.backends.SamplerConfig`, `PyMCSampler`, `NutpieSampler`, `NumPyroSampler`, `BlackjaxSampler`, `get_bayesian_sampler`, `get_gpu_bayesian_sampler` | `spectralbrain.statistics.bayesian` |
| `spectralbrain.backends.parallel_map`, `parallel_batch`, `batch_iterator`, `ram_status`, `memory_guard`, `gc_collect`, `estimate_array_memory`, `shrink_array`, `vram_*` | `spectralbrain.runtime` |
| `spectralbrain.spectral.{anisotropic,collections,descriptors,distances,wavelets}` | `spectralbrain.spectral.lbo.{...}` |
| `spectralbrain.expanded` / `sb.expanded` | `spectralbrain.spectral.operators` |
| `spectralbrain.expanded.operators` | `spectralbrain.spectral.operators.hamiltonian` |
| `spectralbrain.expanded.anisotropic` | `spectralbrain.spectral.operators.finsler` |
| `spectralbrain.expanded.misc` | `spectralbrain.spectral.operators.classical` |
| `spectralbrain.expanded.{_base,biharmonic,extrinsic,graphs,persistent,topologic}` | `spectralbrain.spectral.operators.{...}` |
| `spectralbrain.statistics.landmarking` (`mgp_landmarks`, `compute_si_wks`, `heat_flow_entropy`, `MGPLandmarkResult`) | `spectralbrain.spectral.lbo.descriptors` (also `spectralbrain.spectral`) |
| `spectralbrain.statistics.eda` | `spectralbrain.statistics.analysis` |
| `spectralbrain.statistics.clustering` (module) | `spectralbrain.statistics.clustering` (package; module body in `.methods`) |
| `spectralbrain.statistics.ddcrp` | `spectralbrain.statistics.clustering.ddcrp` |
| `spectralbrain.statistics.atlas_stats` | `spectralbrain.statistics.clustering.atlas_stats` |
| `spectralbrain.statistics._clustercore` | `spectralbrain.statistics.clustering._core` |
| `spectralbrain.viz.brainplots`, `viz.tracts3d`, `viz.geometry.meshes` | `spectralbrain.viz.render3d` |
| `spectralbrain.viz.hipp3d.plot_surface_sixview`, `SIXVIEWS` | `spectralbrain.viz.render3d` |
| `spectralbrain.viz.hipp3d.plot_hippocampus_sixview` | `spectralbrain.viz.hipp` |
| `spectralbrain.viz.panels` | `spectralbrain.viz.clusters` |
| `spectralbrain.viz.eigenmodes` (rendering) | `spectralbrain.viz.spectral` |
| `spectralbrain.viz.eigenmodes.compute_geometric_eigenmodes`, `eigenmode_wavelength` | `spectralbrain.spectral.lbo.modes` |
| `spectralbrain.viz.spectral_deformation.spectral_deformation`, `lateralization_map` | `spectralbrain.spectral.lbo.collections` (also `spectralbrain.spectral`) |
| `spectralbrain.viz.spectral_deformation.render_scale_on_mesh` | `spectralbrain.viz.spectral` |
| `spectralbrain.viz.spectral_deformation.cotangent_stiffness`, `lumped_mass` | `BrainMesh(V, F).compute_laplacian()` |
| `spectralbrain.io.raw_to_pointcloud` | `spectralbrain.io.raw_to_mesh` (or `pointsbrain`) |
| `BrainPointCloud`, `labels_to_pointcloud`, `farthest_point_sampling`, `compute_adjacency_from_knn`, `estimate_point_density`, `detect_density_outliers`, `SyntheticPointCloud`, `example_point_cloud`, `viz.geometry.points.*` | `pointsbrain` |
| `recommend_descriptor(mesh.vertices, ...)` | `recommend_descriptor(mesh, ...)` |
| `load_tractseg(..., output="pointcloud")` | `load_tractseg(...)` (meshes) or `pointsbrain.io.load_tractseg` |
| `pip install spectralbrain[expanded]` | `pip install spectralbrain[operators]` (old name still works) |
| tutorials `07`-`10` | `06`-`09` (06 point clouds moved to pointsbrain) |

## [0.0.7] - 2026-10-03

### Fixed
Library-wide audit for bugs and silent failures (about 120 issues; regression
tests in `tests/test_fix_{core,io,spectral,statistics,viz}.py`). Highlights:

- **Spectral / expanded** — anisotropic Laplacian is symmetric with the correct
  diagonal and `min_curvature` no longer returns the normal; Finsler conductivity
  acts along the requested direction; Hamiltonian spectra keep negative
  eigenvalues; functional maps are full-rank with a documented conformal shape
  difference; Bates and SGW descriptors are sign-invariant; persistent
  Laplacian, Betti curves, persistence images, NetLSD/FGSD normalisations fixed.
- **Core / backends / runtime / utils** — Laplacian caches keyed on the method
  (point-cloud `sigma` no longer leaks into `eigsh`); `shape_index` bounded to
  [-1, 1] and signed mean curvature; degenerate faces dropped (new
  `weld_mesh()`, welded `example_sphere()`); mixed Voronoi areas;
  `marching_cubes` binarises labels and orients faces outward;
  `schaefer_to_yeo` exact from parcel names; GPU CPU-fallback warnings;
  `set_log_level` works; library-wide seed.
- **IO** — SynthStrip/FastSurfer no longer crash; native-space segmentation;
  unique subject keys and `skip_existing`; MNI atlases require a registration;
  correct face orientation for any affine; GIfTI label keys; duplicate subject
  IDs raise; full-precision connectome export.
- **Statistics** — eigenstrapping, Maslov–Sneppen rewiring and phase
  randomisation produce valid nulls; TFCE treats tails separately; NaN-robust
  FDR/correlation; true Jensen–Shannon; ComBat rewritten (`combat_apply`) and
  applied when scoring; GP z-scores include observation noise; Bayesian models
  check diagnostics (`SamplingWarning`), support LOO/WAIC and resized
  prediction; clustering fixes (joint spectral, idct scaling, ToMATo, ST-GNMF).
- **Viz** — explicit RAS cameras for multi-view panels; `savefig` keeps dotted
  names; boolean medial masks; NaN colour; shared colour limits for comparisons.

### Changed (behaviour)
- `parcellate()` raises for MNI-space atlases unless `atlas_reg=` (or
  `atlas_space="native"`) is given.
- `load_group` raises on duplicate subject IDs (pass a `{id: path}` dict).
- `NormativeModel` harmonised with ComBat needs `site=` when scoring.
- Segmentation runs in native space by default (`segment_space="template"`
  restores the old behaviour).
- `compute_functional_map` default `regularize` is now 1e-8.

### Added
- **`spectralbrain.io.meshing`** — deterministic volume → surface meshing that
  produces well-conditioned, LBO-ready meshes from label volumes:
  - `volume_to_mesh(volume, affine, *, raw=False, label=None, closed=True, ...)`
    — signed-distance / Gaussian field → sub-voxel marching cubes → topology
    cleanup → Taubin smoothing → isotropic ACVD remeshing.
  - `refine_mesh(vertices, faces, *, closed=True, ...)` — apply the cleanup /
    smoothing / remeshing steps to an already-triangulated surface.
  - Both are re-exported at the package top level (`sb.volume_to_mesh`,
    `sb.refine_mesh`).
- **`BrainMesh.from_volume(volume, affine, *, raw=False, label=None, ...)`** —
  object-oriented entry point that returns an improved `BrainMesh` directly.
- New core dependencies for the improvement pipeline: `trimesh`, `pyacvd`,
  `pymeshfix` (all lazily imported, with install hints).

### Changed
- **Default meshing behaviour.** Functions that build a mesh from a volume now
  improve the surface by default (`raw=False`). Pass `raw=True` to reproduce the
  previous plain marching-cubes output byte-for-byte.
  - `io.load_tractseg_bundle(output="mesh")` and `io.load_tractseg` gain `raw`
    and use the *open-surface-safe* path (`closed=False`: no watertight forcing,
    no component pruning) — bundle masks are open / branching structures.
  - `viz.tracts3d.mask_to_mesh` gains `raw`; the improved path replaces its
    previous inline Gaussian+Taubin logic with the shared pipeline.
- The low-level primitive `spectralbrain.core.marching_cubes` is **unchanged**:
  it remains a faithful plain-marching-cubes call and backs the `raw=True` path.

### Rationale
- The Laplace–Beltrami operator is sensitive to triangle quality, not vertex
  count; raw marching-cubes meshes carry staircase artefacts, slivers and
  (for multi-label sources) non-manifold / high-genus surfaces that corrupt the
  spectrum and the descriptors built on it (ShapeDNA, HKS, WKS).
- The pipeline is deterministic and shape-agnostic (no learned prior), so it
  never regularises pathological anatomy toward a "normal" shape — important for
  case-control and lateralisation analyses.
- `closed=True` (SDF, single component, watertight repair) is correct for closed
  anatomical structures (hippocampus, subcortical nuclei); `closed=False`
  (Gaussian field, keep components, no watertight forcing) is correct for open /
  branching structures (white-matter bundle masks) so that real anatomy is not
  amputated.
