# Changelog

All notable changes to SpectralBrain are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); the project follows
[SemVer](https://semver.org/) once it reaches 1.0.

## [Unreleased]

## [0.0.8] - 2026-10-03

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
