# Changelog

All notable changes to SpectralBrain are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); the project follows
[SemVer](https://semver.org/) once it reaches 1.0.

## [Unreleased]

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
