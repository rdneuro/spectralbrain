from _nbbuild import build, md, code

cells = [
md(r"""# 02 · Reading real brains: the I/O layer

**SpectralBrain tutorial series — notebook 2 of 10.** (Previous: the Laplace–Beltrami operator.)

Notebook 1 decomposed a hippocampus that was already a `BrainMesh`. Real studies
start from a zoo of file formats. This notebook walks the **five subjects** in
`tutorials/data/` through SpectralBrain's I/O layer and turns each into something
`decompose` accepts.

### Learning objectives
1. Load FreeSurfer surfaces, morphometry, segmentations, HippUnfold, and TractSeg.
2. Turn a volumetric label into a surface with marching cubes.
3. Understand the difference between a `BrainMesh` and a `BrainPointCloud`, and
   when the **robust Laplacian** is needed.
"""),

md(r"""## The teaching dataset

Five subjects, each chosen to exercise one ingestion path. Nothing here is
simulated; these are real FreeSurfer / HippUnfold / TractSeg outputs, trimmed to
exactly what the notebooks need."""),

code(r"""import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np, matplotlib.pyplot as plt
import nibabel as nib, spectralbrain as sb
from _tutorial_utils import data_path, DATA

manifest = json.load(open(DATA / "manifest.json"))
for sid, info in manifest["subjects"].items():
    print(f"  {sid}  [{info['modality']:11s}]  {info['note']}")"""),

md(r"""## 1. FreeSurfer cortical surfaces

A FreeSurfer `?h.pial` / `?h.white` file stores a triangular surface (vertices +
faces). `sb.load_freesurfer_surface` returns them as plain arrays; wrap in a
`BrainMesh`. `quality_report()` flags the mesh problems that matter for spectral
work (non-manifold edges, degenerate triangles)."""),

code(r"""v, f = sb.load_freesurfer_surface(data_path("fs", "sub01", "lh.pial.T1"))
cortex = sb.BrainMesh(v, f)
print(f"left pial: {cortex.n_vertices:,} vertices, {cortex.n_faces:,} faces, "
      f"area {cortex.surface_area():.0f} mm^2")
rep = cortex.quality_report()
print("quality_report keys:", list(rep)[:8])"""),

md(r"""## 2. Morphometry: scalars that live on a surface

FreeSurfer also writes per-vertex scalar maps (`?h.area`, `?h.thickness`,
`?h.curv`). These are **fields on the surface** — one number per vertex — exactly
the shape that descriptors and statistics consume. `load_freesurfer_morph`
returns the array; its length must match the surface's vertex count."""),

code(r"""area = sb.load_freesurfer_morph(data_path("fs", "sub01", "lh.area.pial"))
print(f"area map: {area.shape[0]:,} values  (surface has {cortex.n_vertices:,} vertices)")
print(f"matches surface? {area.shape[0] == cortex.n_vertices}")
print(f"area per vertex: mean={area.mean():.3f}, range [{area.min():.2f}, {area.max():.2f}] mm^2")"""),

md(r"""## 3. Subcortical structures: from a label volume to a surface

Subcortical structures live inside a segmentation *volume*, not as a surface. The
FreeSurfer `aseg` uses label **17** for the left hippocampus and **53** for the
right. We isolate the label and run **marching cubes** to extract its boundary
surface, then wrap it in a `BrainMesh`."""),

code(r"""from spectralbrain.core.base import marching_cubes
seg = nib.load(data_path("fs", "sub01", "aparc.a2009s+aseg.mgz"))
vol = np.asarray(seg.dataobj)
for lab, name in [(17, "left hippocampus"), (53, "right hippocampus")]:
    print(f"  label {lab:3d} ({name}): {(vol == lab).sum():,} voxels")

V, F = marching_cubes((vol == 17).astype(np.float32), seg.affine, level=0.5)
aseg_hipp = sb.BrainMesh(V, F)
print(f"\nmarching-cubes mesh: {aseg_hipp.n_vertices:,} vertices, "
      f"closed surface? {aseg_hipp.is_closed()}")"""),

md(r"""## 4. Hippocampal subfields

FreeSurfer's `hippoAmygLabels` segments the hippocampus into subfields and the
amygdala into nuclei. The label codes follow the standard FreeSurfer scheme; a few
hippocampal codes we can name with confidence are below. We list the labels
present, then extract the hippocampal **tail** (226) as a surface."""),

code(r"""HIPP_LABELS = {203: "parasubiculum", 211: "HATA", 212: "fimbria",
               215: "hippocampal fissure", 226: "hippocampal tail"}
hl = nib.load(data_path("fs", "sub01", "lh.hippoAmygLabels-T1.v22.mgz"))
hd = np.asarray(hl.dataobj)
labs, counts = np.unique(hd[hd > 0], return_counts=True)
print(f"{len(labs)} labels present. Largest few:")
for L, c in sorted(zip(labs.astype(int), counts), key=lambda x: -x[1])[:6]:
    print(f"  {L:5d}: {HIPP_LABELS.get(L, '(amygdala nucleus / other)'):24s} {c:,} voxels")

Vt, Ft = marching_cubes((hd == 226).astype(np.float32), hl.affine, level=0.5)
tail = sb.BrainMesh(Vt, Ft)
print(f"\nhippocampal tail surface: {tail.n_vertices:,} vertices")"""),

md(r"""## 5. HippUnfold surfaces

HippUnfold v2 emits dense midthickness surfaces named by *vertex density*
(`den-8k` ≈ 8,000 vertices). These are GIfTI files; `load_gifti_surface` reads
coordinates and faces by data type (not by intent code, which is a common trap).
Subjects 03 and 04 each contribute a left and right hippocampus."""),

code(r"""hipp_meshes = {}
for sid in ["sub03", "sub04"]:
    for hemi in ["L", "R"]:
        p = data_path("hippunfold", sid, f"hemi-{hemi}_space-T1w_den-8k_label-hipp_midthickness.surf.gii")
        vv, ff = sb.load_gifti_surface(p)
        hipp_meshes[f"{sid}_{hemi}"] = sb.BrainMesh(vv, ff)
for name, m in hipp_meshes.items():
    print(f"  {name}: {m.n_vertices:,} vertices, area {m.surface_area():.0f} mm^2")"""),

md(r"""## 6. White-matter tracts: orientation maps to point clouds

TractSeg can output **Tract Orientation Maps (TOMs)**: a 4-D volume
$(X, Y, Z, 3)$ where each voxel holds the bundle's local orientation *vector*.
That is not a binary mask. To get geometry we collapse the vector field to a mask
(a voxel belongs to the bundle if its orientation vector is non-zero), which is
exactly the preprocessing baked into the staged `bundle_segmentations/` here.
Let us see the conversion on the one TOM we kept."""),

code(r"""tom = np.asarray(nib.load(data_path("tractseg", "sub05", "TOM_CST_left.nii.gz")).dataobj)
print(f"TOM shape: {tom.shape}  (the trailing 3 = orientation vector per voxel)")
mask = np.linalg.norm(tom, axis=-1) > 0
print(f"non-zero voxels after collapsing to a mask: {mask.sum():,}")

staged = np.asarray(nib.load(
    data_path("tractseg", "sub05", "bundle_segmentations", "CST_left.nii.gz")).dataobj)
print(f"staged binary mask voxels: {int(staged.sum()):,}  (matches: {int(staged.sum()) == mask.sum()})")"""),

md(r"""`load_tractseg` reads a whole TractSeg directory of binary masks at once. With
`output='pointcloud'` each bundle becomes a `BrainPointCloud` (voxel centres);
with `output='mesh'` each becomes a marching-cubes `BrainMesh`."""),

code(r"""bundles = sb.load_tractseg(data_path("tractseg", "sub05"), output="pointcloud")
print(f"loaded {len(bundles)} bundles as point clouds:")
for name in sorted(bundles)[:6]:
    print(f"  {name:10s}: {bundles[name].points.shape[0]:,} points")
print("  ...")"""),

md(r"""## 7. Meshes vs point clouds, and the robust Laplacian

A `BrainMesh` carries connectivity (faces), so it uses the **cotangent Laplacian**
of notebook 1. A `BrainPointCloud` has *no faces* — just points. For it,
SpectralBrain builds the **robust / intrinsic Laplacian** (Sharp & Crane 2020),
which defines a Laplace operator directly on points (or on poor-quality meshes)
by locally triangulating neighbourhoods. The descriptor pipeline downstream is
identical; only the operator construction differs.

> The robust path needs the `robust_laplacian` package
> (`pip install robust_laplacian`)."""),

code(r"""cst = bundles["CST_left"]
dec_pc = cst.decompose(k=40)                       # robust Laplacian (point cloud)
print(f"point-cloud CST_left: {cst.points.shape[0]:,} points -> "
      f"{dec_pc.n_eigenvalues} eigenvalues, lambda_1={dec_pc.eigenvalues[1]:.3e}")

tract_meshes = sb.load_tractseg(data_path("tractseg", "sub05"), output="mesh")
dec_m = tract_meshes["CST_left"].decompose(k=40)   # cotangent Laplacian (mesh)
print(f"mesh        CST_left: {tract_meshes['CST_left'].n_vertices:,} vertices -> "
      f"lambda_1={dec_m.eigenvalues[1]:.3e}")"""),

md(r"""### Everything reduces to one of two objects

| Input | Loader | Becomes | Laplacian |
|---|---|---|---|
| FreeSurfer `?h.pial` | `load_freesurfer_surface` | `BrainMesh` | cotangent |
| GIfTI `*.surf.gii` (HippUnfold) | `load_gifti_surface` | `BrainMesh` | cotangent |
| `aseg` / subfield label | `marching_cubes` | `BrainMesh` | cotangent |
| TractSeg masks | `load_tractseg` | `BrainPointCloud` / `BrainMesh` | robust / cotangent |

Once you hold a `BrainMesh` or `BrainPointCloud`, every later notebook applies
unchanged."""),

md(r"""## Exercises

1. **Right hippocampus from `aseg`.** Repeat section 3 with label `53` and compare
   the vertex count to the left. Are subcortical marching-cubes meshes closed?
2. **Morphometry sanity.** Load `rh.area.pial` for `sub02` and confirm its length
   matches the `sub02` right pial surface vertex count.
3. **A second subfield.** Extract the **fimbria** (label 212) for `sub01` and
   render it with `plot_surface_sixview`. How many vertices does such a thin
   structure yield?
4. **All bundles.** Build a table of every TractSeg bundle with its point count,
   sorted from largest (CC) to smallest (CA). Which bundles are too small to
   decompose reliably with `k=40`?
5. **Mesh vs cloud spectra.** Decompose `CST_left` both as a point cloud and as a
   mesh (`k=40`) and plot the two spectra together. Do the low eigenvalues agree?
"""),

md(r"""## What's next

With every structure now a `BrainMesh` or `BrainPointCloud`, **notebook 03**
computes the first real descriptor — **ShapeDNA** — and uses it to compare the
four HippUnfold hippocampi, measuring left–right asymmetry and between-subject
distance from the spectrum alone.
"""),
]

build("02_reading_real_brains_io.ipynb", cells, execute=True)
print("NB02 built + executed OK")
