"""High-level cortical parcellation pipeline.

This module provides the missing bridge between raw data (T1w images
or FreeSurfer outputs) and a ready-to-analyse parcellated surface.
The main entry point is :func:`parcellate`, which auto-detects the
source format and applies the appropriate projection strategy.

Two source modalities are supported:

1. **FreeSurfer subjects_dir** — loads the requested surface, finds
   or projects the target atlas onto it, and returns parcellated
   sub-meshes.
2. **Raw T1w NIfTI** — runs the preprocessing pipeline
   (skull-strip → segment → recon), then follows the FreeSurfer
   pathway.

Three atlas projection strategies are implemented:

A. **Native annot** — the atlas is already on the individual subject
   as a ``.annot`` file (e.g. DKT, Destrieux after ``recon-all``).
B. **fsaverage → individual via surf2surf** — the atlas exists on
   ``fsaverage`` (Schaefer, Glasser) and is resampled to the
   individual via ``mri_surf2surf`` or a Python fallback.
C. **MNI volume → surface via vol2surf** — the atlas is a volumetric
   label map in MNI space (Brainnetome, AAL3, Julich) and is
   projected onto the individual's surface via ``mri_vol2surf`` or a
   pure-Python nearest-voxel sampler. An MNI-space atlas needs a
   registration to the subject (``atlas_reg=``); otherwise pass a volume
   already resampled to subject space with ``atlas_space="native"``.

Example
-------
>>> import spectralbrain as sb
>>>
>>> # From FreeSurfer subjects_dir:
>>> result = sb.io.parcellate(
...     subjects_dir="/data/freesurfer",
...     subject_id="sub-01",
...     atlas="schaefer_200",
...     hemi="lh",
... )
>>> result.parcels[1][0].shape   # vertices of parcel 1
(823, 3)
>>>
>>> # From raw T1w:
>>> result = sb.io.parcellate(
...     t1_path="/data/sub-01/anat/sub-01_T1w.nii.gz",
...     atlas="brainnetome",
...     hemi="lh",
...     atlas_volume_path="/atlases/BN_Atlas_246_1mm.nii.gz",
...     atlas_reg="/data/sub-01/mni_to_sub-01.lta",
... )
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

# Type aliases reused from loaders.py
Vertices = np.ndarray  # (N, 3)
Faces = np.ndarray  # (F, 3)
LabelArray = np.ndarray  # (N,)


# ======================================================================
# §0  Atlas registry — maps atlas names to resolution strategies
# ======================================================================


@dataclass(frozen=True)
class AtlasSpec:
    """Metadata for a supported parcellation atlas.

    Parameters
    ----------
    name : str
        Human-readable atlas name.
    annot_pattern : str or None
        Filename pattern for a FreeSurfer ``.annot`` file on the
        individual or on fsaverage.  ``{hemi}`` is replaced by
        ``'lh'`` or ``'rh'``.
    volume_fetcher : str or None
        Function name in :mod:`nilearn.datasets` that downloads the
        atlas volume, or a direct path / URL.
    strategy : {'native_annot', 'fsaverage_annot', 'mni_volume'}
        Primary projection strategy.
    n_parcels : int or None
        Expected number of parcels per hemisphere (for validation).
    ignore_labels : list of int
        Label IDs to skip (typically [0] for the medial wall).
    """

    name: str
    strategy: Literal["native_annot", "fsaverage_annot", "mni_volume"]
    annot_pattern: str | None = None
    volume_fetcher: str | None = None
    n_parcels: int | None = None
    ignore_labels: list[int] = field(default_factory=lambda: [0])


# Registry of supported atlases.
# Key: canonical short name (lowercased, used in the public API).
ATLAS_REGISTRY: dict[str, AtlasSpec] = {
    # ── Native FreeSurfer parcellations ──
    "dkt": AtlasSpec(
        name="Desikan-Killiany-Tourville (DKT)",
        strategy="native_annot",
        annot_pattern="{hemi}.aparc.DKTatlas.annot",
        n_parcels=31,
    ),
    "desikan": AtlasSpec(
        name="Desikan-Killiany (aparc)",
        strategy="native_annot",
        annot_pattern="{hemi}.aparc.annot",
        n_parcels=34,
    ),
    "destrieux": AtlasSpec(
        name="Destrieux (a2009s)",
        strategy="native_annot",
        annot_pattern="{hemi}.aparc.a2009s.annot",
        n_parcels=74,
    ),
    # ── fsaverage-based (need surf2surf projection) ──
    "schaefer_100": AtlasSpec(
        name="Schaefer 100 (7 networks)",
        strategy="fsaverage_annot",
        annot_pattern="{hemi}.Schaefer2018_100Parcels_7Networks_order.annot",
        n_parcels=50,
    ),
    "schaefer_200": AtlasSpec(
        name="Schaefer 200 (7 networks)",
        strategy="fsaverage_annot",
        annot_pattern="{hemi}.Schaefer2018_200Parcels_7Networks_order.annot",
        n_parcels=100,
    ),
    "schaefer_400": AtlasSpec(
        name="Schaefer 400 (7 networks)",
        strategy="fsaverage_annot",
        annot_pattern="{hemi}.Schaefer2018_400Parcels_7Networks_order.annot",
        n_parcels=200,
    ),
    "schaefer_600": AtlasSpec(
        name="Schaefer 600 (7 networks)",
        strategy="fsaverage_annot",
        annot_pattern="{hemi}.Schaefer2018_600Parcels_7Networks_order.annot",
        n_parcels=300,
    ),
    "schaefer_800": AtlasSpec(
        name="Schaefer 800 (7 networks)",
        strategy="fsaverage_annot",
        annot_pattern="{hemi}.Schaefer2018_800Parcels_7Networks_order.annot",
        n_parcels=400,
    ),
    "schaefer_1000": AtlasSpec(
        name="Schaefer 1000 (7 networks)",
        strategy="fsaverage_annot",
        annot_pattern="{hemi}.Schaefer2018_1000Parcels_7Networks_order.annot",
        n_parcels=500,
    ),
    "glasser": AtlasSpec(
        name="Glasser HCP-MMP 1.0",
        strategy="fsaverage_annot",
        annot_pattern="{hemi}.HCPMMP1.annot",
        n_parcels=180,
    ),
    # ── MNI volume-based (need vol2surf projection) ──
    "brainnetome": AtlasSpec(
        name="Brainnetome Atlas",
        strategy="mni_volume",
        volume_fetcher="fetch_atlas_brainnetome",
        n_parcels=123,  # per hemisphere (246 total)
    ),
    "aal3": AtlasSpec(
        name="AAL3 Atlas",
        strategy="mni_volume",
        volume_fetcher="fetch_atlas_aal",
        n_parcels=85,  # approximate per hemisphere
    ),
    "julich": AtlasSpec(
        name="Julich-Brain Cytoarchitectonic Atlas",
        strategy="mni_volume",
        volume_fetcher="fetch_atlas_juelich",
        n_parcels=None,  # variable
    ),
    "harvard_oxford": AtlasSpec(
        name="Harvard-Oxford Cortical Atlas",
        strategy="mni_volume",
        volume_fetcher="fetch_atlas_harvard_oxford",
        n_parcels=48,
    ),
}

# Aliases for convenience.
ATLAS_REGISTRY["schaefer"] = ATLAS_REGISTRY["schaefer_200"]
ATLAS_REGISTRY["aparc"] = ATLAS_REGISTRY["desikan"]
ATLAS_REGISTRY["mmp"] = ATLAS_REGISTRY["glasser"]
ATLAS_REGISTRY["hcp"] = ATLAS_REGISTRY["glasser"]


def list_atlases() -> list[str]:
    """Return the names of all supported atlases."""
    return sorted(ATLAS_REGISTRY.keys())


def _resolve_atlas(name: str) -> AtlasSpec:
    """Look up an atlas by short name (case-insensitive)."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in ATLAS_REGISTRY:
        available = ", ".join(sorted(ATLAS_REGISTRY.keys()))
        raise ValueError(f"Unknown atlas '{name}'.  Available: {available}")
    return ATLAS_REGISTRY[key]


# ======================================================================
# §1  ParcellationResult — output container
# ======================================================================


@dataclass
class ParcellationResult:
    """Container for the output of :func:`parcellate`.

    Attributes
    ----------
    atlas : AtlasSpec
        The resolved atlas specification.
    hemi : str
        Hemisphere (``'lh'`` or ``'rh'``).
    surface : str
        FreeSurfer surface name (``'white'``, ``'pial'``, ``'inflated'``).
    vertices : ndarray, shape (N, 3)
        Full hemisphere mesh vertices.
    faces : ndarray, shape (F, 3)
        Full hemisphere mesh faces.
    labels : ndarray, shape (N,)
        Per-vertex parcel labels.
    label_names : list of str
        Human-readable parcel names (same order as ``labels``).
    parcels : dict of {int: (vertices, faces)}
        Sub-meshes per parcel (from :func:`apply_parcellation`).
    strategy_used : str
        Which projection strategy was actually used.
    """

    atlas: AtlasSpec
    hemi: str
    surface: str
    vertices: np.ndarray
    faces: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    parcels: dict[int, tuple[Vertices, Faces]]
    strategy_used: str

    @property
    def n_parcels(self) -> int:
        """Number of non-empty parcels."""
        return len(self.parcels)

    @property
    def parcel_ids(self) -> list[int]:
        """Sorted list of parcel IDs."""
        return sorted(self.parcels.keys())

    def get_parcel(self, label_id: int) -> tuple[Vertices, Faces]:
        """Return (vertices, faces) for a specific parcel."""
        if label_id not in self.parcels:
            raise KeyError(f"Parcel {label_id} not found.  Available: {self.parcel_ids}")
        return self.parcels[label_id]

    def summary(self) -> str:
        """Print a brief summary of the parcellation."""
        sizes = {k: v[0].shape[0] for k, v in self.parcels.items()}
        return (
            f"ParcellationResult(atlas={self.atlas.name}, "
            f"hemi={self.hemi}, surface={self.surface}, "
            f"n_parcels={self.n_parcels}, "
            f"mean_vertices={np.mean(list(sizes.values())):.0f}, "
            f"strategy={self.strategy_used})"
        )


# ======================================================================
# §2  FreeSurfer environment helpers
# ======================================================================


def _get_subjects_dir(subjects_dir: PathLike | None = None) -> Path:
    """Resolve SUBJECTS_DIR from arg or environment."""
    if subjects_dir is not None:
        return Path(subjects_dir)
    env = os.environ.get("SUBJECTS_DIR")
    if env:
        return Path(env)
    raise OSError(
        "SUBJECTS_DIR not set and no subjects_dir argument provided.  "
        "Either pass subjects_dir= or set the SUBJECTS_DIR environment "
        "variable."
    )


def _find_freesurfer_home() -> Path | None:
    """Locate FREESURFER_HOME if available."""
    home = os.environ.get("FREESURFER_HOME")
    if home and Path(home).exists():
        return Path(home)
    return None


def _has_freesurfer_cmd(cmd: str) -> bool:
    """Check if a FreeSurfer command is available on PATH."""
    try:
        subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ======================================================================
# §3  Strategy A — native annot (DKT, Destrieux)
# ======================================================================


def _parcellate_native_annot(
    subjects_dir: Path,
    subject_id: str,
    atlas: AtlasSpec,
    hemi: str,
    surface: str,
) -> ParcellationResult:
    """Load a parcellation that's already on the individual subject."""
    from spectralbrain.io.loaders import (
        apply_parcellation,
        load_freesurfer_annot,
        load_freesurfer_surface,
    )

    # Surface
    surf_path = subjects_dir / subject_id / "surf" / f"{hemi}.{surface}"
    if not surf_path.exists():
        raise FileNotFoundError(f"Surface not found: {surf_path}")
    vertices, faces = load_freesurfer_surface(surf_path)

    # Annotation
    annot_file = atlas.annot_pattern.format(hemi=hemi)
    annot_path = subjects_dir / subject_id / "label" / annot_file
    if not annot_path.exists():
        raise FileNotFoundError(
            f"Annotation not found: {annot_path}.  "
            f"Run 'recon-all' or ensure this atlas is generated."
        )
    labels, _ctab, names = load_freesurfer_annot(annot_path)

    # Parcellate
    parcels = apply_parcellation(
        vertices,
        faces,
        labels,
        ignore_labels=atlas.ignore_labels,
    )

    logger.info(
        "Strategy A (native_annot): %s → %d parcels",
        atlas.name,
        len(parcels),
    )
    return ParcellationResult(
        atlas=atlas,
        hemi=hemi,
        surface=surface,
        vertices=vertices,
        faces=faces,
        labels=labels,
        label_names=names,
        parcels=parcels,
        strategy_used="native_annot",
    )


# ======================================================================
# §4  Strategy B — fsaverage annot → individual via surf2surf
# ======================================================================


def _find_fsaverage_annot(
    atlas: AtlasSpec,
    hemi: str,
    subjects_dir: Path,
) -> Path | None:
    """Search for the atlas .annot on fsaverage in several locations."""
    annot_file = atlas.annot_pattern.format(hemi=hemi)

    search_paths = [
        # 1. fsaverage in subjects_dir (standard after recon-all)
        subjects_dir / "fsaverage" / "label" / annot_file,
        # 2. FREESURFER_HOME/subjects/fsaverage
    ]
    fs_home = _find_freesurfer_home()
    if fs_home:
        search_paths.append(fs_home / "subjects" / "fsaverage" / "label" / annot_file)

    for p in search_paths:
        if p.exists():
            logger.info("Found fsaverage annot: %s", p)
            return p

    return None


def _surf2surf_freesurfer(
    subjects_dir: Path,
    subject_id: str,
    hemi: str,
    source_annot: Path,
    target_annot: Path,
) -> None:
    """Project an annot from fsaverage to individual using mri_surf2surf.

    ``mri_surf2surf`` resolves both subjects inside a single ``--sd``. When
    the fsaverage that holds *source_annot* is not under *subjects_dir*
    (e.g. it lives in ``$FREESURFER_HOME/subjects``), a temporary
    SUBJECTS_DIR with symlinks to both subjects is used instead.
    """
    import shutil

    source_annot = Path(source_annot)
    target_annot = Path(target_annot)
    fsa_dir = source_annot.resolve().parent.parent  # .../fsaverage/label/<annot>

    with tempfile.TemporaryDirectory(prefix="sb_surf2surf_") as tmp_name:
        tmp = Path(tmp_name)
        sd = subjects_dir
        local_fsa = subjects_dir / "fsaverage" / "surf" / f"{hemi}.sphere.reg"
        if not local_fsa.exists():
            sd = tmp / "subjects"
            sd.mkdir()
            (sd / "fsaverage").symlink_to(fsa_dir, target_is_directory=True)
            (sd / subject_id).symlink_to(
                (subjects_dir / subject_id).resolve(), target_is_directory=True
            )
            logger.info("fsaverage not in %s; using temporary SUBJECTS_DIR %s", subjects_dir, sd)
        tmp_out = tmp / target_annot.name
        cmd = [
            "mri_surf2surf",
            "--srcsubject",
            "fsaverage",
            "--trgsubject",
            subject_id,
            "--hemi",
            hemi,
            "--sval-annot",
            str(source_annot),
            "--tval",
            str(tmp_out),
            "--sd",
            str(sd),
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"mri_surf2surf failed (exit {result.returncode}):\n"
                f"  stdout: {result.stdout[-500:]}\n"
                f"  stderr: {result.stderr[-500:]}"
            )
        if not tmp_out.exists():
            raise RuntimeError(f"mri_surf2surf reported success but wrote no {tmp_out.name}")
        target_annot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_out, target_annot)
    logger.info("mri_surf2surf → %s", target_annot)


def _surf2surf_python_fallback(
    subjects_dir: Path,
    subject_id: str,
    hemi: str,
    source_annot: Path,
) -> LabelArray:
    """Nearest-vertex projection from fsaverage to individual (Python).

    This is a fallback when ``mri_surf2surf`` is not available.  It
    loads both surfaces, builds a KD-tree on fsaverage, and assigns
    each individual vertex the label of its nearest fsaverage vertex.
    This is an approximation — ``mri_surf2surf`` is more accurate
    because it uses sphere registration.
    """
    from scipy.spatial import cKDTree

    from spectralbrain.io.loaders import (
        load_freesurfer_annot,
        load_freesurfer_surface,
    )

    # Load fsaverage sphere (registration surface)
    fsa_sphere = subjects_dir / "fsaverage" / "surf" / f"{hemi}.sphere.reg"
    if not fsa_sphere.exists():
        fs_home = _find_freesurfer_home()
        if fs_home:
            fsa_sphere = fs_home / "subjects" / "fsaverage" / "surf" / f"{hemi}.sphere.reg"
    if not fsa_sphere.exists():
        raise FileNotFoundError(
            f"fsaverage sphere.reg not found.  Looked in {fsa_sphere}.  "
            f"Set FREESURFER_HOME or ensure fsaverage is in subjects_dir."
        )

    # Load individual sphere
    ind_sphere = subjects_dir / subject_id / "surf" / f"{hemi}.sphere.reg"
    if not ind_sphere.exists():
        raise FileNotFoundError(
            f"Individual sphere.reg not found: {ind_sphere}.  Run 'recon-all' first."
        )

    # Load coordinates
    fsa_coords, _ = load_freesurfer_surface(fsa_sphere)
    ind_coords, _ = load_freesurfer_surface(ind_sphere)

    # Load fsaverage labels
    fsa_labels, _, _ = load_freesurfer_annot(source_annot)

    # Nearest-vertex mapping on the spherical registration surface
    tree = cKDTree(fsa_coords)
    _, idx = tree.query(ind_coords, k=1)

    projected_labels = fsa_labels[idx]

    logger.warning(
        "Using Python nearest-vertex fallback for surf2surf projection.  "
        "Results are approximate.  Install FreeSurfer for mri_surf2surf."
    )
    return projected_labels


def _parcellate_fsaverage_annot(
    subjects_dir: Path,
    subject_id: str,
    atlas: AtlasSpec,
    hemi: str,
    surface: str,
    *,
    cache_annot: bool = True,
) -> ParcellationResult:
    """Project an atlas from fsaverage to individual and parcellate.

    With ``cache_annot=True`` (default) the projected annot is written to
    ``<subject>/label/`` so later calls reuse it; ``False`` keeps the
    subject directory untouched (the projection is done in a temp dir).
    """
    from spectralbrain.io.loaders import (
        apply_parcellation,
        load_freesurfer_annot,
        load_freesurfer_surface,
    )

    # 1. Find the atlas .annot on fsaverage
    fsa_annot = _find_fsaverage_annot(atlas, hemi, subjects_dir)
    if fsa_annot is None:
        raise FileNotFoundError(
            f"Atlas '{atlas.name}' annot file not found on fsaverage.  "
            f"Expected: {atlas.annot_pattern.format(hemi=hemi)}\n"
            f"Download Schaefer annots from: "
            f"https://github.com/ThomasYeoLab/CBIG → stable_projects/"
            f"brain_parcellation/Schaefer2018_LocalGlobal"
        )

    # 2. Check if already projected to individual
    annot_file = atlas.annot_pattern.format(hemi=hemi)
    ind_annot = subjects_dir / subject_id / "label" / annot_file

    if ind_annot.exists():
        logger.info("Atlas already on individual: %s", ind_annot)
        labels, _ctab, names = load_freesurfer_annot(ind_annot)
        strategy = "fsaverage_annot (existing individual annot)"
    elif _has_freesurfer_cmd("mri_surf2surf"):
        # 3. Project: prefer mri_surf2surf, fallback to Python
        if cache_annot:
            _surf2surf_freesurfer(subjects_dir, subject_id, hemi, fsa_annot, ind_annot)
            labels, _ctab, names = load_freesurfer_annot(ind_annot)
        else:
            with tempfile.TemporaryDirectory(prefix="sb_annot_") as tmp:
                tmp_annot = Path(tmp) / annot_file
                _surf2surf_freesurfer(subjects_dir, subject_id, hemi, fsa_annot, tmp_annot)
                labels, _ctab, names = load_freesurfer_annot(tmp_annot)
        strategy = "fsaverage_annot (mri_surf2surf)"
    else:
        labels = _surf2surf_python_fallback(
            subjects_dir,
            subject_id,
            hemi,
            fsa_annot,
        )
        # Recover names from fsaverage annot
        _, _ctab, names = load_freesurfer_annot(fsa_annot)
        strategy = "fsaverage_annot (python_nearest_vertex)"

    # 4. Load surface and parcellate
    surf_path = subjects_dir / subject_id / "surf" / f"{hemi}.{surface}"
    if not surf_path.exists():
        raise FileNotFoundError(f"Surface not found: {surf_path}")
    vertices, faces = load_freesurfer_surface(surf_path)
    if labels.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"Projected labels ({labels.shape[0]}) don't match surface vertices "
            f"({vertices.shape[0]}) for {surf_path}."
        )

    parcels = apply_parcellation(
        vertices,
        faces,
        labels,
        ignore_labels=atlas.ignore_labels,
    )

    logger.info(
        "Strategy B (%s): %s → %d parcels",
        strategy,
        atlas.name,
        len(parcels),
    )
    return ParcellationResult(
        atlas=atlas,
        hemi=hemi,
        surface=surface,
        vertices=vertices,
        faces=faces,
        labels=labels,
        label_names=names,
        parcels=parcels,
        strategy_used=strategy,
    )


# ======================================================================
# §5  Strategy C — MNI volume → surface via vol2surf
# ======================================================================


#: Where nilearn atlas images returned as in-memory objects are cached.
ATLAS_CACHE = Path.home() / ".cache" / "spectralbrain" / "atlases"


def _fetch_atlas_volume(atlas: AtlasSpec) -> Path:
    """Download a volumetric (MNI-space) label atlas using nilearn.

    Returns the path of a 3-D integer label NIfTI. Atlases that nilearn does
    not distribute (Brainnetome) raise with instructions to pass
    ``atlas_volume_path=``.
    """
    fetcher_name = atlas.volume_fetcher
    if fetcher_name is None:
        raise ValueError(f"Atlas '{atlas.name}' has no volume_fetcher configured.")

    if fetcher_name == "fetch_atlas_brainnetome":
        raise ValueError(
            "The Brainnetome atlas is not distributed by nilearn.  Download "
            "BN_Atlas_246_1mm.nii.gz from https://atlas.brainnetome.org and pass "
            "parcellate(..., atlas_volume_path=...)."
        )

    try:
        import nilearn.datasets as datasets
    except ImportError:
        raise ImportError(
            "nilearn is required for volumetric atlas fetching.  Install with: pip install nilearn"
        )

    # Deterministic (max-probability) label maps only — probabilistic 4-D
    # maps cannot be projected as labels.
    fetcher_map = {
        "fetch_atlas_aal": lambda: datasets.fetch_atlas_aal(version="3v2"),  # AAL3
        "fetch_atlas_juelich": lambda: datasets.fetch_atlas_juelich(
            atlas_name="maxprob-thr25-2mm",
        ),
        "fetch_atlas_harvard_oxford": lambda: datasets.fetch_atlas_harvard_oxford(
            atlas_name="cort-maxprob-thr25-2mm",
        ),
    }
    if fetcher_name not in fetcher_map:
        raise ValueError(f"Unknown fetcher '{fetcher_name}'.  Provide atlas_volume_path= manually.")

    try:
        atlas_data = fetcher_map[fetcher_name]()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch atlas '{atlas.name}' via nilearn: {exc}.  "
            f"You can manually provide the atlas volume path via "
            f"parcellate(..., atlas_volume_path=...)"
        ) from exc

    maps = getattr(atlas_data, "maps", None)
    if maps is None and isinstance(atlas_data, dict):
        maps = atlas_data.get("maps")
    if maps is None:
        raise ValueError(f"Unexpected atlas data structure from {fetcher_name}")
    if isinstance(maps, (str, os.PathLike)):
        return Path(maps)
    # Recent nilearn versions return an in-memory image: persist it.
    import nibabel as nib

    ATLAS_CACHE.mkdir(parents=True, exist_ok=True)
    out = ATLAS_CACHE / f"{fetcher_name}.nii.gz"
    nib.save(maps, str(out))
    return out


def _tkr_to_scanner(subjects_dir: Path, subject_id: str) -> np.ndarray:
    """4×4 matrix mapping FreeSurfer surface (tkr-RAS) to scanner RAS.

    ``scanner = Vox2RAS · inv(Vox2RAS_tkr) · tkr`` — i.e. it adds the
    ``c_ras`` offset that FreeSurfer surface files omit.
    """
    import nibabel as nib

    mri = subjects_dir / subject_id / "mri"
    for name in ("orig.mgz", "T1.mgz", "brainmask.mgz", "norm.mgz", "aseg.mgz", "rawavg.mgz"):
        f = mri / name
        if f.exists():
            hdr = nib.load(str(f)).header
            return np.asarray(hdr.get_vox2ras(), float) @ np.linalg.inv(
                np.asarray(hdr.get_vox2ras_tkr(), float)
            )
    raise FileNotFoundError(
        f"No conformed volume (orig.mgz, T1.mgz, …) in {mri}; it is needed to convert "
        "surface tkr-RAS coordinates to scanner RAS (c_ras offset)."
    )


def _python_vol2surf(
    volume_path: Path,
    subjects_dir: Path,
    subject_id: str,
    hemi: str,
) -> LabelArray:
    """Nearest-voxel label sampling at mid-cortical depth (pure Python).

    Mirrors ``mri_vol2surf --regheader --interp nearest --projfrac 0.5``:
    each vertex takes the label of the single atlas voxel containing its
    mid-thickness point (midpoint of white and pial). Surface coordinates
    are converted from tkr-RAS to scanner RAS (``c_ras``) before applying
    the atlas affine, and labels are never averaged, so no non-existent
    label IDs can appear. The volume must already be in the subject's
    scanner space.
    """
    import nibabel as nib

    from spectralbrain.io.loaders import load_freesurfer_surface

    surf_dir = subjects_dir / subject_id / "surf"
    white, _ = load_freesurfer_surface(surf_dir / f"{hemi}.white")
    pial_path = surf_dir / f"{hemi}.pial"
    if pial_path.exists():
        pial, _ = load_freesurfer_surface(pial_path)
        tkr = 0.5 * (white + pial) if pial.shape == white.shape else white
    else:
        logger.warning("%s missing; sampling labels on the white surface.", pial_path)
        tkr = white

    M = _tkr_to_scanner(subjects_dir, subject_id)
    xyz = (M @ np.c_[tkr, np.ones(len(tkr))].T).T[:, :3]

    img = nib.load(str(volume_path))
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{volume_path} is not a 3-D label volume (shape {data.shape}).")
    if np.issubdtype(data.dtype, np.floating):
        rounded = np.rint(data)
        if not np.allclose(data, rounded, atol=1e-3):
            raise ValueError(f"{volume_path} contains non-integer values; not a label map.")
        data = rounded
    data = data.astype(np.int64)

    ijk = (np.linalg.inv(np.asarray(img.affine, float)) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    ijk = np.rint(ijk).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(data.shape)), axis=1)
    labels = np.zeros(len(xyz), dtype=np.int64)
    labels[inside] = data[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
    if not inside.all():
        logger.warning("%d vertices fall outside the atlas volume (label 0).", int((~inside).sum()))
    return labels


def _check_atlas_space(atlas_space: str, atlas_reg: PathLike | None) -> None:
    """Refuse to project an MNI atlas without a registration to the subject."""
    if atlas_space not in ("mni", "native"):
        raise ValueError(f"atlas_space must be 'mni' or 'native', got {atlas_space!r}")
    if atlas_space == "mni" and atlas_reg is None:
        raise ValueError(
            "The atlas volume is in MNI space but no registration to the subject was "
            "given.  Pass atlas_reg= (an LTA/register.dat mapping the atlas to the "
            "subject, e.g. from `mri_coreg` / `mri_vol2vol --mni152reg`), or resample "
            "the atlas into the subject's native space and pass atlas_space='native'."
        )


def _vol2surf_project(
    volume_path: Path,
    subjects_dir: Path,
    subject_id: str,
    hemi: str,
    surface: str = "white",
    *,
    atlas_reg: PathLike | None = None,
    atlas_space: Literal["mni", "native"] = "mni",
) -> LabelArray:
    """Project a volumetric label atlas onto a FreeSurfer surface.

    Labels are sampled at mid-cortical depth (``projfrac 0.5`` from the white
    surface) with nearest-neighbour lookup; the per-vertex labels apply to any
    surface of the subject (white/pial/inflated/sphere share vertex indices).

    Parameters
    ----------
    atlas_reg : PathLike, optional
        FreeSurfer registration (``.lta`` / ``register.dat``) mapping the
        atlas volume onto the subject (``mri_vol2surf --reg``). **Required**
        when ``atlas_space="mni"``: an MNI atlas is not in the subject's
        space, and assuming so (``--regheader``) silently mislabels cortex.
    atlas_space : {"mni", "native"}
        ``"native"`` means the volume is already resampled into the subject's
        scanner space (header registration is then correct).

    Uses ``mri_vol2surf`` when available, else a pure-Python nearest-voxel
    sampler (:func:`_python_vol2surf`, native-space volumes only).
    """
    return _vol2surf_with_method(
        volume_path, subjects_dir, subject_id, hemi, atlas_reg=atlas_reg, atlas_space=atlas_space
    )[0]


def _vol2surf_with_method(
    volume_path: Path,
    subjects_dir: Path,
    subject_id: str,
    hemi: str,
    *,
    atlas_reg: PathLike | None,
    atlas_space: str,
) -> tuple[LabelArray, str]:
    """:func:`_vol2surf_project` that also reports which backend ran."""
    _check_atlas_space(atlas_space, atlas_reg)

    if _has_freesurfer_cmd("mri_vol2surf"):
        with tempfile.TemporaryDirectory(prefix="sb_vol2surf_") as tmp:
            tmp_out = str(Path(tmp) / "labels.mgz")
            reg_args = ["--reg", str(atlas_reg)] if atlas_reg is not None else [
                "--regheader", subject_id
            ]
            cmd = [
                "mri_vol2surf",
                "--mov",
                str(volume_path),
                *reg_args,
                "--hemi",
                hemi,
                "--interp",
                "nearest",  # label map → nearest-neighbour
                "--projfrac",
                "0.5",
                "--surf",
                "white",  # projfrac is defined from white along the thickness
                "--sd",
                str(subjects_dir),
                "--o",
                tmp_out,
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                import nibabel as nib

                img = nib.load(tmp_out)
                labels = np.rint(np.asarray(img.get_fdata()).squeeze()).astype(np.int64)
                logger.info("mri_vol2surf → %d unique labels", len(np.unique(labels)))
                return labels, "mri_vol2surf"
            logger.warning(
                "mri_vol2surf failed (exit %d): %s",
                result.returncode,
                result.stderr[-300:],
            )

    if atlas_reg is not None:
        raise RuntimeError(
            "mri_vol2surf is unavailable (or failed) and a FreeSurfer registration "
            "file cannot be applied in Python.  Install FreeSurfer, or resample the "
            "atlas into subject space and pass atlas_space='native'."
        )
    logger.info("Using Python nearest-voxel vol2surf (c_ras-corrected).")
    return (
        _python_vol2surf(Path(volume_path), subjects_dir, subject_id, hemi),
        "python_nearest_voxel",
    )


def _parcellate_mni_volume(
    subjects_dir: Path,
    subject_id: str,
    atlas: AtlasSpec,
    hemi: str,
    surface: str,
    atlas_volume_path: Path | None = None,
    *,
    atlas_reg: PathLike | None = None,
    atlas_space: Literal["mni", "native"] = "mni",
) -> ParcellationResult:
    """Project a volumetric atlas onto a surface and parcellate."""
    from spectralbrain.io.loaders import (
        apply_parcellation,
        load_freesurfer_surface,
    )

    # 0. Validate the space/registration before any download.
    _check_atlas_space(atlas_space, atlas_reg)

    # 1. Get the atlas volume
    if atlas_volume_path is not None:
        vol_path = Path(atlas_volume_path)
        if not vol_path.exists():
            raise FileNotFoundError(f"Atlas volume not found: {vol_path}")
    else:
        vol_path = _fetch_atlas_volume(atlas)

    # 2. Project volume → surface labels
    labels, method = _vol2surf_with_method(
        vol_path,
        subjects_dir,
        subject_id,
        hemi,
        atlas_reg=atlas_reg,
        atlas_space=atlas_space,
    )

    # 3. Load surface and parcellate
    surf_path = subjects_dir / subject_id / "surf" / f"{hemi}.{surface}"
    vertices, faces = load_freesurfer_surface(surf_path)

    # Ensure label array matches surface
    if labels.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"Projected labels ({labels.shape[0]}) don't match "
            f"surface vertices ({vertices.shape[0]}).  "
            f"The vol2surf projection may have failed."
        )

    parcels = apply_parcellation(
        vertices,
        faces,
        labels,
        ignore_labels=atlas.ignore_labels,
    )

    # Volumetric atlases don't come with FreeSurfer-style .annot name tables;
    # names are indexed by label value (``label_names[k]`` names label ``k``),
    # matching the annot-based strategies.
    max_id = int(labels.max()) if labels.size else 0
    label_names = [f"region_{i}" for i in range(max(max_id, 0) + 1)]

    strategy = f"mni_volume ({method})"
    logger.info(
        "Strategy C (%s): %s → %d parcels",
        strategy,
        atlas.name,
        len(parcels),
    )
    return ParcellationResult(
        atlas=atlas,
        hemi=hemi,
        surface=surface,
        vertices=vertices,
        faces=faces,
        labels=labels,
        label_names=label_names,
        parcels=parcels,
        strategy_used=strategy,
    )


# ======================================================================
# §6  Main entry point — parcellate()
# ======================================================================


def _locate_fastsurfer_subject(fs_out: Path, t1: Path, hemi: str, surface: str) -> str:
    """Find the subject directory FastSurfer created and check it has surfaces."""
    stem = t1.name
    for ext in (".nii.gz", ".nii", ".mgz"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    guesses = [stem, stem.replace("_T1w", "")]
    subj_dirs = sorted(d for d in fs_out.iterdir() if d.is_dir() and (d / "mri").is_dir())
    names = [d.name for d in subj_dirs]
    subject_id = next((g for g in guesses if g in names), None)
    if subject_id is None:
        if len(subj_dirs) != 1:
            raise RuntimeError(
                f"Cannot identify the FastSurfer subject in {fs_out} (found {names or 'none'})."
                "  Run FastSurfer with an explicit --sid and call "
                "parcellate(subjects_dir=..., subject_id=...)."
            )
        subject_id = names[0]
    surf = fs_out / subject_id / "surf" / f"{hemi}.{surface}"
    if not surf.exists():
        raise RuntimeError(
            f"FastSurfer produced no surface {surf}: the container runs in --seg_only "
            "mode, which creates segmentations but no cortical surfaces.  Run the full "
            "FastSurfer surface pipeline (or recon-all) and call "
            "parcellate(subjects_dir=..., subject_id=...)."
        )
    return subject_id


def parcellate(
    *,
    subjects_dir: PathLike | None = None,
    subject_id: str | None = None,
    t1_path: PathLike | None = None,
    atlas: str = "schaefer_200",
    hemi: str = "lh",
    surface: str = "white",
    atlas_volume_path: PathLike | None = None,
    gpu: bool | None = None,
    atlas_reg: PathLike | None = None,
    atlas_space: Literal["mni", "native"] = "mni",
    cache_annot: bool = True,
) -> ParcellationResult:
    """Parcellate a cortical hemisphere into atlas-defined regions.

    This is the high-level entry point that auto-detects the source
    format and applies the appropriate projection strategy.

    Parameters
    ----------
    subjects_dir : PathLike, optional
        Path to the FreeSurfer SUBJECTS_DIR.  Falls back to the
        ``$SUBJECTS_DIR`` environment variable.
    subject_id : str, optional
        FreeSurfer subject ID (e.g. ``'sub-01'``).  Required when
        using FreeSurfer surfaces.
    t1_path : PathLike, optional
        Path to a raw T1-weighted NIfTI file.  If provided (and no
        ``subjects_dir``), the preprocessing pipeline is run first:
        skull-strip → FastSurfer/recon-all → then parcellate.
    atlas : str
        Target atlas short name.  See :func:`list_atlases` for all
        supported names.  Common choices:

        - ``'schaefer_200'`` — Schaefer 200 parcels, 7 networks
        - ``'schaefer_400'`` — Schaefer 400 parcels, 7 networks
        - ``'dkt'`` — Desikan-Killiany-Tourville (FreeSurfer native)
        - ``'destrieux'`` — Destrieux 2009 (FreeSurfer native)
        - ``'glasser'`` — HCP-MMP 1.0 (360 parcels)
        - ``'brainnetome'`` — Brainnetome 246 regions
        - ``'aal3'`` — AAL3 atlas
        - ``'harvard_oxford'`` — Harvard-Oxford cortical

    hemi : {'lh', 'rh'}
        Hemisphere.
    surface : str
        FreeSurfer surface to use: ``'white'``, ``'pial'``,
        ``'inflated'``, ``'sphere'``, etc.
    atlas_volume_path : PathLike, optional
        Explicit path to a volumetric atlas NIfTI file (overrides
        the automatic fetcher for MNI-volume atlases).
    gpu : bool or None
        GPU toggle for preprocessing steps.
    atlas_reg : PathLike, optional
        Volumetric atlases only: FreeSurfer registration (``.lta`` /
        ``register.dat``) mapping the atlas volume onto the subject.
        Required for MNI-space atlases (the default ``atlas_space``); without
        it the projection would silently mislabel cortex.
    atlas_space : {"mni", "native"}
        Volumetric atlases only: space of the atlas volume. ``"native"``
        means it is already resampled into the subject's scanner space.
    cache_annot : bool
        fsaverage atlases: write the projected annot into
        ``<subject>/label/`` for reuse (default). ``False`` leaves the subject
        directory untouched.

    Returns
    -------
    ParcellationResult
        Dataclass with ``.vertices``, ``.faces``, ``.labels``,
        ``.parcels`` (dict of sub-meshes), and metadata.

    Raises
    ------
    ValueError
        If the atlas is not recognised, or if neither ``subjects_dir``
        nor ``t1_path`` is provided.
    FileNotFoundError
        If required files (surfaces, annotations) are missing.
    EnvironmentError
        If FreeSurfer or containers are needed but not available.

    Examples
    --------
    **From FreeSurfer subjects_dir** (most common):

    >>> result = parcellate(
    ...     subjects_dir="/data/freesurfer",
    ...     subject_id="sub-01",
    ...     atlas="schaefer_200",
    ...     hemi="lh",
    ... )
    >>> result.n_parcels
    100
    >>> verts, faces = result.get_parcel(1)

    **From raw T1w** (runs preprocessing first):

    >>> result = parcellate(
    ...     t1_path="/data/sub-01_T1w.nii.gz",
    ...     atlas="dkt",
    ...     hemi="lh",
    ... )

    **With a custom volumetric atlas**:

    >>> result = parcellate(
    ...     subjects_dir="/data/freesurfer",
    ...     subject_id="sub-01",
    ...     atlas="brainnetome",
    ...     atlas_volume_path="/atlases/BN_Atlas_246_2mm.nii.gz",
    ...     atlas_reg="/data/freesurfer/sub-01/mri/transforms/mni_to_sub.lta",
    ... )
    """
    # ── Validate inputs ──
    if hemi not in ("lh", "rh"):
        raise ValueError(f"hemi must be 'lh' or 'rh', got '{hemi}'")

    atlas_spec = _resolve_atlas(atlas)

    # ── Determine the source ──
    if subjects_dir is not None or subject_id is not None:
        # FreeSurfer path
        sd = _get_subjects_dir(subjects_dir)
        if subject_id is None:
            raise ValueError("subject_id is required when using subjects_dir.")
        # Verify subject exists
        subj_dir = sd / subject_id
        if not subj_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

    elif t1_path is not None:
        # Raw T1 path — run preprocessing to generate FS outputs
        t1 = Path(t1_path)
        if not t1.exists():
            raise FileNotFoundError(f"T1w file not found: {t1}")

        logger.info("T1w source provided — running preprocessing pipeline.")
        from spectralbrain.io.preprocess import run_fastsurfer

        # Run FastSurfer to generate FS-compatible outputs
        output_dir = t1.parent / "freesurfer_output"
        fs_out = Path(run_fastsurfer(t1, output_dir=output_dir, gpu=gpu))
        sd = fs_out
        subject_id = _locate_fastsurfer_subject(fs_out, t1, hemi, surface)

        logger.info(
            "Preprocessing complete.  subjects_dir=%s, subject_id=%s",
            sd,
            subject_id,
        )
    else:
        raise ValueError(
            "Provide either (subjects_dir + subject_id) for FreeSurfer "
            "data, or t1_path for raw T1w data."
        )

    # ── Dispatch to the appropriate strategy ──
    if atlas_spec.strategy == "native_annot":
        return _parcellate_native_annot(
            sd,
            subject_id,
            atlas_spec,
            hemi,
            surface,
        )
    elif atlas_spec.strategy == "fsaverage_annot":
        return _parcellate_fsaverage_annot(
            sd,
            subject_id,
            atlas_spec,
            hemi,
            surface,
            cache_annot=cache_annot,
        )
    elif atlas_spec.strategy == "mni_volume":
        return _parcellate_mni_volume(
            sd,
            subject_id,
            atlas_spec,
            hemi,
            surface,
            atlas_volume_path=(Path(atlas_volume_path) if atlas_volume_path else None),
            atlas_reg=atlas_reg,
            atlas_space=atlas_space,
        )
    else:
        raise ValueError(
            f"Unknown strategy '{atlas_spec.strategy}' for atlas "
            f"'{atlas_spec.name}'.  This is a bug in the atlas registry."
        )


# ======================================================================
# §7  Batch parcellation
# ======================================================================


def parcellate_batch(
    subjects_dir: PathLike,
    subject_ids: list[str],
    atlas: str = "schaefer_200",
    hemi: str = "lh",
    surface: str = "white",
    *,
    atlas_volume_path: PathLike | None = None,
    n_jobs: int = 1,
    atlas_reg: PathLike | dict[str, PathLike] | None = None,
    atlas_space: Literal["mni", "native"] = "mni",
    return_failed: bool = False,
) -> dict[str, ParcellationResult] | tuple[dict[str, ParcellationResult], dict[str, str]]:
    """Parcellate multiple subjects in batch.

    Parameters
    ----------
    subjects_dir : PathLike
        FreeSurfer SUBJECTS_DIR.
    subject_ids : list of str
        Subject IDs to parcellate.
    atlas, hemi, surface
        Passed to :func:`parcellate`.
    atlas_volume_path : PathLike, optional
        For volumetric atlases — shared across all subjects.
    n_jobs : int
        Number of parallel workers (1 = sequential).
    atlas_reg : PathLike or dict of {subject_id: PathLike}, optional
        Volumetric atlases: per-subject (dict) or shared registration file
        (see :func:`parcellate`).
    atlas_space : {"mni", "native"}
        Volumetric atlases: space of the atlas volume.
    return_failed : bool
        If True, also return ``{subject_id: error message}``.

    Returns
    -------
    dict of {subject_id: ParcellationResult}
        Results keyed by subject ID.  Subjects whose data fail to load are
        logged and excluded (not raised); programming errors (``TypeError``
        etc.) are re-raised. With ``return_failed=True`` a
        ``(results, failed)`` tuple is returned.
    """
    from spectralbrain.io.group import PROGRAMMING_ERRORS

    sd = Path(subjects_dir)

    def _one(sid: str) -> tuple[str, ParcellationResult | str]:
        reg = atlas_reg.get(sid) if isinstance(atlas_reg, dict) else atlas_reg
        try:
            result = parcellate(
                subjects_dir=sd,
                subject_id=sid,
                atlas=atlas,
                hemi=hemi,
                surface=surface,
                atlas_volume_path=atlas_volume_path,
                atlas_reg=reg,
                atlas_space=atlas_space,
            )
            logger.info("✓ %s: %d parcels", sid, result.n_parcels)
            return sid, result
        except PROGRAMMING_ERRORS:
            raise
        except Exception as exc:
            logger.error("✗ %s: %s", sid, exc)
            return sid, f"{type(exc).__name__}: {exc}"

    if n_jobs == 1:
        pairs = [_one(sid) for sid in subject_ids]
    else:
        from spectralbrain.backends.cpu import parallel_map

        pairs = parallel_map(_one, subject_ids, n_jobs=n_jobs, description="Parcellating")

    results: dict[str, ParcellationResult] = {
        sid: res for sid, res in pairs if isinstance(res, ParcellationResult)
    }
    failed: dict[str, str] = {sid: res for sid, res in pairs if isinstance(res, str)}
    if failed:
        logger.warning("Parcellation failed for %d subject(s): %s", len(failed), ", ".join(failed))

    logger.info(
        "Batch parcellation: %d/%d subjects succeeded.",
        len(results),
        len(subject_ids),
    )
    if return_failed:
        return results, failed
    return results


# ======================================================================
# __all__
# ======================================================================

__all__ = [
    "ATLAS_REGISTRY",
    # Atlas registry
    "AtlasSpec",
    # Result container
    "ParcellationResult",
    "list_atlases",
    # Main entry points
    "parcellate",
    "parcellate_batch",
]
