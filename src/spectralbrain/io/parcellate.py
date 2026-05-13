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
   projected onto the individual's surface via ``mri_vol2surf`` or
   ``nilearn.surface.vol_to_surf``.

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
... )
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from spectralbrain.runtime import AtlasScheme, PathLike, get_logger

logger = get_logger(__name__)

# Type aliases reused from loaders.py
Vertices = np.ndarray   # (N, 3)
Faces = np.ndarray       # (F, 3)
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
    annot_pattern: Optional[str] = None
    volume_fetcher: Optional[str] = None
    n_parcels: Optional[int] = None
    ignore_labels: List[int] = field(default_factory=lambda: [0])


# Registry of supported atlases.
# Key: canonical short name (lowercased, used in the public API).
ATLAS_REGISTRY: Dict[str, AtlasSpec] = {
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


def list_atlases() -> List[str]:
    """Return the names of all supported atlases."""
    return sorted(ATLAS_REGISTRY.keys())


def _resolve_atlas(name: str) -> AtlasSpec:
    """Look up an atlas by short name (case-insensitive)."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in ATLAS_REGISTRY:
        available = ", ".join(sorted(ATLAS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown atlas '{name}'.  Available: {available}"
        )
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
    label_names: List[str]
    parcels: Dict[int, Tuple[Vertices, Faces]]
    strategy_used: str

    @property
    def n_parcels(self) -> int:
        """Number of non-empty parcels."""
        return len(self.parcels)

    @property
    def parcel_ids(self) -> List[int]:
        """Sorted list of parcel IDs."""
        return sorted(self.parcels.keys())

    def get_parcel(self, label_id: int) -> Tuple[Vertices, Faces]:
        """Return (vertices, faces) for a specific parcel."""
        if label_id not in self.parcels:
            raise KeyError(
                f"Parcel {label_id} not found.  "
                f"Available: {self.parcel_ids}"
            )
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

def _get_subjects_dir(subjects_dir: Optional[PathLike] = None) -> Path:
    """Resolve SUBJECTS_DIR from arg or environment."""
    if subjects_dir is not None:
        return Path(subjects_dir)
    env = os.environ.get("SUBJECTS_DIR")
    if env:
        return Path(env)
    raise EnvironmentError(
        "SUBJECTS_DIR not set and no subjects_dir argument provided.  "
        "Either pass subjects_dir= or set the SUBJECTS_DIR environment "
        "variable."
    )


def _find_freesurfer_home() -> Optional[Path]:
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
            capture_output=True, timeout=10,
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
    labels, ctab, names = load_freesurfer_annot(annot_path)

    # Parcellate
    parcels = apply_parcellation(
        vertices, faces, labels,
        ignore_labels=atlas.ignore_labels,
    )

    logger.info(
        "Strategy A (native_annot): %s → %d parcels",
        atlas.name, len(parcels),
    )
    return ParcellationResult(
        atlas=atlas, hemi=hemi, surface=surface,
        vertices=vertices, faces=faces,
        labels=labels, label_names=names,
        parcels=parcels, strategy_used="native_annot",
    )


# ======================================================================
# §4  Strategy B — fsaverage annot → individual via surf2surf
# ======================================================================

def _find_fsaverage_annot(
    atlas: AtlasSpec,
    hemi: str,
    subjects_dir: Path,
) -> Optional[Path]:
    """Search for the atlas .annot on fsaverage in several locations."""
    annot_file = atlas.annot_pattern.format(hemi=hemi)

    search_paths = [
        # 1. fsaverage in subjects_dir (standard after recon-all)
        subjects_dir / "fsaverage" / "label" / annot_file,
        # 2. FREESURFER_HOME/subjects/fsaverage
    ]
    fs_home = _find_freesurfer_home()
    if fs_home:
        search_paths.append(
            fs_home / "subjects" / "fsaverage" / "label" / annot_file
        )

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
    """Project an annot from fsaverage to individual using mri_surf2surf."""
    cmd = [
        "mri_surf2surf",
        "--srcsubject", "fsaverage",
        "--trgsubject", subject_id,
        "--hemi", hemi,
        "--sval-annot", str(source_annot),
        "--tval", str(target_annot),
        "--sd", str(subjects_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"mri_surf2surf failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )
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
            fsa_sphere = (
                fs_home / "subjects" / "fsaverage" / "surf"
                / f"{hemi}.sphere.reg"
            )
    if not fsa_sphere.exists():
        raise FileNotFoundError(
            f"fsaverage sphere.reg not found.  Looked in {fsa_sphere}.  "
            f"Set FREESURFER_HOME or ensure fsaverage is in subjects_dir."
        )

    # Load individual sphere
    ind_sphere = (
        subjects_dir / subject_id / "surf" / f"{hemi}.sphere.reg"
    )
    if not ind_sphere.exists():
        raise FileNotFoundError(
            f"Individual sphere.reg not found: {ind_sphere}.  "
            f"Run 'recon-all' first."
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
) -> ParcellationResult:
    """Project an atlas from fsaverage to individual and parcellate."""
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
        labels, ctab, names = load_freesurfer_annot(ind_annot)
    else:
        # 3. Project: prefer mri_surf2surf, fallback to Python
        if _has_freesurfer_cmd("mri_surf2surf"):
            ind_annot.parent.mkdir(parents=True, exist_ok=True)
            _surf2surf_freesurfer(
                subjects_dir, subject_id, hemi, fsa_annot, ind_annot,
            )
            labels, ctab, names = load_freesurfer_annot(ind_annot)
        else:
            labels = _surf2surf_python_fallback(
                subjects_dir, subject_id, hemi, fsa_annot,
            )
            # Recover names from fsaverage annot
            _, ctab, names = load_freesurfer_annot(fsa_annot)

    # 4. Load surface and parcellate
    surf_path = subjects_dir / subject_id / "surf" / f"{hemi}.{surface}"
    vertices, faces = load_freesurfer_surface(surf_path)

    parcels = apply_parcellation(
        vertices, faces, labels,
        ignore_labels=atlas.ignore_labels,
    )

    strategy = (
        "fsaverage_annot (mri_surf2surf)"
        if _has_freesurfer_cmd("mri_surf2surf")
        else "fsaverage_annot (python_nearest_vertex)"
    )
    logger.info(
        "Strategy B (%s): %s → %d parcels", strategy, atlas.name, len(parcels),
    )
    return ParcellationResult(
        atlas=atlas, hemi=hemi, surface=surface,
        vertices=vertices, faces=faces,
        labels=labels, label_names=names,
        parcels=parcels, strategy_used=strategy,
    )


# ======================================================================
# §5  Strategy C — MNI volume → surface via vol2surf
# ======================================================================

def _fetch_atlas_volume(atlas: AtlasSpec) -> Path:
    """Download a volumetric atlas using nilearn and return its path."""
    try:
        import nilearn.datasets as datasets
    except ImportError:
        raise ImportError(
            "nilearn is required for volumetric atlas fetching.  "
            "Install with: pip install nilearn"
        )

    fetcher_name = atlas.volume_fetcher
    if fetcher_name is None:
        raise ValueError(
            f"Atlas '{atlas.name}' has no volume_fetcher configured."
        )

    # Map fetcher names to nilearn calls
    fetcher_map = {
        "fetch_atlas_brainnetome": lambda: datasets.fetch_atlas_surf_destrieux(
            # Brainnetome is not directly in nilearn; fall back to a
            # manual download path.  This raises a clear error.
        ),
        "fetch_atlas_aal": lambda: datasets.fetch_atlas_aal(version="SPM12"),
        "fetch_atlas_juelich": lambda: datasets.fetch_atlas_juelich(
            atlas_name="prob-2mm",
        ),
        "fetch_atlas_harvard_oxford": lambda: datasets.fetch_atlas_harvard_oxford(
            atlas_name="cort-maxprob-thr25-2mm",
        ),
    }

    if fetcher_name in fetcher_map:
        try:
            atlas_data = fetcher_map[fetcher_name]()
            # nilearn fetchers return dicts or Bunch objects with a 'maps' key
            if hasattr(atlas_data, "maps"):
                return Path(atlas_data.maps)
            elif isinstance(atlas_data, dict) and "maps" in atlas_data:
                return Path(atlas_data["maps"])
            else:
                raise ValueError(
                    f"Unexpected atlas data structure from {fetcher_name}"
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch atlas '{atlas.name}' via nilearn: {exc}.  "
                f"You can manually provide the atlas volume path via "
                f"parcellate(..., atlas_volume_path=...)"
            ) from exc
    else:
        raise ValueError(
            f"Unknown fetcher '{fetcher_name}'.  "
            f"Provide atlas_volume_path= manually."
        )


def _vol2surf_project(
    volume_path: Path,
    subjects_dir: Path,
    subject_id: str,
    hemi: str,
    surface: str = "white",
) -> LabelArray:
    """Project a volumetric atlas onto a FreeSurfer surface.

    Tries ``mri_vol2surf`` first, falls back to
    ``nilearn.surface.vol_to_surf``.
    """
    from spectralbrain.io.loaders import load_freesurfer_surface

    surf_path = subjects_dir / subject_id / "surf" / f"{hemi}.{surface}"
    vertices, _ = load_freesurfer_surface(surf_path)

    # Try FreeSurfer mri_vol2surf
    if _has_freesurfer_cmd("mri_vol2surf"):
        with tempfile.NamedTemporaryFile(suffix=".mgz", delete=False) as tmp:
            tmp_out = tmp.name

        cmd = [
            "mri_vol2surf",
            "--mov", str(volume_path),
            "--regheader", subject_id,
            "--hemi", hemi,
            "--interp", "nearest",     # label map → nearest-neighbour
            "--projfrac", "0.5",
            "--surf", surface,
            "--sd", str(subjects_dir),
            "--o", tmp_out,
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            import nibabel as nib
            img = nib.load(tmp_out)
            labels = np.asarray(img.get_fdata()).squeeze().astype(int)
            os.unlink(tmp_out)
            logger.info("mri_vol2surf → %d unique labels", len(np.unique(labels)))
            return labels
        else:
            logger.warning(
                "mri_vol2surf failed (exit %d), trying nilearn fallback.",
                result.returncode,
            )
            os.unlink(tmp_out)

    # Fallback: nilearn.surface.vol_to_surf
    try:
        from nilearn.surface import vol_to_surf
    except ImportError:
        raise ImportError(
            "nilearn is required for Python vol2surf fallback.  "
            "Install with: pip install nilearn"
        )

    logger.info("Using nilearn vol_to_surf for volume → surface projection.")
    projected = vol_to_surf(
        str(volume_path),
        surf_mesh=str(surf_path),
        interpolation="nearest",
        radius=3.0,           # search radius in mm
    )
    labels = np.round(projected).astype(int)
    return labels


def _parcellate_mni_volume(
    subjects_dir: Path,
    subject_id: str,
    atlas: AtlasSpec,
    hemi: str,
    surface: str,
    atlas_volume_path: Optional[Path] = None,
) -> ParcellationResult:
    """Project a volumetric atlas onto a surface and parcellate."""
    from spectralbrain.io.loaders import (
        apply_parcellation,
        load_freesurfer_surface,
    )

    # 1. Get the atlas volume
    if atlas_volume_path is not None:
        vol_path = Path(atlas_volume_path)
        if not vol_path.exists():
            raise FileNotFoundError(f"Atlas volume not found: {vol_path}")
    else:
        vol_path = _fetch_atlas_volume(atlas)

    # 2. Project volume → surface labels
    labels = _vol2surf_project(
        vol_path, subjects_dir, subject_id, hemi, surface,
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
        vertices, faces, labels,
        ignore_labels=atlas.ignore_labels,
    )

    # Build label names from unique IDs (volumetric atlases don't
    # come with FreeSurfer-style .annot name tables).
    unique_ids = sorted(set(np.unique(labels).tolist()) - set(atlas.ignore_labels))
    label_names = [f"region_{i}" for i in unique_ids]

    strategy = (
        "mni_volume (mri_vol2surf)"
        if _has_freesurfer_cmd("mri_vol2surf")
        else "mni_volume (nilearn_vol_to_surf)"
    )
    logger.info(
        "Strategy C (%s): %s → %d parcels", strategy, atlas.name, len(parcels),
    )
    return ParcellationResult(
        atlas=atlas, hemi=hemi, surface=surface,
        vertices=vertices, faces=faces,
        labels=labels, label_names=label_names,
        parcels=parcels, strategy_used=strategy,
    )


# ======================================================================
# §6  Main entry point — parcellate()
# ======================================================================

def parcellate(
    *,
    subjects_dir: Optional[PathLike] = None,
    subject_id: Optional[str] = None,
    t1_path: Optional[PathLike] = None,
    atlas: str = "schaefer_200",
    hemi: str = "lh",
    surface: str = "white",
    atlas_volume_path: Optional[PathLike] = None,
    gpu: Optional[bool] = None,
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
            raise ValueError(
                "subject_id is required when using subjects_dir."
            )
        # Verify subject exists
        subj_dir = sd / subject_id
        if not subj_dir.exists():
            raise FileNotFoundError(
                f"Subject directory not found: {subj_dir}"
            )

    elif t1_path is not None:
        # Raw T1 path — run preprocessing to generate FS outputs
        t1 = Path(t1_path)
        if not t1.exists():
            raise FileNotFoundError(f"T1w file not found: {t1}")

        logger.info(
            "T1w source provided — running preprocessing pipeline."
        )
        from spectralbrain.io.preprocess import run_fastsurfer

        # Run FastSurfer to generate FS-compatible outputs
        output_dir = t1.parent / "freesurfer_output"
        fs_out = run_fastsurfer(t1, output_dir=output_dir, gpu=gpu)

        # After FastSurfer, the subject_id is typically the input stem
        subject_id = t1.stem.replace("_T1w", "").replace(".nii", "")
        sd = fs_out

        logger.info(
            "Preprocessing complete.  subjects_dir=%s, subject_id=%s",
            sd, subject_id,
        )
    else:
        raise ValueError(
            "Provide either (subjects_dir + subject_id) for FreeSurfer "
            "data, or t1_path for raw T1w data."
        )

    # ── Dispatch to the appropriate strategy ──
    if atlas_spec.strategy == "native_annot":
        return _parcellate_native_annot(
            sd, subject_id, atlas_spec, hemi, surface,
        )
    elif atlas_spec.strategy == "fsaverage_annot":
        return _parcellate_fsaverage_annot(
            sd, subject_id, atlas_spec, hemi, surface,
        )
    elif atlas_spec.strategy == "mni_volume":
        return _parcellate_mni_volume(
            sd, subject_id, atlas_spec, hemi, surface,
            atlas_volume_path=(
                Path(atlas_volume_path) if atlas_volume_path else None
            ),
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
    subject_ids: List[str],
    atlas: str = "schaefer_200",
    hemi: str = "lh",
    surface: str = "white",
    *,
    atlas_volume_path: Optional[PathLike] = None,
    n_jobs: int = 1,
) -> Dict[str, ParcellationResult]:
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

    Returns
    -------
    dict of {subject_id: ParcellationResult}
        Results keyed by subject ID.  Failed subjects are logged
        and excluded (not raised).
    """
    sd = Path(subjects_dir)
    results: Dict[str, ParcellationResult] = {}

    for sid in subject_ids:
        try:
            result = parcellate(
                subjects_dir=sd,
                subject_id=sid,
                atlas=atlas,
                hemi=hemi,
                surface=surface,
                atlas_volume_path=atlas_volume_path,
            )
            results[sid] = result
            logger.info("✓ %s: %d parcels", sid, result.n_parcels)
        except Exception as exc:
            logger.error("✗ %s: %s", sid, exc)
            continue

    logger.info(
        "Batch parcellation: %d/%d subjects succeeded.",
        len(results), len(subject_ids),
    )
    return results


# ======================================================================
# __all__
# ======================================================================

__all__ = [
    # Atlas registry
    "AtlasSpec",
    "ATLAS_REGISTRY",
    "list_atlases",
    # Result container
    "ParcellationResult",
    # Main entry points
    "parcellate",
    "parcellate_batch",
]
