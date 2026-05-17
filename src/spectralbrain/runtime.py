"""SpectralBrain runtime infrastructure.

Provides cross-cutting services consumed by all other modules:
versioning, canonical type aliases, structured logging, Rich progress
bars for various workloads, and Singularity/Apptainer container
management for optional DL-based preprocessing.

This module has **no** intra-library imports so it can be imported
first without circular dependencies.

Examples
--------
>>> from spectralbrain.runtime import __version__, get_logger
>>> logger = get_logger("spectralbrain.core")
>>> logger.info("Loaded SpectralBrain %s", __version__)

>>> from spectralbrain.runtime import progress_simple
>>> with progress_simple("Computing HKS", total=20) as update:
...     for i, t in enumerate(t_values):
...         hks[:, i] = _hks_at_t(eigenvalues, eigenvectors, t)
...         update(1)
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

# ──────────────────────────────────────────────────────────────────────
# Third-party (hard deps: numpy, scipy; soft dep: rich)
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


# ======================================================================
# §1  VERSIONING
# ======================================================================

__version__: str = "0.1.0-dev"
"""Semantic version string — single source of truth.

Surfaced as ``spectralbrain.__version__`` via the package
``__init__.py``.  Build tooling reads this at release time.
"""

VERSION_INFO: Tuple[int, int, int] = (0, 1, 0)
"""(major, minor, patch) as a comparable tuple."""


# ======================================================================
# §2  CANONICAL TYPE ALIASES
# ======================================================================
#
# Every array flowing through SpectralBrain should be annotated with
# one of these aliases.  Static analysers (mypy/pyright) and IDE
# autocompletion use them; runtime code treats them as plain ndarrays.
#
# Naming convention: CamelCase nouns, always NDArray-backed.

# ── Geometric primitives ──────────────────────────────────────────────

Vertices = npt.NDArray[np.floating]
"""Vertex coordinates, shape ``(N, 3)``, float64, in mm (RAS)."""

Faces = npt.NDArray[np.intp]
"""Triangle face indices, shape ``(F, 3)``, int64, **0-indexed**."""

Points = npt.NDArray[np.floating]
"""Point-cloud coordinates, shape ``(N, 3)``, float64.

Semantically identical to :pydata:`Vertices` but used when there is
no face connectivity (volumetric segmentation → voxel centroids).
"""

Normals = npt.NDArray[np.floating]
"""Unit normals, shape ``(N, 3)``, float64 — per-vertex or per-point."""

# ── Spectral primitives ──────────────────────────────────────────────

Eigenvalues = npt.NDArray[np.floating]
"""LBO eigenvalue vector, shape ``(k,)``, float64, ascending.

Non-negative; λ₀ ≈ 0 (constant mode).  λ₁ is the Fiedler value.
"""

Eigenvectors = npt.NDArray[np.floating]
"""LBO eigenvectors, shape ``(N, k)``, float64.

Column *i* is eigenfunction φᵢ evaluated at each vertex/point.
Orthonormal w.r.t. mass matrix: Φᵀ M Φ = I.
"""

SparseMatrix = sp.spmatrix
"""Any SciPy sparse matrix (CSR, CSC, COO).

The stiffness (Laplacian) matrix *L* and the mass matrix *M* live
in this format.  Backends convert to their native sparse types.
"""

MassMatrix = sp.spmatrix
"""Lumped or consistent mass matrix, shape ``(N, N)``, sparse."""

# ── Descriptor outputs ────────────────────────────────────────────────

ScalarMap = npt.NDArray[np.floating]
"""Per-vertex scalar, shape ``(N,)`` — e.g. one HKS time-slice,
Casorati curvature, or a Bayesian surprise score.
"""

DescriptorMatrix = npt.NDArray[np.floating]
"""Multi-scale per-vertex descriptor, shape ``(N, T)`` — e.g. HKS
evaluated at *T* time-scales, or WKS at *T* energies.
"""

GlobalDescriptor = npt.NDArray[np.floating]
"""One vector per shape, shape ``(d,)`` — e.g. ShapeDNA, 3D Zernike
moments, or a Fisher-vector aggregation.
"""

DistanceMatrix = npt.NDArray[np.floating]
"""Pairwise distance/similarity matrix, shape ``(R, R)`` — e.g.
ROI-to-ROI WESD in a geometric connectome.
"""

# ── Neuroimaging types ────────────────────────────────────────────────

LabelArray = npt.NDArray[np.integer]
"""Per-vertex / per-voxel integer label, shape ``(N,)`` — atlas ROI
indices (Schaefer, aseg, etc.).
"""

VolumeImage = Any     # nibabel.nifti1.Nifti1Image (lazy import)
"""NIfTI volume — typed as ``Any`` to avoid hard nibabel dep at import."""

SurfaceImage = Any    # nibabel.gifti.GiftiImage (lazy import)
"""GIfTI surface — typed as ``Any`` to avoid hard nibabel dep at import."""

# ── Analysis / connectome ─────────────────────────────────────────────

ConnectomeMatrix = npt.NDArray[np.floating]
"""ROI × ROI spectral similarity/distance, shape ``(R, R)``."""

NetworkMatrix = npt.NDArray[np.floating]
"""Network × Network summary, shape ``(K, K)`` — block-averaged
:pydata:`ConnectomeMatrix` (e.g. 7×7 for Yeo networks).
"""

# ── Generic helpers ───────────────────────────────────────────────────

PathLike = Union[str, os.PathLike]
"""Anything :class:`pathlib.Path` can consume."""

T = TypeVar("T")
"""Generic type variable used in container utilities."""


# ======================================================================
# §2b  SUPPORTED FORMATS, ATLASES, DESCRIPTORS, OBJECTIVES, BACKENDS
# ======================================================================

class GeometryFormat(Enum):
    """Geometry file formats recognised by ``io.loaders``."""

    FREESURFER_SURFACE = auto()   # .white, .pial, .inflated, .sphere
    FREESURFER_ANNOT = auto()     # .annot parcellation overlay
    FREESURFER_MORPH = auto()     # .thickness, .curv, .sulc
    FREESURFER_LABEL = auto()     # .label ROI mask
    GIFTI_SURFACE = auto()        # .surf.gii
    GIFTI_FUNC = auto()           # .func.gii, .shape.gii
    GIFTI_LABEL = auto()          # .label.gii
    NIFTI_VOLUME = auto()         # .nii, .nii.gz
    MGZ_VOLUME = auto()           # .mgz, .mgh (FreeSurfer volume)
    PLY = auto()                  # Stanford .ply
    OBJ = auto()                  # Wavefront .obj
    STL = auto()                  # Stereolithography .stl
    VTK = auto()                  # VTK legacy or XML
    HDF5 = auto()                 # .h5 — SpectralBrain cache
    NUMPY = auto()                # .npz (vertices + faces arrays)


class AtlasScheme(Enum):
    """Brain atlases supported by ``utils.atlas``."""

    SCHAEFER_100 = "schaefer100"
    SCHAEFER_200 = "schaefer200"
    SCHAEFER_400 = "schaefer400"
    SCHAEFER_600 = "schaefer600"
    SCHAEFER_800 = "schaefer800"
    SCHAEFER_1000 = "schaefer1000"
    DKT = "dkt"
    DESTRIEUX = "destrieux"
    ASEG = "aseg"
    THALAMIC_NUCLEI = "thalamic_nuclei"
    AMYGDALA_NUCLEI = "amygdala_nuclei"
    HIPPOCAMPAL_SUBFIELDS = "hippocampal_subfields"
    JULICH_BRAIN = "julich_brain"
    TIAN_S1 = "tian_s1"
    TIAN_S2 = "tian_s2"
    TIAN_S3 = "tian_s3"
    TIAN_S4 = "tian_s4"
    BRAINNETOME = "brainnetome"
    GLASSER_MMP = "glasser_mmp"


class DescriptorType(Enum):
    """Descriptor identifiers — used by ``recommend_descriptor()``
    and the eligibility registry.
    """

    # LBO-based  (spectral/descriptors.py)
    SHAPEDNA = "shapedna"
    HKS = "hks"
    SI_HKS = "si_hks"
    WKS = "wks"
    GPS = "gps"
    BATES_SP = "bates_sp"
    BKS = "bks"
    IBKS = "ibks"

    # Distances  (spectral/distances.py)
    WESD = "wesd"
    BIHARMONIC = "biharmonic"
    COMMUTE_TIME = "commute_time"
    DIFFUSION = "diffusion"

    # Wavelets  (spectral/wavelets.py)
    SGW_MEXICAN_HAT = "sgw_mexican_hat"
    SGW_HEAT = "sgw_heat"

    # Anisotropic  (spectral/anisotropic.py)
    FINSLER_HKS = "finsler_hks"
    ASMWD = "asmwd"

    # Collection-aware  (spectral/collections.py)
    DWKS = "dwks"

    # Curvature-based
    SHAPE_INDEX = "shape_index"
    CASORATI = "casorati"
    WILLMORE_ENERGY = "willmore_energy"

    # Integral / metric
    INTEGRAL_INVARIANT = "integral_invariant"
    ZERNIKE_3D = "zernike_3d"
    SDF = "sdf"
    AGD = "agd"
    ECCENTRICITY = "eccentricity"

    # Topological
    ECT = "ect"
    PHT = "pht"

    # Information-theoretic
    FRACTAL_DIM = "fractal_dim"


# ── Eligibility registry for recommend_descriptor() ───────────────────

DESCRIPTOR_ELIGIBILITY: Dict[str, List[str]] = {
    "group_discrimination": [
        "shapedna", "hks", "wks", "si_hks", "bates_sp",
        "bks", "wesd", "sgw_mexican_hat", "casorati",
        "integral_invariant", "zernike_3d", "ect", "fractal_dim",
    ],
    "lateralization": [
        "shapedna", "hks", "wks", "bates_sp", "gps",
        "bks", "biharmonic", "sgw_mexican_hat", "casorati",
        "integral_invariant", "eccentricity", "ect",
    ],
    "longitudinal_change": [
        "shapedna", "hks", "wks", "bates_sp", "gps",
        "bks", "wesd", "diffusion", "dwks", "casorati",
        "integral_invariant", "ect", "fractal_dim",
    ],
    "subregion_detection": [
        "hks", "wks", "gps", "bks", "sgw_mexican_hat",
        "biharmonic", "dwks", "finsler_hks",
        "shape_index", "casorati", "sdf", "agd", "eccentricity",
    ],
}


class AnalysisObjective(Enum):
    """Objectives for ``recommend_descriptor()``."""

    GROUP_DISCRIMINATION = "group_discrimination"
    LATERALIZATION = "lateralization"
    LONGITUDINAL_CHANGE = "longitudinal_change"
    SUBREGION_DETECTION = "subregion_detection"


class BackendName(Enum):
    """Compute backends."""

    NUMPY = "numpy"
    JAX = "jax"
    CUPY = "cupy"


# ======================================================================
# §3  STRUCTURED LOGGING
# ======================================================================

_CONSOLE: Optional[Console] = Console(stderr=True) if _HAS_RICH else None

_LIB_LOGGER_NAME: str = "spectralbrain"


def get_logger(
    name: str = _LIB_LOGGER_NAME,
    *,
    level: int = logging.INFO,
    rich: bool = True,
) -> logging.Logger:
    """Return a configured logger for a SpectralBrain module.

    Installs a :class:`rich.logging.RichHandler` on first call (if
    Rich is available).  Subsequent calls with the same *name* return
    the existing logger.

    Parameters
    ----------
    name : str
        Logger name — submodules should pass ``__name__``.
    level : int
        Logging level (default ``logging.INFO``).
    rich : bool
        Use Rich formatting when available.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    if rich and _HAS_RICH:
        handler = RichHandler(
            console=_CONSOLE,
            show_path=False,
            show_time=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(handler)
    return logger


def set_log_level(level: Union[int, str]) -> None:
    """Set the log level for the entire library.

    Parameters
    ----------
    level : int or str
        E.g. ``logging.DEBUG`` or ``"WARNING"``.
    """
    numeric = level if isinstance(level, int) else getattr(logging, level.upper())
    logging.getLogger(_LIB_LOGGER_NAME).setLevel(numeric)


# ======================================================================
# §4  RICH PROGRESS BARS
# ======================================================================

def _fallback_update_factory(
    description: str,
    total: Optional[int],
) -> Generator[Callable[[int], None], None, None]:
    """Plain-text fallback when Rich is not installed."""
    done = 0

    def _update(n: int = 1) -> None:
        """Update the progress state."""
        nonlocal done
        done += n
        if total and done % max(1, total // 10) == 0:
            print(f"\r  {description}: {100 * done / total:.0f}%",
                  end="", flush=True)

    yield _update
    print()


@contextmanager
def progress_simple(
    description: str = "Processing",
    total: Optional[int] = None,
) -> Generator[Callable[[int], None], None, None]:
    """Simple progress bar with ETA.

    Parameters
    ----------
    description : str
        Label shown left of the bar.
    total : int or None
        Step count.  ``None`` → indeterminate spinner.

    Yields
    ------
    Callable[[int], None]
        ``update(n)`` advances the bar by *n* steps.

    Examples
    --------
    >>> with progress_simple("Eigensolve", total=n_structures) as tick:
    ...     for s in structures:
    ...         decompose(s)
    ...         tick(1)
    """
    if not _HAS_RICH:
        yield from _fallback_update_factory(description, total)
        return

    cols = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*cols, console=_CONSOLE, transient=True) as prog:
        tid = prog.add_task(description, total=total)

        def _update(n: int = 1) -> None:
            """Update the progress state."""
            prog.update(tid, advance=n)

        yield _update


@contextmanager
def progress_parallel(
    description: str = "Parallel jobs",
    total: Optional[int] = None,
) -> Generator[Callable[[int], None], None, None]:
    """Thread-safe progress bar for ``joblib.Parallel`` callbacks.

    Parameters
    ----------
    description : str
        Label.
    total : int or None
        Total number of jobs.

    Yields
    ------
    Callable[[int], None]
        Thread-safe ``update(n)``.

    Examples
    --------
    >>> from joblib import Parallel, delayed
    >>> with progress_parallel("Subjects", total=228) as tick:
    ...     def _run(s):
    ...         result = process(s); tick(1); return result
    ...     Parallel(n_jobs=8)(delayed(_run)(s) for s in subjects)
    """
    if not _HAS_RICH:
        yield from _fallback_update_factory(description, total)
        return

    cols = [
        SpinnerColumn("dots"),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*cols, console=_CONSOLE, transient=False,
                  refresh_per_second=10) as prog:
        tid = prog.add_task(description, total=total)

        def _update(n: int = 1) -> None:
            """Update the progress state."""
            prog.update(tid, advance=n)

        yield _update


class NestedProgress:
    """Two-level Rich progress for nested loops.

    Parameters
    ----------
    outer_description : str
        Outer-loop label (e.g. ``"Subjects"``).
    outer_total : int
        Number of outer iterations.
    inner_description : str
        Inner-loop label (e.g. ``"Structures"``).
    inner_total : int
        Inner iterations **per** outer step.

    Examples
    --------
    >>> with NestedProgress("Subjects", 228, "ROIs", 44) as np:
    ...     for subj in subjects:
    ...         for roi in rois:
    ...             compute(subj, roi)
    ...             np.advance_inner()
    ...         np.advance_outer()
    """

    def __init__(
        self,
        outer_description: str,
        outer_total: int,
        inner_description: str,
        inner_total: int,
    ) -> None:
        """Initialise the nested progress tracker."""
        self.outer_description = outer_description
        self.outer_total = outer_total
        self.inner_description = inner_description
        self.inner_total = inner_total
        self._progress: Optional[Progress] = None
        self._outer_id: Optional[int] = None
        self._inner_id: Optional[int] = None

    # -- context manager ------------------------------------------------

    def __enter__(self) -> NestedProgress:
        """Enter the context manager and start the progress bar."""
        if not _HAS_RICH:
            return self
        cols = [
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        self._progress = Progress(*cols, console=_CONSOLE, transient=False)
        self._progress.__enter__()
        self._outer_id = self._progress.add_task(
            f"[cyan]{self.outer_description}", total=self.outer_total,
        )
        self._inner_id = self._progress.add_task(
            f"  [green]{self.inner_description}", total=self.inner_total,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        """Exit the context manager and stop the progress bar."""
        if self._progress is not None:
            self._progress.__exit__(*exc)

    # -- public API -----------------------------------------------------

    def advance_inner(self, n: int = 1) -> None:
        """Advance inner bar by *n* steps."""
        if self._progress is not None and self._inner_id is not None:
            self._progress.update(self._inner_id, advance=n)

    def advance_outer(self, n: int = 1) -> None:
        """Advance outer bar by *n* and reset the inner bar."""
        if self._progress is None:
            return
        if self._outer_id is not None:
            self._progress.update(self._outer_id, advance=n)
        if self._inner_id is not None:
            self._progress.reset(self._inner_id)


@contextmanager
def progress_spinner(
    description: str = "Working",
) -> Generator[None, None, None]:
    """Indeterminate spinner for unknown-duration operations.

    Parameters
    ----------
    description : str
        Label next to the spinner.

    Examples
    --------
    >>> with progress_spinner("Downloading container"):
    ...     download_large_file(url, dest)
    """
    if not _HAS_RICH:
        print(f"  {description}…", end="", flush=True)
        yield
        print(" done.")
        return

    cols = [
        SpinnerColumn("dots12"),
        TextColumn("[bold yellow]{task.description}"),
        TimeElapsedColumn(),
    ]
    with Progress(*cols, console=_CONSOLE, transient=True) as prog:
        prog.add_task(description, total=None)
        yield


# ======================================================================
# §5  CONTAINER MANAGER  (Singularity / Apptainer)
# ======================================================================

_logger = get_logger(f"{_LIB_LOGGER_NAME}.runtime")

_DEFAULT_CACHE_DIR: Path = Path(
    os.environ.get(
        "SPECTRALBRAIN_CACHE",
        str(Path.home() / ".cache" / "spectralbrain" / "containers"),
    )
)


@dataclass
class ContainerSpec:
    """Specification for one DL preprocessing container.

    Parameters
    ----------
    name : str
        Human-readable tool name.
    sif_filename : str
        Local cache filename.
    source_url : str
        HTTPS download URL.
    sha256 : str
        Expected SHA-256 digest for integrity check.
    size_mb : int
        Approximate download size (shown to user).
    entrypoint : str
        Command template — use ``{input}`` and ``{output}`` placeholders.
    gpu_required : bool
        Needs ``--nv`` GPU passthrough.
    """

    name: str
    sif_filename: str
    source_url: str
    sha256: str
    size_mb: int
    entrypoint: str
    gpu_required: bool = True


# ── Default registry (placeholders until GHCR images are built) ──────

CONTAINER_REGISTRY: Dict[str, ContainerSpec] = {
    "hdbet": ContainerSpec(
        name="HD-BET",
        sif_filename="spectralbrain_hdbet_v1.0.sif",
        source_url=(
            "https://github.com/rdneuro/spectralbrain-containers"
            "/releases/download/v0.1.0/spectralbrain_hdbet_v1.0.sif"
        ),
        sha256="placeholder",
        size_mb=2400,
        entrypoint="hd-bet -i {input} -o {output} -mode fast -tta 0",
    ),
    "synthseg": ContainerSpec(
        name="SynthSeg",
        sif_filename="spectralbrain_synthseg_v2.0.sif",
        source_url=(
            "https://github.com/rdneuro/spectralbrain-containers"
            "/releases/download/v0.1.0/spectralbrain_synthseg_v2.0.sif"
        ),
        sha256="placeholder",
        size_mb=1800,
        entrypoint="mri_synthseg --i {input} --o {output} --robust",
    ),
    "fastsurfer": ContainerSpec(
        name="FastSurfer",
        sif_filename="spectralbrain_fastsurfer_v2.3.sif",
        source_url=(
            "https://github.com/rdneuro/spectralbrain-containers"
            "/releases/download/v0.1.0/spectralbrain_fastsurfer_v2.3.sif"
        ),
        sha256="placeholder",
        size_mb=3500,
        entrypoint=(
            "run_fastsurfer.sh --t1 {input} --sd {output} --seg_only"
        ),
    ),
}


def _detect_runtime() -> Optional[str]:
    """Find ``apptainer`` or ``singularity`` on PATH."""
    for name in ("apptainer", "singularity"):
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def _has_nvidia_gpu() -> bool:
    """Return True if ``nvidia-smi`` exits successfully."""
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class ContainerManager:
    """Download, cache, verify, and execute Singularity containers.

    Containers are ``.sif`` files stored under a local cache directory
    (default ``~/.cache/spectralbrain/containers/``; override with the
    ``SPECTRALBRAIN_CACHE`` environment variable).  Each is downloaded
    once on first use and verified via SHA-256.

    Parameters
    ----------
    cache_dir : PathLike
        Container storage directory.
    registry : dict, optional
        Tool name → :class:`ContainerSpec` mapping.

    Examples
    --------
    >>> cm = ContainerManager()
    >>> cm.status()
    >>> cm.run("hdbet",
    ...        input_path="sub-01_T1w.nii.gz",
    ...        output_path="sub-01_brain.nii.gz")
    """

    def __init__(
        self,
        cache_dir: PathLike = _DEFAULT_CACHE_DIR,
        registry: Optional[Dict[str, ContainerSpec]] = None,
    ) -> None:
        """Initialise the container runner configuration."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or CONTAINER_REGISTRY
        self._runtime: Optional[str] = _detect_runtime()

    # ── properties ────────────────────────────────────────────────────

    @property
    def runtime_available(self) -> bool:
        """True if Singularity/Apptainer is installed."""
        return self._runtime is not None

    @property
    def runtime_name(self) -> str:
        """``'apptainer'``, ``'singularity'``, or ``'none'``."""
        return Path(self._runtime).stem if self._runtime else "none"

    # ── internal helpers ──────────────────────────────────────────────

    def _sif_path(self, tool: str) -> Path:
        """Resolve the Singularity/Apptainer image path."""
        return self.cache_dir / self.registry[tool].sif_filename

    # ── public API ────────────────────────────────────────────────────

    def is_cached(self, tool: str) -> bool:
        """Check whether a container is already downloaded."""
        return self._sif_path(tool).exists()

    def ensure(self, tool: str) -> Path:
        """Download *tool* container if not cached.

        Parameters
        ----------
        tool : str
            Registry key (e.g. ``"hdbet"``).

        Returns
        -------
        Path
            Local ``.sif`` path.

        Raises
        ------
        KeyError
            Unknown tool name.
        RuntimeError
            Download or checksum failure.
        """
        if tool not in self.registry:
            available = ", ".join(self.registry)
            raise KeyError(
                f"Unknown container '{tool}'. Available: {available}"
            )
        sif = self._sif_path(tool)
        if sif.exists():
            _logger.info("Container [bold]%s[/] already cached.", tool)
            return sif

        spec = self.registry[tool]
        _logger.info(
            "Downloading [bold]%s[/] (~%d MB) — first time only.",
            spec.name, spec.size_mb,
        )
        tmp = sif.with_suffix(".part")
        try:
            with progress_spinner(f"Downloading {spec.name}"):
                urllib.request.urlretrieve(spec.source_url, tmp)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Download failed for {spec.name}: {exc}") from exc

        if spec.sha256 != "placeholder":
            digest = _sha256(tmp)
            if digest != spec.sha256:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA-256 mismatch for {spec.name}: "
                    f"expected {spec.sha256[:16]}…, got {digest[:16]}…"
                )

        tmp.rename(sif)
        _logger.info("Cached [bold]%s[/] → %s", spec.name, sif)
        return sif

    def run(
        self,
        tool: str,
        *,
        input_path: PathLike,
        output_path: PathLike,
        extra_binds: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
        gpu: Optional[bool] = None,
    ) -> subprocess.CompletedProcess:
        """Execute a containerised preprocessing tool.

        Parameters
        ----------
        tool : str
            Registry key.
        input_path : PathLike
            Input file (bind-mounted read-only).
        output_path : PathLike
            Output file (parent dir bind-mounted read-write).
        extra_binds : list of str, optional
            Additional ``"host:container"`` bind specs.
        extra_args : list of str, optional
            Arguments appended to the entrypoint.
        gpu : bool or None
            Force GPU on/off; ``None`` = auto-detect.

        Returns
        -------
        subprocess.CompletedProcess

        Raises
        ------
        EnvironmentError
            No container runtime found.
        subprocess.CalledProcessError
            Container exited with non-zero status.
        """
        if self._runtime is None:
            raise EnvironmentError(
                "No container runtime found. Install Apptainer:\n"
                "  https://apptainer.org/docs/admin/latest/installation.html"
            )
        sif = self.ensure(tool)
        spec = self.registry[tool]

        inp = Path(input_path).resolve()
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [self._runtime, "exec"]

        use_gpu = gpu if gpu is not None else (
            spec.gpu_required and _has_nvidia_gpu()
        )
        if use_gpu:
            cmd.append("--nv")

        cmd += ["--bind", f"{inp.parent}:/input:ro",
                "--bind", f"{out.parent}:/output:rw"]
        for b in extra_binds or []:
            cmd += ["--bind", b]

        cmd.append(str(sif))
        cmd += spec.entrypoint.format(
            input=f"/input/{inp.name}",
            output=f"/output/{out.name}",
        ).split()
        cmd += extra_args or []

        _logger.info(
            "Running [bold]%s[/] (%s)",
            spec.name, "GPU" if use_gpu else "CPU",
        )
        _logger.debug("$ %s", " ".join(cmd))
        return subprocess.run(cmd, check=True, capture_output=True, text=True)

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Print and return status of all registered containers."""
        report: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.registry.items():
            sif = self._sif_path(name)
            report[name] = {
                "cached": sif.exists(),
                "path": str(sif) if sif.exists() else None,
                "size_mb": spec.size_mb,
                "gpu": spec.gpu_required,
            }
        if _HAS_RICH and _CONSOLE is not None:
            _CONSOLE.rule("[bold]SpectralBrain containers")
            _CONSOLE.print(f"Runtime : [bold]{self.runtime_name}[/]")
            _CONSOLE.print(f"Cache   : {self.cache_dir}\n")
            for name, info in report.items():
                ok = "✓" if info["cached"] else "✗"
                c = "green" if info["cached"] else "red"
                _CONSOLE.print(
                    f"  [{c}]{ok}[/]  [bold]{name:12s}[/]  "
                    f"{info['size_mb']:>5d} MB  "
                    f"{'GPU' if info['gpu'] else 'CPU'}"
                )
        return report

    def clean(self, tool: Optional[str] = None) -> None:
        """Remove cached container(s).

        Parameters
        ----------
        tool : str or None
            Specific tool, or ``None`` to clear all.
        """
        targets = [tool] if tool else list(self.registry)
        for t in targets:
            sif = self._sif_path(t)
            if sif.exists():
                sif.unlink()
                _logger.info("Removed %s", sif)


# ======================================================================
# §6  __all__
# ======================================================================

__all__: List[str] = [
    # §1 Versioning
    "__version__",
    "VERSION_INFO",
    # §2 Types — geometric
    "Vertices", "Faces", "Points", "Normals",
    # §2 Types — spectral
    "Eigenvalues", "Eigenvectors", "SparseMatrix", "MassMatrix",
    # §2 Types — descriptors
    "ScalarMap", "DescriptorMatrix", "GlobalDescriptor", "DistanceMatrix",
    # §2 Types — neuroimaging
    "LabelArray", "VolumeImage", "SurfaceImage",
    # §2 Types — analysis
    "ConnectomeMatrix", "NetworkMatrix",
    # §2 Types — generic
    "PathLike",
    # §2b Enums
    "GeometryFormat", "AtlasScheme", "DescriptorType",
    "AnalysisObjective", "BackendName",
    # §2b Eligibility
    "DESCRIPTOR_ELIGIBILITY",
    # §3 Logging
    "get_logger", "set_log_level",
    # §4 Progress
    "progress_simple", "progress_parallel",
    "progress_spinner", "NestedProgress",
    # §5 Containers
    "ContainerSpec", "ContainerManager", "CONTAINER_REGISTRY",
]
