"""Group-level loading for cohort statistics.

This module closes the gap between per-file I/O and the group-statistics
functions in :mod:`spectralbrain.statistics.analysis`. The workflow is:

1. **Discover** one file per subject from a BIDS/derivatives tree, a
   FreeSurfer ``SUBJECTS_DIR``, or an explicit list/dict of paths.
2. **Load** every subject in parallel (joblib via
   :func:`spectralbrain.backends.cpu.parallel_map`), fail-soft: a subject
   that errors is logged and dropped rather than aborting the cohort.
3. **Stack** the per-subject arrays into a single ``(S, N)`` (or
   ``(S, N, T)``) array, packaged in a :class:`GroupData` object that
   carries subject IDs and parsed BIDS entities (for covariates).

Two loading modes:

- ``mode="maps"`` — load a per-vertex overlay/metric or a precomputed
  descriptor field that is **already vertex-corresponded** on a common
  template (the light path).
- ``mode="pipeline"`` — load each surface, build the Laplace–Beltrami
  decomposition, and compute a spectral descriptor per subject (the heavy
  path, where joblib and the GPU backends pay off).

The resulting :class:`GroupData` plugs straight into
:func:`group_comparison`, which dispatches to the vertex-wise tests.

Examples
--------
>>> # HippUnfold-style derivatives, descriptor fields already on template:
>>> files = discover_bids(
...     "/data/derivatives/hippunfold",
...     "sub-{sub}/surf/sub-{sub}_hemi-L_*_thickness.shape.gii",
... )
>>> group = load_group(files, mode="maps", n_jobs=8)
>>> res = group_comparison(group, group.covariate("group"), test="ttest")

>>> # Full pipeline from FreeSurfer surfaces, HKS per subject, on GPU:
>>> files = discover_freesurfer("/data/fs", surface="white", hemi="lh")
>>> from spectralbrain.backends import TorchBackend
>>> group = load_group(
...     files, mode="pipeline", descriptor="hks", k=100,
...     backend=TorchBackend(), n_jobs=4,
... )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spectralbrain.io.loaders import load as _load
from spectralbrain.runtime import PathLike, get_logger
from spectralbrain.utils.helpers import parse_bids_filename

logger = get_logger(__name__)


# ======================================================================
# §1  GROUP CONTAINER
# ======================================================================


@dataclass
class GroupData:
    """A loaded cohort ready for group statistics.

    Attributes
    ----------
    data : ndarray or list of ndarray
        Stacked per-subject arrays, shape ``(S, N)`` or ``(S, N, T)``.
        Falls back to a list if subjects have heterogeneous shapes.
    subject_ids : list of str
        Subject identifiers, aligned with ``data``'s first axis.
    entities : list of dict
        BIDS entities parsed from each source filename (for covariates).
    paths : list of Path
        Source file per subject.
    faces : ndarray, optional
        Template faces ``(F, 3)`` — useful for TFCE adjacency.
    metadata : dict
        Bookkeeping (mode, number of failed subjects, …).
    """

    data: Any
    subject_ids: list[str]
    entities: list[dict[str, str]]
    paths: list[Path]
    faces: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.subject_ids)

    @property
    def n_subjects(self) -> int:
        """Number of successfully loaded subjects."""
        return len(self.subject_ids)

    @property
    def is_stacked(self) -> bool:
        """True if ``data`` is a single stacked array."""
        return isinstance(self.data, np.ndarray)

    def covariate(self, key: str, default: Any = None) -> np.ndarray:
        """Return a BIDS entity (e.g. ``"ses"``, ``"group"``) per subject.

        Parameters
        ----------
        key : str
            Entity key to pull from each subject's parsed filename.
        default : Any
            Value for subjects missing the entity.

        Returns
        -------
        ndarray, shape (S,)
        """
        return np.array([e.get(key, default) for e in self.entities], dtype=object)

    def split(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split the stacked data into two groups by a 2-level label array.

        Parameters
        ----------
        labels : array-like, shape (S,)
            Exactly two distinct non-null values define the two groups.

        Returns
        -------
        group_a, group_b : ndarray
            Subsets of ``data`` for the two label levels (in sorted order).
        """
        if not self.is_stacked:
            raise ValueError("Cannot split: data is not a single stacked array.")
        labels = np.asarray(labels)
        mask_valid = np.array([x is not None for x in labels])
        uniq = sorted({x for x in labels[mask_valid]})
        if len(uniq) != 2:
            raise ValueError(f"split() needs exactly 2 label levels; got {len(uniq)}: {uniq}")
        a = self.data[labels == uniq[0]]
        b = self.data[labels == uniq[1]]
        return a, b


# ======================================================================
# §2  DISCOVERY
# ======================================================================


def _norm_sid(s: str) -> str:
    """Normalise a subject id to the ``sub-XXX`` form."""
    return s if s.startswith("sub-") else f"sub-{s}"


def discover_bids(
    root: PathLike,
    pattern: str,
    *,
    subjects: list[str] | None = None,
) -> dict[str, Path]:
    """Discover one file per subject in a BIDS / derivatives tree.

    Parameters
    ----------
    root : PathLike
        Dataset (or derivatives) root.
    pattern : str
        Glob relative to *root* with a ``{sub}`` placeholder (the bare
        label, no ``sub-`` prefix), e.g.
        ``"sub-{sub}/anat/sub-{sub}_hemi-L_thickness.shape.gii"``.
    subjects : list of str, optional
        Restrict to these subjects (``"sub-01"`` or ``"01"``). Defaults to
        every ``sub-*`` directory under *root*.

    Returns
    -------
    dict of {subject_id: Path}
        Subjects with no match (or whose match is ambiguous) are logged;
        the first match is taken when several exist.
    """
    root = Path(root)
    if subjects is None:
        labels = sorted(p.name[4:] for p in root.glob("sub-*") if p.is_dir())
    else:
        labels = [s[4:] if s.startswith("sub-") else s for s in subjects]

    found: dict[str, Path] = {}
    for label in labels:
        matches = sorted(root.glob(pattern.replace("{sub}", label)))
        if not matches:
            logger.warning("No file for sub-%s (pattern %r)", label, pattern)
            continue
        if len(matches) > 1:
            logger.warning("Multiple files for sub-%s; using %s", label, matches[0].name)
        found[_norm_sid(label)] = matches[0]
    logger.info("BIDS discovery: %d subjects matched.", len(found))
    return found


def discover_freesurfer(
    subjects_dir: PathLike,
    *,
    hemi: str = "lh",
    surface: str | None = None,
    measure: str | None = None,
    subjects: list[str] | None = None,
) -> dict[str, Path]:
    """Discover FreeSurfer surface or morphometry files per subject.

    Provide exactly one of *surface* (geometry, e.g. ``"white"``) or
    *measure* (overlay, e.g. ``"thickness"``). The path resolved is
    ``{subjects_dir}/{sub}/surf/{hemi}.{name}``.

    Parameters
    ----------
    subjects_dir : PathLike
        FreeSurfer ``SUBJECTS_DIR``.
    hemi : str
        ``"lh"`` or ``"rh"``.
    surface : str, optional
        Surface geometry name (``"white"``, ``"pial"``, …).
    measure : str, optional
        Morphometry overlay (``"thickness"``, ``"curv"``, ``"sulc"``, …).
    subjects : list of str, optional
        Restrict to these subject directory names.

    Returns
    -------
    dict of {subject_id: Path}
    """
    sd = Path(subjects_dir)
    if (surface is None) == (measure is None):
        raise ValueError("Provide exactly one of `surface` or `measure`.")
    name = surface if surface is not None else measure

    if subjects is None:
        candidates = sorted(p.name for p in sd.iterdir() if (p / "surf").is_dir())
    else:
        candidates = list(subjects)

    found: dict[str, Path] = {}
    for sub in candidates:
        f = sd / sub / "surf" / f"{hemi}.{name}"
        if f.exists():
            found[sub] = f
        else:
            logger.warning("Missing %s", f)
    logger.info("FreeSurfer discovery: %d subjects matched.", len(found))
    return found


# ======================================================================
# §3  PER-SUBJECT LOADERS
# ======================================================================


def _default_map_loader(path: PathLike) -> np.ndarray:
    """Extract a per-vertex array from any supported overlay/metric file."""
    r = _load(path)
    if "scalars" in r:
        return np.asarray(r["scalars"])
    if "data" in r:  # volume → flattened
        return np.asarray(r["data"]).ravel()
    if "vertices" in r:
        raise ValueError(
            f"{Path(path).name} is surface geometry (no per-vertex scalar). "
            "Use mode='pipeline' to compute a descriptor, or point at an "
            "overlay/metric file."
        )
    raise ValueError(f"No per-vertex array found in {Path(path).name}")


def _make_descriptor_loader(
    descriptor: str = "hks",
    *,
    k: int = 100,
    backend: Any | None = None,
    laplacian: str = "cotangent",
    **descriptor_kwargs: Any,
) -> Callable[[PathLike], np.ndarray]:
    """Build a loader: surface file → decompose → spectral descriptor."""
    from spectralbrain.core.meshes import BrainMesh
    from spectralbrain.spectral import descriptors as _desc

    table: dict[str, Callable[..., np.ndarray]] = {
        "hks": _desc.compute_hks,
        "si_hks": _desc.compute_si_hks,
        "wks": _desc.compute_wks,
        "gps": _desc.compute_gps,
        "shapedna": _desc.compute_shapedna,
    }
    if descriptor not in table:
        raise ValueError(f"Unknown descriptor {descriptor!r}; choose from {list(table)}")
    fn = table[descriptor]

    def _loader(path: PathLike) -> np.ndarray:
        r = _load(path)
        if "vertices" not in r or "faces" not in r:
            raise ValueError(f"{Path(path).name} is not a surface mesh.")
        mesh = BrainMesh(r["vertices"], r["faces"])
        decomp = mesh.decompose(k=k, laplacian_method=laplacian, backend=backend)
        return np.asarray(fn(decomp, **descriptor_kwargs))

    return _loader


# ======================================================================
# §4  GROUP LOADER
# ======================================================================


def _finalize_group(
    items: list[tuple[str, Path]],
    arrays: list[np.ndarray | None],
    *,
    mode: str,
    stack: bool = True,
    faces: np.ndarray | None = None,
) -> GroupData:
    """Drop failed subjects, stack if shapes match, and package a GroupData."""
    ok = [(s, p, a) for (s, p), a in zip(items, arrays) if a is not None]
    n_failed = len(items) - len(ok)
    sids = [s for s, _, _ in ok]
    out_paths = [p for _, p, _ in ok]
    entities = [parse_bids_filename(p.name) for p in out_paths]
    loaded = [a for _, _, a in ok]

    data: Any
    if stack and loaded and len({a.shape for a in loaded}) == 1:
        data = np.stack(loaded, axis=0)
    else:
        if stack and loaded:
            logger.warning("Heterogeneous subject shapes; returning a list, not a stacked array.")
        data = loaded

    logger.info("Loaded group: %d/%d subjects (mode=%s).", len(ok), len(items), mode)
    return GroupData(
        data=data,
        subject_ids=sids,
        entities=entities,
        paths=out_paths,
        faces=faces,
        metadata={"mode": mode, "n_failed": n_failed, "n_requested": len(items)},
    )


def load_group(
    files: dict[str, PathLike] | list[PathLike],
    *,
    mode: str = "maps",
    loader: Callable[[PathLike], np.ndarray] | None = None,
    n_jobs: int = 1,
    stack: bool = True,
    descriptor: str = "hks",
    k: int = 100,
    backend: Any | None = None,
    descriptor_kwargs: dict[str, Any] | None = None,
    template_faces: np.ndarray | None = None,
) -> GroupData:
    """Load and stack a cohort for group statistics.

    Parameters
    ----------
    files : dict or list
        ``{subject_id: path}`` (e.g. from :func:`discover_bids` /
        :func:`discover_freesurfer`) or a plain list of paths (subject IDs
        are then parsed from the filenames).
    mode : ``"maps"`` or ``"pipeline"``
        ``"maps"`` loads vertex-corresponded overlays/descriptor fields;
        ``"pipeline"`` loads each surface and computes a descriptor.
    loader : callable, optional
        Custom ``path -> ndarray`` loader. Overrides *mode*.
    n_jobs : int
        Parallel workers for loading (joblib). ``1`` = sequential.
    stack : bool
        Stack into one array when subject shapes match (else keep a list).
    descriptor, k, backend, descriptor_kwargs
        Pipeline-mode options (descriptor name, eigenpairs, compute
        backend, and extra keyword arguments forwarded to the descriptor).
    template_faces : ndarray, optional
        Stored on the result for downstream TFCE adjacency.

    Returns
    -------
    GroupData
    """
    if isinstance(files, dict):
        items: list[tuple[str, Path]] = [(s, Path(p)) for s, p in files.items()]
    else:
        items = []
        for p in files:
            p = Path(p)
            sub = parse_bids_filename(p.name).get("sub")
            items.append((_norm_sid(sub) if sub else p.stem, p))

    if loader is None:
        if mode == "maps":
            loader = _default_map_loader
        elif mode == "pipeline":
            loader = _make_descriptor_loader(
                descriptor, k=k, backend=backend, **(descriptor_kwargs or {})
            )
        else:
            raise ValueError(f"Unknown mode {mode!r}; use 'maps' or 'pipeline'.")

    paths = [p for _, p in items]
    active_loader = loader

    def _one(path: Path) -> np.ndarray | None:
        try:
            return np.asarray(active_loader(path))
        except Exception as exc:
            logger.error("✗ %s: %s", path.name, exc)
            return None

    if n_jobs == 1:
        arrays = [_one(p) for p in paths]
    else:
        from spectralbrain.backends.cpu import parallel_map

        arrays = parallel_map(_one, paths, n_jobs=n_jobs, description="Loading group")

    return _finalize_group(items, arrays, mode=mode, stack=stack, faces=template_faces)


# ======================================================================
# §5  TEMPLATE RESAMPLING (FreeSurfer)
# ======================================================================


def _resolve_sphere(subjects_dir: Path, subject: str, hemi: str) -> Path:
    """Locate a subject's (or template's) ``{hemi}.sphere.reg``."""
    from spectralbrain.io.parcellate import _find_freesurfer_home

    cand = subjects_dir / subject / "surf" / f"{hemi}.sphere.reg"
    if cand.exists():
        return cand
    fs_home = _find_freesurfer_home()
    if fs_home:
        alt = fs_home / "subjects" / subject / "surf" / f"{hemi}.sphere.reg"
        if alt.exists():
            return alt
    raise FileNotFoundError(
        f"{hemi}.sphere.reg not found for '{subject}'. Run recon-all, or set "
        "FREESURFER_HOME so fsaverage can be located."
    )


def resample_to_template(
    values: np.ndarray,
    subjects_dir: PathLike,
    subject_id: str,
    hemi: str,
    *,
    template: str = "fsaverage",
    method: str = "nearest",
    k: int = 3,
) -> np.ndarray:
    """Resample a native-space per-vertex overlay onto a template surface.

    Mirrors FreeSurfer's ``mri_surf2surf``: both surfaces are brought into
    spherical-registration space (``{hemi}.sphere.reg``) and the template
    vertices sample the subject overlay there. ``"nearest"`` takes the
    closest subject vertex (matching SpectralBrain's existing label
    projection); ``"linear"`` blends the *k* nearest by inverse distance,
    which is smoother for continuous metrics.

    Parameters
    ----------
    values : ndarray, shape (N_subject,)
        Per-vertex overlay on the subject's native surface.
    subjects_dir : PathLike
        FreeSurfer ``SUBJECTS_DIR`` (must also contain *template*).
    subject_id : str
        Subject directory name.
    hemi : str
        ``"lh"`` or ``"rh"``.
    template : str
        Template subject (default ``"fsaverage"``).
    method : ``"nearest"`` or ``"linear"``
        Interpolation on the registration sphere.
    k : int
        Neighbours for ``"linear"`` inverse-distance weighting.

    Returns
    -------
    ndarray, shape (N_template,)
        Overlay resampled onto the template surface.
    """
    from scipy.spatial import cKDTree

    from spectralbrain.io.loaders import load_freesurfer_surface

    sd = Path(subjects_dir)
    values = np.asarray(values)

    src_coords, _ = load_freesurfer_surface(_resolve_sphere(sd, subject_id, hemi))
    tgt_coords, _ = load_freesurfer_surface(_resolve_sphere(sd, template, hemi))

    if len(values) != len(src_coords):
        raise ValueError(
            f"Overlay has {len(values)} values but subject surface has {len(src_coords)} vertices."
        )

    tree = cKDTree(src_coords)
    if method == "nearest":
        _, idx = tree.query(tgt_coords, k=1)
        return values[idx]
    if method == "linear":
        dist, idx = tree.query(tgt_coords, k=k)
        w = 1.0 / np.clip(dist, 1e-12, None)
        w /= w.sum(axis=1, keepdims=True)
        return (values[idx] * w).sum(axis=1)
    raise ValueError(f"Unknown method {method!r}; use 'nearest' or 'linear'.")


def load_group_freesurfer(
    subjects_dir: PathLike,
    *,
    measure: str,
    hemi: str = "lh",
    template: str = "fsaverage",
    resample: bool = True,
    method: str = "nearest",
    subjects: list[str] | None = None,
    n_jobs: int = 1,
) -> GroupData:
    """Load a FreeSurfer morphometry measure across a cohort onto a template.

    For each subject the native overlay (``{hemi}.{measure}``) is loaded and
    — unless ``resample=False`` — resampled to *template* via
    :func:`resample_to_template`, so the cohort stacks into a single
    vertex-corresponded ``(S, N_template)`` array ready for
    :func:`group_comparison`.

    Parameters
    ----------
    subjects_dir : PathLike
        FreeSurfer ``SUBJECTS_DIR``.
    measure : str
        Morphometry overlay (``"thickness"``, ``"curv"``, ``"sulc"``, …).
    hemi : str
        ``"lh"`` or ``"rh"``.
    template : str
        Target template subject (default ``"fsaverage"``).
    resample : bool
        Resample to *template* (True) or keep native space (False; only
        stackable if every subject shares the vertex count).
    method : ``"nearest"`` or ``"linear"``
        Resampling interpolation.
    subjects : list of str, optional
        Restrict to these subject directory names.
    n_jobs : int
        Parallel workers (joblib).

    Returns
    -------
    GroupData
    """
    from spectralbrain.io.loaders import load_freesurfer_morph

    sd = Path(subjects_dir)
    files = discover_freesurfer(sd, hemi=hemi, measure=measure, subjects=subjects)
    items: list[tuple[str, Path]] = list(files.items())

    def _one(item: tuple[str, Path]) -> np.ndarray | None:
        sid, path = item
        try:
            vals = np.asarray(load_freesurfer_morph(path))
            if resample:
                vals = resample_to_template(vals, sd, sid, hemi, template=template, method=method)
            return vals
        except Exception as exc:
            logger.error("✗ %s: %s", sid, exc)
            return None

    if n_jobs == 1:
        arrays = [_one(it) for it in items]
    else:
        from spectralbrain.backends.cpu import parallel_map

        arrays = parallel_map(_one, items, n_jobs=n_jobs, description="Loading FS group")

    mode = f"freesurfer:{measure}" + (f"→{template}" if resample else "")
    return _finalize_group(items, arrays, mode=mode)


# ======================================================================
# §6  ANALYSIS GLUE
# ======================================================================


def group_comparison(
    group: GroupData | tuple[np.ndarray, np.ndarray],
    labels: np.ndarray | None = None,
    *,
    test: str = "ttest",
    **kwargs: Any,
) -> Any:
    """Run a vertex-wise group comparison on a loaded cohort.

    Parameters
    ----------
    group : GroupData or (group_a, group_b)
        A loaded cohort (split via *labels*) or a pre-split pair of arrays.
    labels : array-like, shape (S,), optional
        Two-level grouping variable (required when *group* is a
        :class:`GroupData`). Often ``group.covariate("group")``.
    test : ``"ttest"``, ``"mannwhitney"``, or ``"permutation"``
        Which vertex-wise test from
        :mod:`spectralbrain.statistics.analysis` to run.
    **kwargs
        Forwarded to the chosen test (e.g. ``correction``, ``alpha``,
        ``n_permutations``).

    Returns
    -------
    VertexWiseResult
    """
    from spectralbrain.statistics import analysis as _analysis

    tests: dict[str, Callable[..., Any]] = {
        "ttest": _analysis.vertexwise_ttest,
        "mannwhitney": _analysis.vertexwise_mannwhitney,
        "permutation": _analysis.vertexwise_permutation,
    }
    if test not in tests:
        raise ValueError(f"Unknown test {test!r}; choose from {list(tests)}")

    if isinstance(group, GroupData):
        if labels is None:
            raise ValueError("`labels` is required when passing a GroupData.")
        group_a, group_b = group.split(labels)
    else:
        group_a, group_b = group

    return tests[test](group_a, group_b, **kwargs)


__all__ = [
    "GroupData",
    "discover_bids",
    "discover_freesurfer",
    "group_comparison",
    "load_group",
    "load_group_freesurfer",
    "resample_to_template",
]
