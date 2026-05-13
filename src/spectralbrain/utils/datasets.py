"""Template data, example datasets, and public dataset fetchers.

Provides quick access to template geometries (fsaverage, MNI152)
and synthetic example datasets for tutorials and testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralbrain.runtime import Faces, PathLike, Vertices, get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "spectralbrain" / "datasets"


# ======================================================================
# §1  SYNTHETIC EXAMPLE DATASETS
# ======================================================================

def make_two_group_example(
    n_per_group: int = 30,
    n_vertices: int = 500,
    n_scales: int = 10,
    effect_size: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a two-group synthetic dataset for tutorials.

    Creates descriptors for controls and patients with a focal
    spectral difference at a subset of vertices.

    Parameters
    ----------
    n_per_group : int
    n_vertices : int
    n_scales : int
    effect_size : float
        Cohen's d of the planted difference.
    seed : int

    Returns
    -------
    dict
        Keys: ``"controls"`` (n, N, T), ``"patients"`` (n, N, T),
        ``"labels"`` (2n,), ``"affected_vertices"`` (bool mask),
        ``"ages"`` (2n,), ``"metadata"``.
    """
    rng = np.random.default_rng(seed)

    controls = rng.normal(0, 1, (n_per_group, n_vertices, n_scales))
    patients = rng.normal(0, 1, (n_per_group, n_vertices, n_scales))

    # Plant a focal effect at ~20% of vertices.
    affected = np.zeros(n_vertices, dtype=bool)
    affected[rng.choice(n_vertices, n_vertices // 5, replace=False)] = True
    patients[:, affected, :] += effect_size

    labels = np.array([0] * n_per_group + [1] * n_per_group)
    ages = rng.normal(45, 15, 2 * n_per_group).clip(18, 85)

    return {
        "controls": controls,
        "patients": patients,
        "labels": labels,
        "affected_vertices": affected,
        "ages": ages,
        "metadata": {
            "n_per_group": n_per_group,
            "effect_size": effect_size,
            "n_affected": int(affected.sum()),
        },
    }


def make_normative_example(
    n_subjects: int = 200,
    n_vertices: int = 500,
    age_range: Tuple[float, float] = (20, 80),
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a synthetic normative cohort with age effects.

    Parameters
    ----------
    n_subjects : int
    n_vertices : int
    age_range : tuple
    seed : int

    Returns
    -------
    dict
        Keys: ``"descriptors"`` (S, N), ``"ages"`` (S,),
        ``"sex"`` (S,), ``"age_slope"`` (N,).
    """
    rng = np.random.default_rng(seed)
    ages = rng.uniform(*age_range, n_subjects)
    sex = rng.binomial(1, 0.5, n_subjects)

    # Per-vertex age slope (some positive, some negative).
    age_slope = rng.normal(0, 0.01, n_vertices)

    descriptors = (
        rng.normal(0, 1, (n_subjects, n_vertices))
        + ages[:, None] * age_slope[None, :]
        + sex[:, None] * rng.normal(0, 0.1, n_vertices)[None, :]
    )

    return {
        "descriptors": descriptors,
        "ages": ages,
        "sex": sex,
        "age_slope": age_slope,
    }


def make_connectome_example(
    n_subjects: int = 40,
    n_parcels: int = 50,
    n_networks: int = 5,
    group_effect: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic geometric connectomes for two groups.

    Parameters
    ----------
    n_subjects : int
    n_parcels : int
    n_networks : int
    group_effect : float
    seed : int

    Returns
    -------
    dict
        Keys: ``"connectomes"`` (S, R, R), ``"labels"`` (S,),
        ``"network_assignments"`` dict.
    """
    rng = np.random.default_rng(seed)
    n_half = n_subjects // 2

    # Base connectome with block structure.
    net_assign = {i: f"Net{i % n_networks}" for i in range(n_parcels)}
    block = np.zeros((n_parcels, n_parcels))
    for i in range(n_parcels):
        for j in range(i + 1, n_parcels):
            same_net = (i % n_networks) == (j % n_networks)
            block[i, j] = 0.7 if same_net else 0.3
            block[j, i] = block[i, j]

    connectomes = []
    for s in range(n_subjects):
        noise = rng.normal(0, 0.1, (n_parcels, n_parcels))
        noise = (noise + noise.T) / 2
        C = block + noise
        if s >= n_half:
            # Add group effect to intra-network edges.
            for i in range(n_parcels):
                for j in range(i + 1, n_parcels):
                    if (i % n_networks) == (j % n_networks):
                        C[i, j] -= group_effect
                        C[j, i] = C[i, j]
        np.fill_diagonal(C, 0)
        connectomes.append(C)

    return {
        "connectomes": np.array(connectomes),
        "labels": np.array([0] * n_half + [1] * n_half),
        "network_assignments": net_assign,
    }


def make_laterality_example(
    n_subjects: int = 40,
    n_features: int = 50,
    asymmetry: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic bilateral descriptors for asymmetry analysis.

    Parameters
    ----------
    n_subjects : int
    n_features : int
    asymmetry : float
        Planted L > R asymmetry in patients.
    seed : int

    Returns
    -------
    dict
        Keys: ``"left"`` (S, d), ``"right"`` (S, d), ``"labels"`` (S,).
    """
    rng = np.random.default_rng(seed)
    n_half = n_subjects // 2

    left = rng.normal(0, 1, (n_subjects, n_features))
    right = rng.normal(0, 1, (n_subjects, n_features))

    # Patients: left > right for some features.
    affected = rng.choice(n_features, n_features // 3, replace=False)
    left[n_half:, affected] += asymmetry

    return {
        "left": left,
        "right": right,
        "labels": np.array([0] * n_half + [1] * n_half),
        "affected_features": affected,
    }


# ======================================================================
# §2  TEMPLATE LOADERS
# ======================================================================

def fetch_fsaverage(
    mesh: str = "pial",
    hemisphere: str = "lh",
) -> Tuple[Vertices, Faces]:
    """Load fsaverage template surfaces from nibabel's bundled data.

    Parameters
    ----------
    mesh : str
        ``"pial"``, ``"white"``, ``"inflated"``, ``"sphere"``.
    hemisphere : str
        ``"lh"`` or ``"rh"``.

    Returns
    -------
    vertices, faces
    """
    try:
        import nibabel as nib
        from nibabel import freesurfer as fs
    except ImportError as exc:
        raise ImportError("nibabel required for fsaverage.") from exc

    # nibabel ships fsaverage in its data directory.
    try:
        data_dir = Path(nib.__file__).parent / "freesurfer" / "data"
        surf_path = data_dir / f"fsaverage" / "surf" / f"{hemisphere}.{mesh}"
        if surf_path.exists():
            v, f = fs.read_geometry(str(surf_path))
            return np.asarray(v, np.float64), np.asarray(f, np.int64)
    except Exception:
        pass

    # Fallback: try nilearn's fetch_surf_fsaverage.
    try:
        from nilearn.datasets import fetch_surf_fsaverage
        fsavg = fetch_surf_fsaverage(mesh=f"fsaverage")
        key = f"{mesh}_{hemisphere}"
        v, f = nib.load(fsavg[key]).darrays[0].data, nib.load(fsavg[key]).darrays[1].data
        return np.asarray(v, np.float64), np.asarray(f, np.int64)
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not load fsaverage. Install nibabel or nilearn:\n"
        "  pip install nibabel nilearn"
    )


# ======================================================================
# §3  EXAMPLE MESH/POINTCLOUD
# ======================================================================

def example_sphere(
    n_lat: int = 30,
    n_lon: int = 60,
    radius: float = 50.0,
) -> Tuple[Vertices, Faces]:
    """Quick sphere mesh for testing.

    Returns
    -------
    vertices, faces
    """
    from spectralbrain.statistics.surrogates import SyntheticMesh
    return SyntheticMesh(seed=0).sphere(n_lat, n_lon, radius)


def example_point_cloud(
    n_points: int = 1000,
    shape: str = "sphere",
    seed: int = 0,
) -> np.ndarray:
    """Quick point cloud for testing.

    Parameters
    ----------
    n_points : int
    shape : str
        ``"sphere"``, ``"ellipsoid"``, ``"blob"``, ``"multi_cluster"``.
    seed : int

    Returns
    -------
    ndarray, shape (n_points, 3)
    """
    from spectralbrain.statistics.surrogates import SyntheticPointCloud
    gen = SyntheticPointCloud(seed=seed)
    if shape == "sphere":
        return gen.sphere(n_points)
    elif shape == "ellipsoid":
        return gen.ellipsoid(n_points)
    elif shape == "blob":
        return gen.blob(n_points)
    elif shape == "multi_cluster":
        return gen.multi_cluster(n_points)
    raise ValueError(f"Unknown shape: {shape!r}")


__all__ = [
    "make_two_group_example",
    "make_normative_example",
    "make_connectome_example",
    "make_laterality_example",
    "fetch_fsaverage",
    "example_sphere",
    "example_point_cloud",
]
