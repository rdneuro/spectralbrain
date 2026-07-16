"""Deterministic volume → surface meshing for well-conditioned LBO analysis.

The Laplace–Beltrami operator (and the descriptors built on it — ShapeDNA,
HKS, WKS, …) is sensitive to **triangle quality**, not to vertex count. A mesh
extracted directly from a binary label volume by plain marching cubes carries
artefacts that corrupt the spectrum: the voxel *staircase*, *sliver* triangles
(whose vanishing angles make cotangent weights blow up or turn negative),
non-uniform sampling, and — for multi-label sources — non-manifold, multi-part,
high-genus surfaces.

This module turns a volume into an **LBO-ready** mesh through deterministic,
shape-agnostic geometry processing (no learned prior, so it never regularises
pathological anatomy toward a "normal" shape):

    1. label → scalar field   : binary mask → signed distance field (SDF)
                                 (or Gaussian-smoothed occupancy)
    2. Gaussian smoothing      : band-limit the boundary, topology preserved
    3. marching cubes          : sub-voxel, staircase-free iso-surface
    4. topology cleanup        : largest connected component, manifold repair,
                                 watertight (closed structures only)
    5. Taubin smoothing (λ/μ)  : volume-preserving denoise
    6. isotropic remeshing     : ACVD → uniform, well-shaped triangles

Public API
----------
volume_to_mesh
    Volume (or label volume) → ``(vertices, faces)``. ``raw=True`` reproduces
    the legacy plain-marching-cubes path exactly; ``raw=False`` (default)
    applies the improvement pipeline.
refine_mesh
    Apply steps 4–6 to an *existing* mesh (e.g. a surface that is already
    triangulated but poorly conditioned for the LBO).

Both are re-exported at :mod:`spectralbrain.io` and the package top level;
:meth:`spectralbrain.BrainMesh.from_volume` is the object-oriented entry point.

Notes
-----
``closed=True`` (the default) is appropriate for closed anatomical structures
(hippocampus, subcortical nuclei): it uses an SDF, keeps a single connected
component, and repairs the surface to a watertight 2-manifold. ``closed=False``
is the correct choice for open / branching / multi-part structures (e.g.
white-matter bundle masks): it skips the watertight and single-component steps
so that legitimate anatomy is never amputated.

Optional dependencies (lazily imported, with install hints): ``trimesh``
(topology + Taubin), ``pyacvd`` + ``pyvista`` (isotropic remeshing),
``pymeshfix`` (watertight repair). ``scikit-image`` and ``scipy`` are core.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import ndimage as ndi

from spectralbrain.core.base import marching_cubes as _raw_marching_cubes
from spectralbrain.runtime import Faces, Vertices, get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import trimesh as _trimesh_t

logger = get_logger(__name__)

FieldMode = Literal["sdf", "gaussian"]

__all__ = ["refine_mesh", "volume_to_mesh"]


# ======================================================================
# §0  Optional-dependency guards (lazy, with install hints)
# ======================================================================
def _require_trimesh() -> Any:
    """Import ``trimesh`` or raise with an install hint."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised only w/o dep
        raise ImportError(
            "trimesh is required for mesh improvement.\n"
            "  pip install trimesh    (or: pip install 'spectralbrain[viz]')\n"
            "Alternatively pass raw=True to skip the improvement pipeline."
        ) from exc
    return trimesh


def _require_pyacvd() -> tuple[Any, Any]:
    """Import ``pyvista`` and ``pyacvd`` or raise with an install hint."""
    try:
        import pyacvd
        import pyvista
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyacvd + pyvista are required for isotropic remeshing.\n"
            "  pip install pyacvd pyvista\n"
            "Alternatively pass remesh=False."
        ) from exc
    return pyvista, pyacvd


def _require_pymeshfix() -> Any:
    """Import ``pymeshfix`` or raise with an install hint."""
    try:
        import pymeshfix
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pymeshfix is required for watertight repair.\n"
            "  pip install pymeshfix\n"
            "Alternatively pass watertight=False."
        ) from exc
    return pymeshfix


# ======================================================================
# §1  Volume → scalar field
# ======================================================================
def _binarize(volume: np.ndarray, label: int | Sequence[int] | None,
              level: float) -> np.ndarray:
    """Return a boolean mask from a label volume or an intensity threshold."""
    volume = np.asarray(volume)
    if label is None:
        return volume > level
    labels = [label] if np.isscalar(label) else list(label)
    return np.isin(volume, labels)


def _mask_to_field(
    mask: np.ndarray,
    *,
    mode: FieldMode = "sdf",
    presmooth_vox: float = 0.5,
    sigma_vox: float = 0.6,
    pad: int = 4,
) -> tuple[np.ndarray, float, int]:
    """Convert a binary mask into a smooth scalar field for marching cubes.

    Parameters
    ----------
    mask : ndarray of bool, shape (X, Y, Z)
    mode : {"sdf", "gaussian"}
        ``"sdf"`` — signed distance field (inside negative), iso-level 0.
        Robust for thin, curved structures. ``"gaussian"`` — Gaussian-smoothed
        occupancy in [0, 1], iso-level 0.5.
    presmooth_vox : float
        Gaussian σ (voxels) on the binary occupancy before the SDF/occupancy
        step. ``0`` disables.
    sigma_vox : float
        Gaussian σ (voxels) applied to the field itself. ``0`` disables.
    pad : int
        Zero-padding (voxels) on every side so the surface closes even when the
        structure touches the volume border (needed for watertightness).

    Returns
    -------
    field : ndarray of float
        Padded scalar field (voxel grid).
    level : float
        Iso-level for marching cubes.
    pad : int
        The padding actually applied (echoed for the vertex de-offset).
    """
    mask = np.asarray(mask).astype(bool)
    if pad > 0:
        mask = np.pad(mask, pad, mode="constant", constant_values=False)

    occ = mask.astype(np.float64)
    if presmooth_vox and presmooth_vox > 0:
        occ = ndi.gaussian_filter(occ, presmooth_vox)

    if mode == "gaussian":
        field = ndi.gaussian_filter(occ, sigma_vox) if sigma_vox > 0 else occ
        return field, 0.5, pad

    if mode == "sdf":
        binm = occ >= 0.5
        if not binm.any():
            raise ValueError("Mask is empty after smoothing; cannot build an SDF.")
        d_in = ndi.distance_transform_edt(binm)
        d_out = ndi.distance_transform_edt(~binm)
        sdf = d_out - d_in  # positive outside, negative inside
        if sigma_vox and sigma_vox > 0:
            sdf = ndi.gaussian_filter(sdf, sigma_vox)
        return sdf, 0.0, pad

    raise ValueError(f"Unknown field mode {mode!r}; use 'sdf' or 'gaussian'.")


def _marching_cubes_field(
    field: np.ndarray,
    level: float,
    affine: np.ndarray,
    pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes on a scalar field → world-space ``(vertices, faces)``.

    Runs in voxel space (spacing 1), removes the padding offset, then maps to
    world coordinates through the affine — mirroring the native path so that
    anisotropic voxels and orientation are handled identically.
    """
    from skimage.measure import marching_cubes as _mc

    verts_vox, faces, _normals, _values = _mc(
        np.asarray(field, np.float64), level=level, method="lewiner"
    )
    verts_vox = verts_vox - float(pad)
    ones = np.ones((verts_vox.shape[0], 1))
    verts_world = (np.asarray(affine, float) @ np.hstack([verts_vox, ones]).T).T[:, :3]
    return (np.ascontiguousarray(verts_world, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.int64))


# ======================================================================
# §2  Mesh cleanup / smoothing / remeshing (private "improvement" steps)
# ======================================================================
def _to_trimesh(verts: np.ndarray, faces: np.ndarray) -> _trimesh_t.Trimesh:
    trimesh = _require_trimesh()
    return trimesh.Trimesh(vertices=np.asarray(verts, float),
                           faces=np.asarray(faces, np.int64), process=False)


def _largest_component(mesh: _trimesh_t.Trimesh) -> _trimesh_t.Trimesh:
    """Keep the largest connected component (by surface area)."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    areas = [float(p.area) for p in parts]
    return parts[int(np.argmax(areas))]


def _basic_clean(mesh: _trimesh_t.Trimesh) -> _trimesh_t.Trimesh:
    """Remove degenerate/duplicate faces, merge vertices, fix winding/normals."""
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def _make_watertight(mesh: _trimesh_t.Trimesh, info: dict[str, Any]) -> _trimesh_t.Trimesh:
    """Fill holes and, if needed and available, repair to a watertight 2-manifold."""
    if mesh.is_watertight:
        info["watertight_repair"] = "already_watertight"
        return mesh
    try:
        mesh.fill_holes()
    except Exception:  # pragma: no cover - trimesh version dependent
        pass
    if mesh.is_watertight:
        info["watertight_repair"] = "trimesh_fill_holes"
        return mesh
    pymeshfix = _require_pymeshfix()
    mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mf.repair(verbose=False)
    info["watertight_repair"] = "pymeshfix"
    return _to_trimesh(mf.v, mf.f)


def _taubin(mesh: _trimesh_t.Trimesh, iterations: int,
            lamb: float, nu: float) -> _trimesh_t.Trimesh:
    """Volume-preserving Taubin smoothing (trimesh)."""
    trimesh = _require_trimesh()
    return trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=nu, iterations=iterations)


def _target_n_points(area_mm2: float, edge_mm: float) -> int:
    """Vertex count for a desired mean edge length on a closed triangulation.

    ``n_faces ≈ Area / ((√3/4)·edge²)`` and ``n_verts ≈ n_faces / 2``.
    """
    tri_area = (np.sqrt(3.0) / 4.0) * edge_mm ** 2
    n_faces = area_mm2 / max(tri_area, 1e-9)
    return int(max(200, round(n_faces / 2.0)))


def _isotropic_remesh(mesh: _trimesh_t.Trimesh, n_points: int,
                      info: dict[str, Any]) -> _trimesh_t.Trimesh:
    """Isotropic remeshing via ACVD (pyacvd)."""
    pv, pyacvd = _require_pyacvd()
    pd = pv.wrap(mesh)  # trimesh -> pyvista PolyData
    clus = pyacvd.Clustering(pd)
    n_have = pd.n_points
    n_sub = 0
    # Subdivide until the point pool is dense enough to cluster uniformly.
    while clus.mesh.n_points < 3 * n_points and n_sub < 4:
        clus.subdivide(1)
        n_sub += 1
    clus.cluster(n_points)
    out = clus.create_mesh()
    faces = out.faces.reshape(-1, 4)[:, 1:]  # drop the leading '3' of each face
    info["remesh"] = f"acvd(n_points={n_points}, subdiv={n_sub}, src_pts={n_have})"
    return _basic_clean(_to_trimesh(out.points, faces))


# ======================================================================
# §3  Public API
# ======================================================================
def refine_mesh(
    vertices: Vertices,
    faces: Faces,
    *,
    closed: bool = True,
    keep_largest: bool | None = None,
    watertight: bool | None = None,
    taubin_iterations: int = 12,
    taubin_lambda: float = 0.5,
    taubin_nu: float = 0.53,
    remesh: bool | None = None,
    target_edge_mm: float = 0.7,
    n_points: int | None = None,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Improve an existing triangle mesh for Laplace–Beltrami analysis (steps 4–6).

    Applies topology cleanup, Taubin smoothing and isotropic remeshing to an
    already-triangulated surface. Use this when a mesh is available but poorly
    conditioned for the LBO (slivers, non-uniform sampling, small handles).

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    closed : bool
        Whether the surface should be a closed 2-manifold (True for anatomical
        closed structures; False for open/branching/multi-part surfaces). When
        True, defaults enable largest-component selection, watertight repair and
        remeshing; when False these default off so real anatomy is not amputated.
    keep_largest : bool, optional
        Keep only the largest connected component. Default: ``closed``.
    watertight : bool, optional
        Repair to a watertight 2-manifold. Default: ``closed``.
    taubin_iterations : int
        Number of (λ, μ) Taubin cycles. ``0`` disables smoothing.
    taubin_lambda, taubin_nu : float
        Taubin factors (shrink-free requires ``nu > lambda``; trimesh uses the
        magnitude of the negative step as ``nu``).
    remesh : bool, optional
        Apply isotropic ACVD remeshing. Default: ``closed``.
    target_edge_mm : float
        Desired mean edge length; sets the ACVD target vertex count.
    n_points : int, optional
        Explicit ACVD target vertex count (overrides ``target_edge_mm``).
    return_info : bool
        If True, also return a provenance dict.

    Returns
    -------
    (vertices, faces) or (vertices, faces, info)
    """
    trimesh = _require_trimesh()  # noqa: F841 - fail early with a clear hint

    if keep_largest is None:
        keep_largest = closed
    if watertight is None:
        watertight = closed
    if remesh is None:
        remesh = closed

    verts = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")

    info: dict[str, Any] = {"closed": closed, "keep_largest": keep_largest,
                            "watertight": watertight, "remesh": None}

    mesh = _to_trimesh(verts, faces)
    if keep_largest:
        mesh = _largest_component(mesh)
    mesh = _basic_clean(mesh)
    if watertight:
        mesh = _make_watertight(mesh, info)
        mesh = _basic_clean(mesh)

    if taubin_iterations and taubin_iterations > 0:
        mesh = _taubin(mesh, taubin_iterations, taubin_lambda, taubin_nu)

    if remesh:
        n_target = n_points or _target_n_points(float(mesh.area), target_edge_mm)
        mesh = _isotropic_remesh(mesh, n_target, info)
        if watertight:  # ACVD may open the surface at cluster poles
            mesh = _make_watertight(mesh, info)
            mesh = _basic_clean(mesh)

    out_v = np.asarray(mesh.vertices, np.float64)
    out_f = np.asarray(mesh.faces, np.int64)
    info["n_vertices"] = len(out_v)
    info["n_faces"] = len(out_f)
    logger.info("refine_mesh: %d verts, %d faces (closed=%s, remesh=%s)",
                len(out_v), len(out_f), closed, info["remesh"])
    if return_info:
        return out_v, out_f, info
    return out_v, out_f


def volume_to_mesh(
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    raw: bool = False,
    label: int | Sequence[int] | None = None,
    level: float = 0.5,
    step_size: int = 1,
    closed: bool = True,
    field_mode: FieldMode = "sdf",
    presmooth_vox: float = 0.5,
    sigma_vox: float = 0.6,
    pad: int = 4,
    return_info: bool = False,
    **refine_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract a mesh from a (label) volume, improved by default for the LBO.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z)
        Intensity or integer-label volume.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine.
    raw : bool
        If ``True``, skip all improvement and reproduce the legacy path exactly:
        plain marching cubes on the (optionally label-binarised) volume at
        ``level`` via :func:`spectralbrain.core.marching_cubes`. If ``False``
        (default), run the deterministic improvement pipeline.
    label : int or sequence of int, optional
        Restrict to these label id(s) (mask = ``np.isin(volume, label)``). When
        ``None``, the volume is thresholded at ``level`` (raw path) or its
        ``>level`` occupancy is used (improved path).
    level : float
        Iso-level / threshold for the raw path and for label-free binarisation.
    step_size : int
        Marching-cubes step for the raw path (larger = coarser/faster).
    closed : bool
        Treat the structure as a closed 2-manifold (see :func:`refine_mesh`).
        For open / branching / multi-part structures (e.g. white-matter bundle
        masks) pass ``closed=False``.
    field_mode : {"sdf", "gaussian"}
        Scalar-field construction for the improved path. Forced to
        ``"gaussian"`` when ``closed=False`` (an SDF iso-surface is ill-suited
        to open sheets).
    presmooth_vox, sigma_vox, pad :
        Field-construction parameters (see :func:`_mask_to_field`).
    return_info : bool
        If True, also return a provenance dict.
    **refine_kwargs
        Forwarded to :func:`refine_mesh` (e.g. ``taubin_iterations``,
        ``remesh``, ``target_edge_mm``, ``n_points``, ``keep_largest``,
        ``watertight``).

    Returns
    -------
    (vertices, faces) or (vertices, faces, info)

    Notes
    -----
    ``raw=False`` changes the mesh relative to plain marching cubes (denser,
    smoother, uniformly triangulated, genus-0 for closed structures). This is a
    deliberate default so that meshes fed to the LBO are well-conditioned; pass
    ``raw=True`` for byte-for-byte legacy behaviour.
    """
    volume = np.asarray(volume)
    affine = np.asarray(affine, float)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D (X, Y, Z), got shape {volume.shape}")
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be (4, 4), got {affine.shape}")

    # ---- raw / legacy path: faithful plain marching cubes -----------------
    if raw:
        field = _binarize(volume, label, level).astype(np.float64) if label is not None \
            else volume.astype(np.float64)
        lvl = 0.5 if label is not None else level
        verts, faces = _raw_marching_cubes(field, affine, level=lvl, step_size=step_size)
        info = {"method": "raw", "level": lvl, "step_size": step_size,
                "backend": "spectralbrain.core.marching_cubes"}
        if return_info:
            return verts, faces, info
        return verts, faces

    # ---- improved path ----------------------------------------------------
    mask = _binarize(volume, label, level)
    if not mask.any():
        raise ValueError("Empty mask: no voxels selected by the given label/level.")

    mode: FieldMode = "gaussian" if not closed else field_mode
    field, iso, pad_used = _mask_to_field(
        mask, mode=mode, presmooth_vox=presmooth_vox, sigma_vox=sigma_vox, pad=pad
    )
    verts, faces = _marching_cubes_field(field, iso, affine, pad_used)

    v, f, info = refine_mesh(verts, faces, closed=closed, return_info=True,
                             **refine_kwargs)
    info.update({"method": "improved", "field_mode": mode})
    if return_info:
        return v, f, info
    return v, f
