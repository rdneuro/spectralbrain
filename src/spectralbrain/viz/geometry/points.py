"""Point cloud visualization with vedo and Open3D.

This module provides publication-quality 3D renders of point clouds
used throughout SpectralBrain's atlas-free analysis pipeline.  Every
function produces a headless offscreen render and saves to PNG at
600 DPI-equivalent resolution.  An optional Open3D fallback is
available for environments without VTK.

The six main figure types cover the critical visual outputs of the
SpectralBrain point cloud pathway:

1. **Scalar scatter** — 3D point cloud coloured by a spectral
   descriptor (HKS, WKS, curvature, cluster label).
2. **MLS reconstruction** — raw cloud → smoothed → reconstructed
   surface, showing the atlas-free mesh generation pipeline.
3. **Cluster overlay** — K-means or spectral clusters with
   per-cluster PCA ellipsoids and centroids.
4. **Multi-panel comparison** — side-by-side panels comparing
   descriptors, subjects, or hemispheres.
5. **Warp / morphing** — source → target deformation field,
   useful for longitudinal or group template analyses.
6. **Voronoi diagram** — Voronoi tessellation of a projected
   point cloud, coloured by cluster or scalar.

All functions follow the SpectralBrain convention:
``(fig_or_path, metadata_dict)`` return.  For vedo-based renders
the first element is the output PNG path (a ``pathlib.Path``);
for matplotlib composites it is ``(fig, ax)`` as usual.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_DEFAULT_SIZE: Tuple[int, int] = (1600, 1200)
"""Default render window size in pixels (width, height)."""

_DEFAULT_SCALE: int = 2
"""Screenshot scale factor.  size × scale gives final PNG pixels.
At size=(1600,1200) and scale=2 → 3200×2400 px ≈ 600 DPI at ~5.3 in."""

_DEFAULT_BG: str = "white"

# Descriptor → colourmap mapping (shared with graphics.py / brainplots.py)
_SCALAR_CMAPS: Dict[str, str] = {
    "hks": "inferno",
    "wks": "cividis",
    "bks": "magma",
    "gps": "viridis",
    "shapedna": "plasma",
    "curvature": "RdBu_r",
    "cluster": "Set1",
    "z_score": "RdBu_r",
}


# ======================================================================
# §0  Lazy imports & environment setup
# ======================================================================

def _ensure_offscreen() -> None:
    """Set VTK offscreen environment variables if not already set."""
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")


def _get_vedo():
    """Lazily import vedo with offscreen configuration."""
    _ensure_offscreen()
    try:
        import vedo
        # Start Xvfb on headless Linux if needed
        try:
            vedo.start_xvfb()
        except Exception:
            pass
        return vedo
    except ImportError:
        raise ImportError(
            "vedo is required for point cloud visualization.  "
            "Install with: pip install vedo"
        )


def _get_open3d():
    """Lazily import Open3D as optional fallback."""
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None


def _resolve_cmap(scalar_name: Optional[str], cmap: Optional[str]) -> str:
    """Pick a colormap: explicit > name-based lookup > inferno fallback."""
    if cmap is not None:
        return cmap
    if scalar_name is not None:
        key = scalar_name.lower().split("_")[0]
        return _SCALAR_CMAPS.get(key, "inferno")
    return "inferno"


def _save_vedo_screenshot(
    plotter,
    save: Optional[PathLike],
    *,
    scale: int = _DEFAULT_SCALE,
) -> Path:
    """Capture a vedo Plotter to PNG and close it.

    Parameters
    ----------
    plotter : vedo.Plotter
        Active plotter (after ``.show()``).
    save : path or None
        Output path.  If None a temp file is created.
    scale : int
        Pixel multiplier for the screenshot.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    if save is None:
        fd, save = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(save), scale=scale)
    plotter.close()
    logger.info("Saved point-cloud render → %s", save)
    return save


# ======================================================================
# §1  Scalar scatter — 3D point cloud with descriptor overlay
# ======================================================================

def plot_point_cloud(
    coords: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    scalar_name: str = "HKS",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    point_size: int = 6,
    title: Optional[str] = None,
    camera: Optional[Dict[str, Any]] = None,
    show_scalarbar: bool = True,
    bg: str = _DEFAULT_BG,
    size: Tuple[int, int] = _DEFAULT_SIZE,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render a 3D point cloud coloured by a scalar descriptor.

    This is the workhorse visualisation for atlas-free analyses:
    thalamic nuclei point clouds, hippocampal point clouds, or any
    subcortical structure where no mesh connectivity is available.

    Parameters
    ----------
    coords : (N, 3) array
        Point positions in mm (RAS or scanner space).
    scalars : (N,) array or None
        Per-point scalar values for colouring.  If None the cloud
        is rendered in uniform grey.
    scalar_name : str
        Human-readable name used for the colourbar title and for
        automatic colourmap selection when *cmap* is None.
    cmap : str or None
        Matplotlib colourmap name.  None → auto from *scalar_name*.
    vmin, vmax : float or None
        Colour range limits.  None → auto from data percentiles.
    point_size : int
        Point radius in screen pixels.
    title : str or None
        Title text rendered on the image.
    camera : dict or None
        Camera config: ``{'pos', 'focal_point', 'viewup'}``.
    show_scalarbar : bool
        Whether to display a colourbar legend.
    bg : str
        Background colour.
    size : (int, int)
        Window size in pixels (width, height).
    scale : int
        Screenshot scale multiplier.
    save : path or None
        Output PNG path.  None → auto temp file.

    Returns
    -------
    (Path, dict)
        Path to the saved PNG and metadata dict with keys
        ``'n_points'``, ``'scalar_range'``, ``'cmap'``.
    """
    vedo = _get_vedo()
    coords = np.asarray(coords, dtype=np.float64)
    assert coords.ndim == 2 and coords.shape[1] == 3, (
        f"coords must be (N, 3), got {coords.shape}"
    )

    pts = vedo.Points(coords, r=point_size, c="gray", alpha=1.0)

    cmap_name = _resolve_cmap(scalar_name, cmap)
    meta: Dict[str, Any] = {
        "n_points": len(coords),
        "cmap": cmap_name,
        "scalar_range": None,
    }

    if scalars is not None:
        scalars = np.asarray(scalars, dtype=np.float64)
        assert scalars.shape[0] == coords.shape[0], (
            f"scalars length ({scalars.shape[0]}) != coords length ({coords.shape[0]})"
        )
        # Auto range from percentiles to avoid outlier dominance
        if vmin is None:
            vmin = float(np.nanpercentile(scalars, 1))
        if vmax is None:
            vmax = float(np.nanpercentile(scalars, 99))

        pts.pointdata[scalar_name] = scalars
        pts.cmap(cmap_name, scalar_name, vmin=vmin, vmax=vmax)
        if show_scalarbar:
            pts.add_scalarbar(title=scalar_name)

        meta["scalar_range"] = (vmin, vmax)

    plt = vedo.Plotter(
        offscreen=True,
        size=size,
        bg=bg,
        title=title or "",
    )

    show_kwargs: Dict[str, Any] = {"viewup": "z", "zoom": 1.2}
    if camera is not None:
        show_kwargs["camera"] = camera

    plt.show(pts, **show_kwargs)
    out = _save_vedo_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §2  MLS surface reconstruction pipeline
# ======================================================================

def plot_mls_reconstruction(
    coords: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    scalar_name: str = "HKS",
    cmap: Optional[str] = None,
    mls_factor: float = 0.2,
    recon_dims: Tuple[int, int, int] = (80, 80, 80),
    point_size: int = 4,
    bg: str = _DEFAULT_BG,
    size: Tuple[int, int] = (2400, 800),
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Three-panel pipeline: raw → MLS-smoothed → reconstructed surface.

    Demonstrates the atlas-free mesh generation pipeline used when no
    a priori mesh connectivity is available (e.g., thalamic nuclei
    from point-cloud segmentation).

    Parameters
    ----------
    coords : (N, 3) array
        Raw point cloud coordinates.
    scalars : (N,) array or None
        Optional per-point scalar for colourmap.
    scalar_name : str
        Scalar label for colourbar and cmap lookup.
    cmap : str or None
        Colourmap name (auto-resolved if None).
    mls_factor : float
        MLS smoothing factor (fraction of bounding box diagonal).
    recon_dims : (int, int, int)
        Grid resolution for Poisson surface reconstruction.
    point_size : int
        Point radius in screen pixels.
    bg : str
        Background colour.
    size : (int, int)
        Window size in pixels — wider to accommodate 3 panels.
    scale : int
        Screenshot scale factor.
    save : path or None
        Output PNG path.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'n_points'``,
        ``'n_mesh_vertices'``, ``'n_mesh_faces'``.
    """
    vedo = _get_vedo()
    coords = np.asarray(coords, dtype=np.float64)
    cmap_name = _resolve_cmap(scalar_name, cmap)

    # Panel 1: raw point cloud
    raw = vedo.Points(coords, r=point_size, c="gray")

    # Panel 2: MLS-smoothed point cloud
    smoothed = vedo.Points(coords, r=point_size, c="steelblue")
    smoothed = smoothed.smooth_mls_2d(f=mls_factor)

    # Panel 3: Poisson surface reconstruction from smoothed points
    recon = smoothed.clone().reconstruct_surface(dims=recon_dims)
    if scalars is not None:
        # Transfer scalars to reconstructed mesh via nearest-neighbour
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        recon_verts = recon.vertices
        _, idx = tree.query(recon_verts, k=1)
        interp_scalars = np.asarray(scalars, dtype=np.float64)[idx]
        recon.pointdata[scalar_name] = interp_scalars
        recon.cmap(cmap_name, scalar_name)
        recon.add_scalarbar(title=scalar_name)
    else:
        recon.color("gold")

    # Build multi-panel
    plt = vedo.Plotter(
        shape=(1, 3),
        offscreen=True,
        size=size,
        bg=bg,
    )
    plt.at(0).show(raw, title="Raw points", viewup="z", zoom=1.1)
    plt.at(1).show(smoothed, title="MLS smoothed", viewup="z", zoom=1.1)
    plt.at(2).show(recon, title="Reconstructed surface", viewup="z", zoom=1.1)

    meta = {
        "n_points": len(coords),
        "n_mesh_vertices": recon.vertices.shape[0] if recon.vertices is not None else 0,
        "n_mesh_faces": len(recon.cells) if recon.cells is not None else 0,
        "mls_factor": mls_factor,
        "recon_dims": recon_dims,
    }

    out = _save_vedo_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §3  Cluster overlay with PCA ellipsoids
# ======================================================================

def plot_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    show_ellipsoids: bool = True,
    ellipsoid_alpha: float = 0.12,
    show_centroids: bool = True,
    centroid_size: int = 14,
    cmap: str = "Set1",
    point_size: int = 5,
    title: Optional[str] = None,
    bg: str = _DEFAULT_BG,
    size: Tuple[int, int] = _DEFAULT_SIZE,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render clustered point cloud with per-cluster PCA ellipsoids.

    Useful for visualising K-means, spectral clustering, or HDBSCAN
    results on subcortical point clouds.

    Parameters
    ----------
    coords : (N, 3) array
        Point positions.
    labels : (N,) int array
        Cluster assignments (0-indexed).
    show_ellipsoids : bool
        Overlay translucent PCA ellipsoids around each cluster.
    ellipsoid_alpha : float
        Ellipsoid transparency (0 = invisible, 1 = opaque).
    show_centroids : bool
        Mark cluster centroids with large dots.
    centroid_size : int
        Centroid marker size in pixels.
    cmap : str
        Categorical colourmap for cluster colouring.
    point_size : int
        Point radius in pixels.
    title : str or None
        Figure title.
    bg, size, scale, save
        Standard rendering parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'n_clusters'``,
        ``'cluster_sizes'``, ``'cluster_centroids'``.
    """
    vedo = _get_vedo()
    from vedo.pointcloud.fits import pca_ellipsoid

    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    unique_labels = np.unique(labels[labels >= 0])  # skip noise label -1
    n_clusters = len(unique_labels)

    # Build coloured point cloud
    pts = vedo.Points(coords, r=point_size)
    pts.pointdata["cluster"] = labels.astype(float)
    pts.cmap(cmap, "cluster")

    actors = [pts]

    # Per-cluster ellipsoids and centroids
    cluster_sizes = {}
    centroids = {}
    for k in unique_labels:
        mask = labels == k
        cluster_sizes[int(k)] = int(mask.sum())
        cluster_coords = coords[mask]
        centroids[int(k)] = cluster_coords.mean(axis=0).tolist()

        if show_ellipsoids and mask.sum() >= 4:
            try:
                ell = pca_ellipsoid(vedo.Points(cluster_coords))
                ell.alpha(ellipsoid_alpha)
                actors.append(ell)
            except Exception as exc:
                logger.warning("PCA ellipsoid failed for cluster %d: %s", k, exc)

        if show_centroids:
            centroid = vedo.Points(
                cluster_coords.mean(axis=0, keepdims=True),
                r=centroid_size,
                c="black",
            )
            actors.append(centroid)

    plt = vedo.Plotter(
        offscreen=True,
        size=size,
        bg=bg,
        title=title or f"{n_clusters} clusters",
    )
    plt.show(*actors, viewup="z", zoom=1.2)

    meta = {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "cluster_centroids": centroids,
    }

    out = _save_vedo_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §4  Multi-panel comparison
# ======================================================================

def plot_point_cloud_panel(
    panels: List[Dict[str, Any]],
    *,
    shape: Optional[Tuple[int, int]] = None,
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Multi-panel point cloud comparison.

    Each panel is a dict with keys:

    - ``'coords'`` : (N, 3) array (required)
    - ``'scalars'`` : (N,) array or None
    - ``'scalar_name'`` : str (default ``'value'``)
    - ``'cmap'`` : str or None
    - ``'vmin'``, ``'vmax'`` : float or None
    - ``'point_size'`` : int (default 5)
    - ``'title'`` : str (default ``''``)

    Parameters
    ----------
    panels : list of dict
        One dict per panel.
    shape : (rows, cols) or None
        Grid layout.  None → single row.
    bg : str
        Background colour.
    size : (int, int) or None
        Total window size.  None → 800px × n_cols by 800px × n_rows.
    scale : int
        Screenshot scale.
    save : path or None
        Output PNG path.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'n_panels'``.
    """
    vedo = _get_vedo()
    n = len(panels)
    if shape is None:
        shape = (1, n)
    if size is None:
        size = (800 * shape[1], 800 * shape[0])

    plt = vedo.Plotter(
        shape=shape,
        offscreen=True,
        size=size,
        bg=bg,
    )

    for i, panel in enumerate(panels):
        coords = np.asarray(panel["coords"], dtype=np.float64)
        scalars = panel.get("scalars")
        scalar_name = panel.get("scalar_name", "value")
        cmap_name = _resolve_cmap(scalar_name, panel.get("cmap"))
        point_size = panel.get("point_size", 5)
        panel_title = panel.get("title", "")

        pts = vedo.Points(coords, r=point_size, c="gray")
        if scalars is not None:
            scalars = np.asarray(scalars, dtype=np.float64)
            v0 = panel.get("vmin") or float(np.nanpercentile(scalars, 1))
            v1 = panel.get("vmax") or float(np.nanpercentile(scalars, 99))
            pts.pointdata[scalar_name] = scalars
            pts.cmap(cmap_name, scalar_name, vmin=v0, vmax=v1)
            pts.add_scalarbar(title=scalar_name)

        plt.at(i).show(pts, title=panel_title, viewup="z", zoom=1.1)

    meta = {"n_panels": n, "shape": shape}
    out = _save_vedo_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §5  Warp / morphing between two point clouds
# ======================================================================

def plot_warp(
    source: np.ndarray,
    target: np.ndarray,
    sigma: float = 1.0,
    *,
    show_displacement: bool = True,
    source_color: str = "steelblue",
    target_color: str = "tomato",
    warped_color: str = "gold",
    point_size: int = 5,
    arrow_scale: float = 0.3,
    title: Optional[str] = None,
    bg: str = _DEFAULT_BG,
    size: Tuple[int, int] = (2400, 800),
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Visualise thin-plate spline warp between two point clouds.

    Three panels: source cloud, target cloud, and warped result
    with displacement arrows overlaid.

    Parameters
    ----------
    source : (N, 3) array
        Source (reference) point cloud.
    target : (N, 3) array
        Target point cloud — must have same N as source for
        the TPS warp to be meaningful.
    sigma : float
        TPS stiffness parameter.
    show_displacement : bool
        Overlay arrows from source to warped positions.
    source_color, target_color, warped_color : str
        Point cloud colours for each panel.
    point_size : int
        Point size in pixels.
    arrow_scale : float
        Arrow length multiplier.
    title : str or None
        Figure title.
    bg, size, scale, save
        Standard rendering parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'mean_displacement'``,
        ``'max_displacement'``.
    """
    vedo = _get_vedo()
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert source.shape == target.shape, (
        f"source and target shapes must match: {source.shape} vs {target.shape}"
    )

    src_pts = vedo.Points(source, r=point_size, c=source_color)
    tgt_pts = vedo.Points(target, r=point_size, c=target_color)

    # Apply thin-plate spline warp
    warped_pts = src_pts.clone().warp(source, target, sigma=sigma)
    warped_pts.color(warped_color)

    # Compute displacement vectors for metadata
    warped_coords = warped_pts.vertices
    disp = np.linalg.norm(warped_coords - source, axis=1)

    actors_warped = [warped_pts]
    if show_displacement:
        # Draw arrows from source to warped positions
        arrows = vedo.Arrows(
            source, warped_coords,
            c="black", alpha=0.4, s=arrow_scale,
        )
        actors_warped.append(arrows)

    plt = vedo.Plotter(
        shape=(1, 3),
        offscreen=True,
        size=size,
        bg=bg,
    )
    plt.at(0).show(src_pts, title="Source", viewup="z", zoom=1.1)
    plt.at(1).show(tgt_pts, title="Target", viewup="z", zoom=1.1)
    plt.at(2).show(*actors_warped, title="Warped", viewup="z", zoom=1.1)

    meta = {
        "n_points": len(source),
        "sigma": sigma,
        "mean_displacement": float(np.mean(disp)),
        "max_displacement": float(np.max(disp)),
    }

    out = _save_vedo_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §6  Voronoi diagram
# ======================================================================

def plot_voronoi(
    coords: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    scalar_name: str = "cluster",
    cmap: Optional[str] = None,
    *,
    projection: Literal["xy", "xz", "yz"] = "xy",
    padding: float = 0.1,
    wireframe_color: str = "black",
    wireframe_width: int = 1,
    point_size: int = 8,
    bg: str = _DEFAULT_BG,
    size: Tuple[int, int] = _DEFAULT_SIZE,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Voronoi tessellation of a point cloud projected onto a 2D plane.

    Particularly useful for spatial domain analysis of thalamic nuclei
    or hippocampal subfield parcellations in the unfolded space.

    Parameters
    ----------
    coords : (N, 3) or (N, 2) array
        Point positions.  3D points are projected onto *projection*.
    scalars : (N,) array or None
        Per-point values for cell colouring.
    scalar_name : str
        Scalar label for colourbar and cmap lookup.
    cmap : str or None
        Colourmap name.
    projection : {'xy', 'xz', 'yz'}
        Projection plane for 3D → 2D.
    padding : float
        Voronoi cell boundary padding.
    wireframe_color : str
        Cell boundary line colour.
    wireframe_width : int
        Cell boundary line width.
    point_size : int
        Overlay point size.
    bg, size, scale, save
        Standard rendering parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata with ``'n_cells'``.
    """
    vedo = _get_vedo()
    coords = np.asarray(coords, dtype=np.float64)

    # Project 3D → 2D if necessary
    if coords.shape[1] == 3:
        proj_map = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
        axes = proj_map[projection]
        coords_2d = np.column_stack([
            coords[:, axes[0]],
            coords[:, axes[1]],
            np.zeros(len(coords)),
        ])
    elif coords.shape[1] == 2:
        coords_2d = np.column_stack([coords, np.zeros(len(coords))])
    else:
        raise ValueError(f"coords must be (N, 2) or (N, 3), got {coords.shape}")

    pts = vedo.Points(coords_2d, r=point_size, c="black")

    # Generate Voronoi cells
    voronoi = pts.generate_voronoi(padding=padding)
    voronoi.wireframe().linewidth(wireframe_width).color(wireframe_color)

    cmap_name = _resolve_cmap(scalar_name, cmap)
    if scalars is not None:
        scalars = np.asarray(scalars, dtype=np.float64)
        voronoi.pointdata[scalar_name] = scalars
        voronoi.cmap(cmap_name, scalar_name)
        voronoi.add_scalarbar(title=scalar_name)

    plt = vedo.Plotter(offscreen=True, size=size, bg=bg)
    plt.show(voronoi, pts, zoom=1.1)

    meta = {
        "n_points": len(coords),
        "n_cells": len(voronoi.cells) if voronoi.cells is not None else 0,
        "projection": projection,
    }

    out = _save_vedo_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §7  Open3D fallback — basic point cloud render
# ======================================================================

def plot_point_cloud_o3d(
    coords: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    cmap: str = "inferno",
    point_size: float = 2.0,
    width: int = 1600,
    height: int = 1200,
    save: Optional[PathLike] = None,
) -> Optional[Path]:
    """Minimal Open3D point cloud render (fallback when vedo unavailable).

    Parameters
    ----------
    coords : (N, 3) array
        Point positions.
    scalars : (N,) array or None
        Per-point scalar for colourmap.
    cmap : str
        Matplotlib colourmap name.
    point_size : float
        Point size.
    width, height : int
        Image dimensions.
    save : path or None
        Output PNG path.

    Returns
    -------
    Path or None
        Output path if successful, None otherwise.
    """
    o3d = _get_open3d()
    if o3d is None:
        logger.warning("Open3D not available — cannot render point cloud")
        return None

    from matplotlib import cm

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(coords, dtype=np.float64))

    if scalars is not None:
        scalars = np.asarray(scalars, dtype=np.float64)
        norm = (scalars - scalars.min()) / (scalars.max() - scalars.min() + 1e-12)
        colormap = cm.get_cmap(cmap)
        colors = colormap(norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([1.0, 1.0, 1.0])

    vis.poll_events()
    vis.update_renderer()

    if save is None:
        fd, save = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    save = Path(save)
    vis.capture_screen_image(str(save))
    vis.destroy_window()

    logger.info("Saved Open3D render → %s", save)
    return save


# ======================================================================
# __all__
# ======================================================================

__all__ = [
    # Constants
    "_DEFAULT_SIZE",
    "_DEFAULT_SCALE",
    "_SCALAR_CMAPS",
    # Core renders
    "plot_point_cloud",
    "plot_mls_reconstruction",
    "plot_clusters",
    "plot_point_cloud_panel",
    "plot_warp",
    "plot_voronoi",
    # Open3D fallback
    "plot_point_cloud_o3d",
]
