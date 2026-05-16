"""Clustering visualisation — 3D mesh renders and 2D statistical plots.

Provides publication-quality figures for every output of
:mod:`spectralbrain.statistics.clustering`: spatial cluster maps,
GNMF components, persistence landscapes, temporal profiles, method
comparisons, Bayesian confirmation diagnostics, and fused-descriptor
panels.

Figure types
------------
**3D mesh renders (vedo)**

1. Cluster map — mesh coloured by integer labels, 3-pose panel
2. Cluster boundaries — wireframe with coloured boundary edges
3. Multi-method comparison — side-by-side cluster maps
4. GNMF spatial components — one panel per W column
5. Soft membership — mesh coloured by posterior probability
6. Exploded clusters — spatially separated cluster fragments
7. HKS + clusters progression — scalar + labels across t
8. Persistence basins — mesh coloured by persistence-based partition
9. Fusion panel — HKS / WKS / Fused side by side

**2D statistical plots (matplotlib)**

10. Cluster HKS time-profiles — mean ± SEM per cluster
11. Silhouette diagram — per-sample silhouette ordered by cluster
12. Cluster quality comparison — bar chart across methods
13. Method agreement heatmap — ARI / NMI matrix
14. Persistence diagram — birth vs death scatter
15. GNMF temporal factors — F matrix as line profiles
16. Bayesian confirmation — posterior probabilities + credible intervals
17. Cluster size distribution — bar chart
18. UMAP / PCA scatter — embedding coloured by clusters
19. Co-clustering checkerboard — vertex × time block structure

Architecture
------------
* **vedo** for all 3D renders (offscreen VTK → PNG).
* **matplotlib** for all 2D plots (publication style via graphics.py).
* 3D functions return ``(Path, dict)`` — PNG path + metadata.
* 2D functions return ``(Figure, Axes)`` for customisation.
* Every function accepts ``save`` for auto-export.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from spectralbrain.runtime import PathLike, get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_SIZE: Tuple[int, int] = (1600, 1200)
_DEFAULT_SCALE: int = 2
_DEFAULT_BG: str = "white"
DPI: int = 600

# Qualitative palette for cluster labels — optimised for
# colorblind safety (Paul Tol's muted scheme + extensions)
CLUSTER_COLORS: List[str] = [
    "#4477AA",   # blue
    "#EE6677",   # rose
    "#228833",   # green
    "#CCBB44",   # sand
    "#66CCEE",   # cyan
    "#AA3377",   # purple
    "#EE8866",   # orange
    "#44AA99",   # teal
    "#332288",   # indigo
    "#CC6677",   # wine
    "#882255",   # plum
    "#117733",   # forest
    "#999933",   # olive
    "#DDCC77",   # wheat
]

# Standard 3-pose views for brain structures
VIEWS_3POSE: List[str] = ["left_lateral", "anterior", "superior"]

# Camera presets — identical to geometry/meshes.py for consistency
CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    "anterior":      {"azimuth": 0,   "elevation": 0},
    "posterior":      {"azimuth": 180, "elevation": 0},
    "left_lateral":   {"azimuth": -90, "elevation": 0},
    "right_lateral":  {"azimuth": 90,  "elevation": 0},
    "superior":       {"azimuth": 0,   "elevation": 90},
    "inferior":       {"azimuth": 0,   "elevation": -90},
    "left_medial":    {"azimuth": 90,  "elevation": 0},
    "right_medial":   {"azimuth": -90, "elevation": 0},
    "oblique_left":   {"azimuth": -45, "elevation": 30},
    "oblique_right":  {"azimuth": 45,  "elevation": 30},
}


# ──────────────────────────────────────────────────────────────────────
# Lazy imports & helpers
# ──────────────────────────────────────────────────────────────────────

def _ensure_offscreen() -> None:
    os.environ.setdefault("VTK_USE_OFFSCREEN", "1")


def _get_vedo():
    _ensure_offscreen()
    try:
        import vedo
        try:
            vedo.start_xvfb()
        except Exception:
            pass
        return vedo
    except ImportError:
        raise ImportError(
            "vedo is required for 3D cluster visualization.  "
            "Install with: pip install vedo"
        )


def _build_vedo_mesh(vertices, faces, vedo_module):
    """Construct a vedo Mesh from numpy arrays."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    cells = np.column_stack([np.full(F.shape[0], 3, dtype=np.int64), F])
    mesh = vedo_module.Mesh([V, cells])
    return mesh


def _save_screenshot(plotter, save, *, scale=_DEFAULT_SCALE):
    """Capture a vedo Plotter to PNG and close it."""
    if save is None:
        fd, save = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(save), scale=scale)
    plotter.close()
    logger.info("Saved cluster render → %s", save)
    return save


def _cluster_cmap(n_clusters: int):
    """Build a ListedColormap from the cluster palette."""
    colors = CLUSTER_COLORS[:n_clusters] if n_clusters <= len(CLUSTER_COLORS) \
        else (CLUSTER_COLORS * ((n_clusters // len(CLUSTER_COLORS)) + 1))[:n_clusters]
    return mcolors.ListedColormap(colors)


def _apply_style():
    """Apply SpectralBrain publication style."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except ImportError:
        pass
    plt.rcParams.update({
        "savefig.dpi": DPI, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "font.size": 9,
    })


def _savefig(fig: Figure, save: Optional[PathLike]) -> None:
    """Save matplotlib figure in PNG + PDF."""
    if save is not None:
        p = Path(save)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=DPI, bbox_inches="tight",
                    facecolor="white", transparent=False)
        # also save PDF if extension is png
        if p.suffix.lower() == ".png":
            fig.savefig(str(p.with_suffix(".pdf")),
                        bbox_inches="tight", facecolor="white")
        logger.info("Saved figure → %s", p)


# ======================================================================
# §1  3D CLUSTER MAP — mesh coloured by labels, 3-pose panel
# ======================================================================

def plot_cluster_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    *,
    views: Optional[List[str]] = None,
    noise_color: str = "lightgray",
    lighting: str = "default",
    show_scalarbar: bool = True,
    title: Optional[str] = None,
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render mesh coloured by cluster labels in a multi-view panel.

    Each cluster gets a distinct colour from the colorblind-safe
    palette.  Noise vertices (label = -1) are rendered in grey.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    labels : (V,) int array
        Cluster labels.  -1 = noise / unassigned.
    views : list of str or None
        Camera preset names.  None → 3-pose (lateral, anterior, superior).
    noise_color : str
        Colour for noise vertices.
    lighting : str
    show_scalarbar : bool
    title : str or None
    bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
        PNG path and metadata.
    """
    vedo = _get_vedo()
    labels = np.asarray(labels, dtype=np.int64)
    if views is None:
        views = VIEWS_3POSE
    n_views = len(views)
    if size is None:
        size = (600 * n_views, 600)

    # --- build RGBA per vertex ---
    unique_labels = sorted(set(labels[labels >= 0]))
    n_clusters = len(unique_labels)
    cmap = _cluster_cmap(n_clusters)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    rgba = np.zeros((len(labels), 4), dtype=np.float64)
    for i, lab in enumerate(labels):
        if lab < 0:
            rgba[i] = mcolors.to_rgba(noise_color)
        else:
            rgba[i] = cmap(label_to_idx[lab])

    # vedo expects (V, 4) uint8 for vertex colours
    rgba_u8 = (rgba * 255).astype(np.uint8)

    plt = vedo.Plotter(
        shape=(1, n_views), offscreen=True, size=size, bg=bg,
    )

    for vi, view_name in enumerate(views):
        mesh = _build_vedo_mesh(vertices, faces, vedo)
        mesh.pointdata["ClusterRGBA"] = rgba_u8
        mesh.pointdata.select("ClusterRGBA")
        mesh.lighting(lighting)

        preset = CAMERA_PRESETS.get(view_name, {})
        plt.at(vi).show(
            mesh,
            title=view_name.replace("_", " ").title() if not title else title,
            viewup="z", zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

    meta = {
        "n_clusters": n_clusters,
        "n_noise": int((labels < 0).sum()),
        "views": views,
        "cluster_colors": {lab: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                           for i, lab in enumerate(unique_labels)},
    }
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §2  CLUSTER BOUNDARIES — mesh with highlighted boundary edges
# ======================================================================

def plot_cluster_boundaries(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    *,
    mesh_color: str = "ivory",
    mesh_alpha: float = 0.6,
    boundary_width: float = 3.0,
    views: Optional[List[str]] = None,
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render mesh with cluster boundaries as coloured lines.

    Identifies edges where adjacent triangles have different cluster
    labels and renders them as coloured tubes on a semi-transparent
    surface.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    labels : (V,) int array
    mesh_color : str
        Base mesh colour.
    mesh_alpha : float
        Base mesh opacity.
    boundary_width : float
        Width of boundary lines.
    views, lighting, bg, size, scale, save
        Standard render parameters.

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()
    labels = np.asarray(labels, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    verts = np.asarray(vertices, dtype=np.float64)

    if views is None:
        views = VIEWS_3POSE
    n_views = len(views)
    if size is None:
        size = (600 * n_views, 600)

    # --- find boundary edges ---
    edges_set = set()
    boundary_edges = []
    for f in faces:
        for e in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]:
            e_sorted = tuple(sorted(e))
            if e_sorted in edges_set:
                continue
            edges_set.add(e_sorted)
            if labels[e[0]] != labels[e[1]] and labels[e[0]] >= 0 and labels[e[1]] >= 0:
                boundary_edges.append(e_sorted)

    n_boundary = len(boundary_edges)
    logger.info("Found %d boundary edges between clusters.", n_boundary)

    # --- build line segments ---
    if boundary_edges:
        pts_list = []
        for e in boundary_edges:
            pts_list.append([verts[e[0]], verts[e[1]]])
        lines = vedo.Lines(pts_list, c="red", lw=boundary_width)
    else:
        lines = None

    plt_obj = vedo.Plotter(
        shape=(1, n_views), offscreen=True, size=size, bg=bg,
    )

    for vi, view_name in enumerate(views):
        mesh = _build_vedo_mesh(verts, faces, vedo)
        mesh.color(mesh_color).alpha(mesh_alpha).lighting(lighting)

        actors = [mesh]
        if lines is not None:
            actors.append(lines.clone())

        preset = CAMERA_PRESETS.get(view_name, {})
        plt_obj.at(vi).show(
            *actors,
            title=view_name.replace("_", " ").title(),
            viewup="z", zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

    meta = {"n_boundary_edges": n_boundary, "views": views}
    out = _save_screenshot(plt_obj, save, scale=scale)
    return out, meta


# ======================================================================
# §3  MULTI-METHOD COMPARISON — side-by-side cluster maps
# ======================================================================

def plot_method_comparison_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    results: Dict[str, np.ndarray],
    *,
    view: str = "left_lateral",
    noise_color: str = "lightgray",
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Compare multiple clustering methods on the same mesh.

    Each method gets one panel, all from the same camera angle.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    results : dict[str, ndarray]
        Method name → (V,) label array.
    view : str
        Camera preset for all panels.
    noise_color, lighting, bg, size, scale, save
        Standard parameters.

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()
    methods = list(results.keys())
    n = len(methods)
    if size is None:
        size = (600 * n, 600)

    plt = vedo.Plotter(shape=(1, n), offscreen=True, size=size, bg=bg)
    preset = CAMERA_PRESETS.get(view, {})

    for i, method in enumerate(methods):
        lab = np.asarray(results[method], dtype=np.int64)
        unique = sorted(set(lab[lab >= 0]))
        n_clust = len(unique)
        cmap = _cluster_cmap(n_clust)
        lab_to_idx = {l: j for j, l in enumerate(unique)}

        rgba = np.zeros((len(lab), 4), dtype=np.float64)
        for vi, l in enumerate(lab):
            rgba[vi] = mcolors.to_rgba(noise_color) if l < 0 else cmap(lab_to_idx[l])
        rgba_u8 = (rgba * 255).astype(np.uint8)

        mesh = _build_vedo_mesh(vertices, faces, vedo)
        mesh.pointdata["ClusterRGBA"] = rgba_u8
        mesh.pointdata.select("ClusterRGBA")
        mesh.lighting(lighting)

        plt.at(i).show(
            mesh,
            title=f"{method} (k={n_clust})",
            viewup="z", zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

    meta = {"methods": methods, "view": view}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §4  GNMF SPATIAL COMPONENTS — one panel per W column
# ======================================================================

def plot_gnmf_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    W: np.ndarray,
    *,
    cmap: str = "inferno",
    max_components: int = 8,
    view: str = "left_lateral",
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render GNMF spatial factor columns as separate mesh panels.

    Each column of W represents the soft membership weight for one
    spatial component.  Displayed as scalar overlays on the mesh.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    W : (V, K) array
        Spatial factors from cluster_gnmf.
    cmap : str
    max_components : int
        Show at most this many components.
    view, lighting, bg, size, scale, save
        Standard parameters.

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()
    W = np.asarray(W, dtype=np.float64)
    K = min(W.shape[1], max_components)

    n_cols = min(K, 4)
    n_rows = (K + n_cols - 1) // n_cols
    if size is None:
        size = (500 * n_cols, 500 * n_rows)

    plt = vedo.Plotter(
        shape=(n_rows, n_cols), offscreen=True, size=size, bg=bg,
    )
    preset = CAMERA_PRESETS.get(view, {})

    for k in range(K):
        row, col = divmod(k, n_cols)
        mesh = _build_vedo_mesh(vertices, faces, vedo)
        scalars = W[:, k]
        vmin = float(np.nanpercentile(scalars, 1))
        vmax = float(np.nanpercentile(scalars, 99))
        mesh.pointdata[f"W_{k}"] = scalars
        mesh.cmap(cmap, f"W_{k}", vmin=vmin, vmax=vmax)
        mesh.add_scalarbar(title=f"Component {k}")
        mesh.lighting(lighting)

        plt.at(row * n_cols + col).show(
            mesh, title=f"W[:, {k}]", viewup="z", zoom=1.1,
            **{kk: v for kk, v in preset.items() if kk in ("azimuth", "elevation")},
        )

    meta = {"n_components": K, "view": view}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §5  SOFT MEMBERSHIP — mesh coloured by probability
# ======================================================================

def plot_soft_membership(
    vertices: np.ndarray,
    faces: np.ndarray,
    probabilities: np.ndarray,
    cluster_idx: int = 0,
    *,
    cmap: str = "YlOrRd",
    views: Optional[List[str]] = None,
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render soft membership probability for one cluster.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    probabilities : (V, K) array
        Soft membership matrix.
    cluster_idx : int
        Which cluster's probability to display.
    cmap, views, lighting, bg, size, scale, save
        Standard parameters.

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()
    P = np.asarray(probabilities, dtype=np.float64)
    scalars = P[:, cluster_idx]

    if views is None:
        views = VIEWS_3POSE
    n_views = len(views)
    if size is None:
        size = (600 * n_views, 600)

    plt = vedo.Plotter(shape=(1, n_views), offscreen=True, size=size, bg=bg)

    for vi, view_name in enumerate(views):
        mesh = _build_vedo_mesh(vertices, faces, vedo)
        mesh.pointdata["P"] = scalars
        mesh.cmap(cmap, "P", vmin=0.0, vmax=1.0)
        mesh.add_scalarbar(title=f"P(cluster={cluster_idx})")
        mesh.lighting(lighting)

        preset = CAMERA_PRESETS.get(view_name, {})
        plt.at(vi).show(
            mesh, title=view_name.replace("_", " ").title(),
            viewup="z", zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

    meta = {"cluster_idx": cluster_idx}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §6  EXPLODED CLUSTERS — spatially separated fragments
# ======================================================================

def plot_cluster_exploded(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    *,
    explosion_factor: float = 1.5,
    view: str = "oblique_left",
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Tuple[int, int] = (1600, 1200),
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Exploded view — each cluster is displaced outward from centroid.

    Useful for inspecting cluster topology on convoluted structures
    like the hippocampus where clusters may overlap visually.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    labels : (V,) int array
    explosion_factor : float
        Distance multiplier for displacement. 1.0 = in place.
    view, lighting, bg, size, scale, save
        Standard parameters.

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()
    labels = np.asarray(labels, dtype=np.int64)
    verts = np.asarray(vertices, dtype=np.float64)
    fcs = np.asarray(faces, dtype=np.int64)

    unique_labels = sorted(set(labels[labels >= 0]))
    n_clusters = len(unique_labels)
    cmap = _cluster_cmap(n_clusters)
    lab_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # global centroid
    global_center = verts.mean(axis=0)

    actors = []
    for lab in unique_labels:
        mask = labels == lab
        vert_idx = np.where(mask)[0]

        # remap vertex indices for the sub-mesh
        idx_map = {old: new for new, old in enumerate(vert_idx)}

        sub_faces = []
        for f in fcs:
            if mask[f[0]] and mask[f[1]] and mask[f[2]]:
                sub_faces.append([idx_map[f[0]], idx_map[f[1]], idx_map[f[2]]])

        if not sub_faces:
            continue

        sub_verts = verts[vert_idx].copy()
        sub_faces_arr = np.array(sub_faces, dtype=np.int64)

        # displacement vector
        cluster_center = sub_verts.mean(axis=0)
        direction = cluster_center - global_center
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
        displacement = direction * norm * explosion_factor

        sub_verts += displacement

        mesh = _build_vedo_mesh(sub_verts, sub_faces_arr, vedo)
        color = CLUSTER_COLORS[lab_to_idx[lab] % len(CLUSTER_COLORS)]
        mesh.color(color).lighting(lighting)
        actors.append(mesh)

    preset = CAMERA_PRESETS.get(view, {})
    plt = vedo.Plotter(offscreen=True, size=size, bg=bg)
    plt.show(
        *actors,
        title="Exploded Cluster View",
        viewup="z", zoom=0.9,
        **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
    )

    meta = {"n_clusters": n_clusters, "explosion_factor": explosion_factor}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §7  HKS + CLUSTERS PROGRESSION — scalar + labels across t
# ======================================================================

def plot_hks_cluster_progression(
    vertices: np.ndarray,
    faces: np.ndarray,
    H: np.ndarray,
    labels: np.ndarray,
    t_indices: Optional[List[int]] = None,
    *,
    n_panels: int = 4,
    view: str = "left_lateral",
    cmap_hks: str = "inferno",
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Show HKS scalar at selected time-scales alongside cluster map.

    Top row: HKS at different t.  Bottom row: cluster labels.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    H : (V, T) array
        HKS matrix.
    labels : (V,) int array
    t_indices : list of int or None
        Column indices into H to show.  None → linearly spaced.
    n_panels : int
        Number of time-scale panels.
    view, cmap_hks, lighting, bg, size, scale, save
        Standard parameters.

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()
    H = np.asarray(H, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    T = H.shape[1]
    if t_indices is None:
        t_indices = np.linspace(0, T - 1, n_panels, dtype=int).tolist()
    n_panels = len(t_indices)

    if size is None:
        size = (500 * n_panels, 1000)

    # 2 rows: top = HKS, bottom = clusters
    plt = vedo.Plotter(
        shape=(2, n_panels), offscreen=True, size=size, bg=bg,
    )
    preset = CAMERA_PRESETS.get(view, {})

    unique = sorted(set(labels[labels >= 0]))
    n_clusters = len(unique)
    ccmap = _cluster_cmap(n_clusters)
    lab_to_idx = {l: i for i, l in enumerate(unique)}

    # cluster RGBA
    rgba = np.zeros((len(labels), 4), dtype=np.float64)
    for i, l in enumerate(labels):
        rgba[i] = mcolors.to_rgba("lightgray") if l < 0 else ccmap(lab_to_idx[l])
    rgba_u8 = (rgba * 255).astype(np.uint8)

    for pi, ti in enumerate(t_indices):
        # top row: HKS
        mesh_hks = _build_vedo_mesh(vertices, faces, vedo)
        sc = H[:, ti]
        v0 = float(np.nanpercentile(sc, 1))
        v1 = float(np.nanpercentile(sc, 99))
        mesh_hks.pointdata["HKS"] = sc
        mesh_hks.cmap(cmap_hks, "HKS", vmin=v0, vmax=v1)
        mesh_hks.add_scalarbar(title=f"t={ti}")
        mesh_hks.lighting(lighting)

        plt.at(0 * n_panels + pi).show(
            mesh_hks, title=f"HKS t[{ti}]", viewup="z", zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

        # bottom row: clusters
        mesh_cl = _build_vedo_mesh(vertices, faces, vedo)
        mesh_cl.pointdata["ClusterRGBA"] = rgba_u8
        mesh_cl.pointdata.select("ClusterRGBA")
        mesh_cl.lighting(lighting)

        plt.at(1 * n_panels + pi).show(
            mesh_cl, title="Clusters", viewup="z", zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

    meta = {"t_indices": t_indices, "n_panels": n_panels}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §8  FUSION PANEL — HKS / WKS / Fused side by side
# ======================================================================

def plot_fusion_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    hks_scalar: np.ndarray,
    wks_scalar: np.ndarray,
    fused_scalar: np.ndarray,
    *,
    cmap_hks: str = "inferno",
    cmap_wks: str = "cividis",
    cmap_fused: str = "magma",
    view: str = "left_lateral",
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Tuple[int, int] = (1800, 600),
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Side-by-side render of HKS, WKS, and fused descriptor.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    hks_scalar, wks_scalar, fused_scalar : (V,) arrays
        Single-scale scalars for each descriptor type.
    cmap_hks, cmap_wks, cmap_fused : str
    view, lighting, bg, size, scale, save

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()
    preset = CAMERA_PRESETS.get(view, {})
    plt = vedo.Plotter(shape=(1, 3), offscreen=True, size=size, bg=bg)

    for pi, (sc, name, cm) in enumerate([
        (hks_scalar, "HKS", cmap_hks),
        (wks_scalar, "WKS", cmap_wks),
        (fused_scalar, "Fused", cmap_fused),
    ]):
        sc = np.asarray(sc, dtype=np.float64)
        v0 = float(np.nanpercentile(sc, 1))
        v1 = float(np.nanpercentile(sc, 99))
        mesh = _build_vedo_mesh(vertices, faces, vedo)
        mesh.pointdata[name] = sc
        mesh.cmap(cm, name, vmin=v0, vmax=v1)
        mesh.add_scalarbar(title=name)
        mesh.lighting(lighting)

        plt.at(pi).show(
            mesh, title=name, viewup="z", zoom=1.1,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

    meta = {"view": view}
    out = _save_screenshot(plt, save, scale=scale)
    return out, meta


# ======================================================================
# §10  CLUSTER HKS TIME-PROFILES — mean ± SEM per cluster
# ======================================================================

def plot_cluster_profiles(
    H: np.ndarray,
    labels: np.ndarray,
    t_values: Optional[np.ndarray] = None,
    *,
    log_t: bool = True,
    show_sem: bool = True,
    title: str = "Cluster HKS Profiles",
    xlabel: str = "Diffusion time t",
    ylabel: str = "HKS(x, t)",
    figsize: Tuple[float, float] = (7, 4),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Mean ± SEM HKS profiles per cluster.

    Parameters
    ----------
    H : (N, T) array
    labels : (N,) int array
    t_values : (T,) array or None
    log_t : bool
        Use log-scale on x-axis.
    show_sem : bool
        Show shaded SEM bands.
    title, xlabel, ylabel, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    H = np.asarray(H, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    T = H.shape[1]
    if t_values is None:
        t_values = np.arange(T, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)

    unique = sorted(set(labels[labels >= 0]))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lab in enumerate(unique):
        mask = labels == lab
        cluster_h = H[mask]
        mean = cluster_h.mean(axis=0)
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        if log_t:
            ax.plot(t_values, mean, color=color, label=f"Cluster {lab}",
                    linewidth=1.8)
        else:
            ax.plot(t_values, mean, color=color, label=f"Cluster {lab}",
                    linewidth=1.8)

        if show_sem and mask.sum() > 1:
            sem = cluster_h.std(axis=0) / np.sqrt(mask.sum())
            ax.fill_between(t_values, mean - sem, mean + sem,
                            color=color, alpha=0.2)

    if log_t:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=min(len(unique), 4))

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §11  SILHOUETTE DIAGRAM
# ======================================================================

def plot_silhouette_diagram(
    H: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
    title: str = "Silhouette Diagram",
    figsize: Tuple[float, float] = (6, 5),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Per-sample silhouette plot ordered by cluster.

    Parameters
    ----------
    H : (N, T) array or (N, N) precomputed distance
    labels : (N,) int array
    metric : str
    title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    _apply_style()
    labels = np.asarray(labels, dtype=np.int64)
    valid = labels >= 0
    H_v = H[valid]
    lab_v = labels[valid]

    sil_vals = silhouette_samples(H_v, lab_v, metric=metric)
    avg_sil = silhouette_score(H_v, lab_v, metric=metric)

    unique = sorted(set(lab_v))
    fig, ax = plt.subplots(figsize=figsize)

    y_lower = 0
    for i, lab in enumerate(unique):
        cluster_sil = np.sort(sil_vals[lab_v == lab])
        cluster_size = len(cluster_sil)
        y_upper = y_lower + cluster_size

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.barh(
            range(y_lower, y_upper),
            cluster_sil,
            height=1.0,
            color=color,
            edgecolor="none",
            label=f"Cluster {lab}",
        )
        # label centroid
        ax.text(-0.05, y_lower + 0.5 * cluster_size,
                str(lab), fontsize=8, va="center", ha="right")
        y_lower = y_upper + 2  # gap between clusters

    ax.axvline(avg_sil, color="red", linestyle="--", linewidth=1,
               label=f"Mean = {avg_sil:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Vertices (sorted by cluster)")
    ax.set_title(title)
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §12  CLUSTER QUALITY COMPARISON — bar chart across methods
# ======================================================================

def plot_quality_comparison(
    quality_dict: Dict[str, Dict[str, float]],
    *,
    metrics: Optional[List[str]] = None,
    title: str = "Clustering Quality Comparison",
    figsize: Tuple[float, float] = (8, 4),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Grouped bar chart comparing quality metrics across methods.

    Parameters
    ----------
    quality_dict : dict[str, dict[str, float]]
        Outer key = method name, inner dict = metric → value.
        Example: ``{"hdbscan": {"silhouette": 0.42}, ...}``
    metrics : list of str or None
        Which metrics to plot.  None → all common metrics.
    title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    methods = list(quality_dict.keys())

    if metrics is None:
        all_keys = set()
        for v in quality_dict.values():
            all_keys.update(v.keys())
        metrics = sorted(all_keys)

    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=figsize)
    for i, method in enumerate(methods):
        vals = [quality_dict[method].get(m, 0.0) for m in metrics]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.bar(x + i * width, vals, width, label=method, color=color)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=7)

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §13  METHOD AGREEMENT HEATMAP — ARI / NMI matrix
# ======================================================================

def plot_agreement_heatmap(
    agreement_matrix: np.ndarray,
    method_names: List[str],
    *,
    metric_name: str = "ARI",
    cmap: str = "YlGnBu",
    title: str = "Inter-Method Agreement",
    figsize: Tuple[float, float] = (6, 5),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Heatmap of pairwise clustering agreement.

    Parameters
    ----------
    agreement_matrix : (M, M) array
        Pairwise ARI or NMI scores.
    method_names : list of str
    metric_name : str
    cmap, title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(agreement_matrix, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(method_names, fontsize=8)

    # annotate cells
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            ax.text(j, i, f"{agreement_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if agreement_matrix[i, j] > 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name, fontsize=9)
    ax.set_title(title)

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §14  PERSISTENCE DIAGRAM — birth vs death scatter
# ======================================================================

def plot_persistence_diagram(
    diagram: np.ndarray,
    *,
    title: str = "Persistence Diagram",
    figsize: Tuple[float, float] = (5, 5),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Birth-death scatter for persistence-based clustering.

    Parameters
    ----------
    diagram : (n_pairs, 2) array
        Each row is (birth, death).
    title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    diagram = np.asarray(diagram, dtype=np.float64)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistence = deaths - births

    fig, ax = plt.subplots(figsize=figsize)

    # diagonal
    lims = [min(births.min(), deaths.min()) - 0.1,
            max(births.max(), deaths.max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.5)

    # colour by persistence
    sc = ax.scatter(births, deaths, c=persistence, cmap="plasma",
                    s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="Persistence")

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §15  GNMF TEMPORAL FACTORS — F matrix profiles
# ======================================================================

def plot_gnmf_temporal_factors(
    F: np.ndarray,
    t_values: Optional[np.ndarray] = None,
    *,
    log_t: bool = True,
    title: str = "GNMF Temporal Factors",
    figsize: Tuple[float, float] = (7, 4),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Plot rows of the GNMF F matrix as temporal profiles.

    Each row of F is a canonical HKS-like curve for one component.

    Parameters
    ----------
    F : (K, T) array
    t_values : (T,) array or None
    log_t : bool
    title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    F = np.asarray(F, dtype=np.float64)
    K, T = F.shape
    if t_values is None:
        t_values = np.arange(T, dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(K):
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.plot(t_values, F[k], color=color, linewidth=1.5,
                label=f"Component {k}")

    if log_t:
        ax.set_xscale("log")
    ax.set_xlabel("Diffusion time t")
    ax.set_ylabel("F(t)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=min(K, 4))

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §16  BAYESIAN CONFIRMATION — posteriors + credible intervals
# ======================================================================

def plot_bayesian_confirmation(
    label_probabilities: np.ndarray,
    credible_intervals: Dict[int, Dict[str, Any]],
    agreement_ari: float,
    *,
    title: str = "Bayesian Cluster Confirmation",
    figsize: Tuple[float, float] = (10, 4),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Two-panel Bayesian confirmation diagnostic.

    Left: stacked area of posterior membership probabilities.
    Right: credible intervals (94% HDI) of cluster centroid norms.

    Parameters
    ----------
    label_probabilities : (N, K) array
    credible_intervals : dict[int, dict]
        Per-cluster HDI summaries from confirm_clusters_bayesian.
    agreement_ari : float
    title, figsize, save

    Returns
    -------
    (Figure, (Axes, Axes))
    """
    _apply_style()
    P = np.asarray(label_probabilities, dtype=np.float64)
    N, K = P.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: sorted probability distribution ---
    # sort vertices by max-probability cluster for visual clarity
    max_cluster = P.argmax(axis=1)
    max_prob = P.max(axis=1)
    order = np.lexsort((max_prob, max_cluster))
    P_sorted = P[order]

    bottom = np.zeros(N)
    for k in range(K):
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax1.fill_between(range(N), bottom, bottom + P_sorted[:, k],
                         color=color, alpha=0.8, label=f"Cl {k}")
        bottom += P_sorted[:, k]

    ax1.set_xlabel("Vertices (sorted)")
    ax1.set_ylabel("P(cluster)")
    ax1.set_title(f"Posterior Membership (ARI={agreement_ari:.3f})")
    ax1.set_xlim(0, N)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=6, ncol=min(K, 4), loc="lower left")

    # --- Right: credible intervals ---
    clusters = sorted(credible_intervals.keys())
    y_pos = np.arange(len(clusters))
    means = []
    lows = []
    highs = []

    for k in clusters:
        ci = credible_intervals[k]
        # norm of centroid mean as a summary scalar
        m = np.linalg.norm(ci["mean"])
        lo = np.linalg.norm(ci["hdi_3"])
        hi = np.linalg.norm(ci["hdi_97"])
        means.append(m)
        lows.append(lo)
        highs.append(hi)

    means = np.array(means)
    lows = np.array(lows)
    highs = np.array(highs)

    for i, k in enumerate(clusters):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax2.plot([lows[i], highs[i]], [y_pos[i], y_pos[i]],
                 color=color, linewidth=2.5, solid_capstyle="round")
        ax2.plot(means[i], y_pos[i], "o", color=color, markersize=6)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Cluster {k}" for k in clusters], fontsize=8)
    ax2.set_xlabel("‖μ_k‖ (centroid norm)")
    ax2.set_title("94% Credible Intervals")
    ax2.invert_yaxis()

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    _savefig(fig, save)
    return fig, (ax1, ax2)


# ======================================================================
# §17  CLUSTER SIZE DISTRIBUTION
# ======================================================================

def plot_cluster_sizes(
    labels: np.ndarray,
    *,
    title: str = "Cluster Size Distribution",
    figsize: Tuple[float, float] = (6, 3.5),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Bar chart of cluster sizes with noise count.

    Parameters
    ----------
    labels : (N,) int array
    title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    labels = np.asarray(labels, dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (lab, cnt) in enumerate(zip(unique, counts)):
        if lab < 0:
            color = "lightgray"
            label = f"Noise (n={cnt})"
        else:
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            label = f"Cluster {lab}"
        ax.bar(i, cnt, color=color, edgecolor="k", linewidth=0.3)
        ax.text(i, cnt + max(counts) * 0.01, str(cnt),
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(
        ["Noise" if l < 0 else str(l) for l in unique],
        fontsize=8,
    )
    ax.set_ylabel("Number of vertices")
    ax.set_title(title)

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §18  UMAP / PCA SCATTER — embedding coloured by clusters
# ======================================================================

def plot_cluster_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    method_name: str = "UMAP",
    noise_color: str = "lightgray",
    noise_alpha: float = 0.3,
    point_size: float = 3.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """2D scatter of dimensionality-reduced embedding coloured by cluster.

    Parameters
    ----------
    embedding : (N, 2) array
    labels : (N,) int array
    method_name : str
        For axis labels (e.g., "UMAP", "PCA", "t-SNE").
    noise_color, noise_alpha, point_size
    title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    embedding = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    fig, ax = plt.subplots(figsize=figsize)

    # noise first (background)
    noise = labels < 0
    if noise.any():
        ax.scatter(embedding[noise, 0], embedding[noise, 1],
                   c=noise_color, alpha=noise_alpha, s=point_size,
                   label="Noise", rasterized=True)

    unique = sorted(set(labels[labels >= 0]))
    for i, lab in enumerate(unique):
        mask = labels == lab
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=point_size, alpha=0.7,
                   label=f"Cluster {lab}", rasterized=True)

    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(title or f"Cluster Map in {method_name} Space")
    ax.legend(fontsize=7, markerscale=3, ncol=min(len(unique) + 1, 4))

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §19  CO-CLUSTERING CHECKERBOARD — vertex × time block structure
# ======================================================================

def plot_coclustering_heatmap(
    H: np.ndarray,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    *,
    cmap: str = "viridis",
    title: str = "Co-Clustering Structure",
    figsize: Tuple[float, float] = (8, 6),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Reordered heatmap revealing vertex × time co-cluster blocks.

    Sorts rows and columns by their cluster labels so that the
    checkerboard block structure of the co-clustering is visible.

    Parameters
    ----------
    H : (N, T) array
    row_labels : (N,) int array
        Vertex cluster labels.
    col_labels : (T,) int array
        Time/scale cluster labels.
    cmap, title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    H = np.asarray(H, dtype=np.float64)
    row_labels = np.asarray(row_labels, dtype=np.int64)
    col_labels = np.asarray(col_labels, dtype=np.int64)

    # sort rows and columns by cluster label
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)
    H_sorted = H[row_order][:, col_order]

    # apply log for visual dynamic range
    H_vis = np.log(H_sorted + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(H_vis, aspect="auto", cmap=cmap, interpolation="none")

    # draw block boundaries
    row_sorted_labels = row_labels[row_order]
    col_sorted_labels = col_labels[col_order]

    for i in range(1, len(row_sorted_labels)):
        if row_sorted_labels[i] != row_sorted_labels[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(1, len(col_sorted_labels)):
        if col_sorted_labels[j] != col_sorted_labels[j - 1]:
            ax.axvline(j - 0.5, color="white", linewidth=0.5)

    fig.colorbar(im, ax=ax, label="log(descriptor value)", shrink=0.8)
    ax.set_xlabel("Time/Energy scale (sorted by cluster)")
    ax.set_ylabel("Vertices (sorted by cluster)")
    ax.set_title(title)

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §20  SUMMARY PANEL — comprehensive overview figure
# ======================================================================

def plot_cluster_summary(
    vertices: np.ndarray,
    faces: np.ndarray,
    H: np.ndarray,
    labels: np.ndarray,
    t_values: Optional[np.ndarray] = None,
    *,
    method_name: str = "GNMF",
    figsize: Tuple[float, float] = (14, 10),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, np.ndarray]:
    """Comprehensive 2×3 summary panel for a clustering result.

    Layout:
        [0,0] 3D cluster map (embedded PNG)
        [0,1] Cluster sizes bar chart
        [0,2] UMAP scatter coloured by cluster
        [1,0] HKS profiles per cluster
        [1,1] Silhouette diagram
        [1,2] Persistence diagram (if available)

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) array
    H : (V, T) array
    labels : (V,) int array
    t_values : (T,) or None
    method_name : str
    figsize, save

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    _apply_style()
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # --- [0,0] 3D render as embedded image ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_path = tf.name
    try:
        plot_cluster_map(vertices, faces, labels,
                         views=["left_lateral"], save=tmp_path)
        img = mpimg.imread(tmp_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"{method_name} Cluster Map")
        axes[0, 0].axis("off")
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f"3D render failed:\n{e}",
                        ha="center", va="center", fontsize=8,
                        transform=axes[0, 0].transAxes)
        axes[0, 0].axis("off")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # --- [0,1] Cluster sizes ---
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
              for i in range(len(unique))]
    axes[0, 1].bar(range(len(unique)), counts, color=colors,
                    edgecolor="k", linewidth=0.3)
    axes[0, 1].set_xticks(range(len(unique)))
    axes[0, 1].set_xticklabels([str(u) for u in unique], fontsize=7)
    axes[0, 1].set_ylabel("n vertices")
    axes[0, 1].set_title("Cluster Sizes")

    # --- [0,2] UMAP scatter ---
    try:
        import umap as umap_mod
        X = np.log(H + 1e-12)
        embedding = umap_mod.UMAP(
            n_components=2, random_state=42
        ).fit_transform(X)
        for i, lab in enumerate(sorted(set(labels[labels >= 0]))):
            mask = labels == lab
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            axes[0, 2].scatter(embedding[mask, 0], embedding[mask, 1],
                               c=color, s=2, alpha=0.5, rasterized=True)
        noise = labels < 0
        if noise.any():
            axes[0, 2].scatter(embedding[noise, 0], embedding[noise, 1],
                               c="lightgray", s=1, alpha=0.2, rasterized=True)
        axes[0, 2].set_title("UMAP Embedding")
        axes[0, 2].set_xlabel("UMAP 1")
        axes[0, 2].set_ylabel("UMAP 2")
    except ImportError:
        from sklearn.decomposition import PCA
        X = np.log(H + 1e-12)
        embedding = PCA(n_components=2).fit_transform(X)
        for i, lab in enumerate(sorted(set(labels[labels >= 0]))):
            mask = labels == lab
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            axes[0, 2].scatter(embedding[mask, 0], embedding[mask, 1],
                               c=color, s=2, alpha=0.5, rasterized=True)
        axes[0, 2].set_title("PCA Embedding")
        axes[0, 2].set_xlabel("PC 1")
        axes[0, 2].set_ylabel("PC 2")

    # --- [1,0] HKS profiles per cluster ---
    if t_values is None:
        t_vals = np.arange(H.shape[1], dtype=np.float64)
    else:
        t_vals = np.asarray(t_values)

    for i, lab in enumerate(sorted(set(labels[labels >= 0]))):
        mask = labels == lab
        mean_h = H[mask].mean(axis=0)
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        axes[1, 0].plot(t_vals, mean_h, color=color, linewidth=1.2,
                        label=f"Cl {lab}")
        if mask.sum() > 1:
            sem = H[mask].std(axis=0) / np.sqrt(mask.sum())
            axes[1, 0].fill_between(t_vals, mean_h - sem, mean_h + sem,
                                    color=color, alpha=0.15)

    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("HKS")
    axes[1, 0].set_title("Cluster HKS Profiles")
    axes[1, 0].legend(fontsize=6, ncol=min(len(unique), 4))

    # --- [1,1] Silhouette ---
    try:
        from sklearn.metrics import silhouette_samples
        valid = labels >= 0
        sil = silhouette_samples(H[valid], labels[valid])
        y_lower = 0
        for i, lab in enumerate(sorted(set(labels[labels >= 0]))):
            cl_sil = np.sort(sil[labels[valid] == lab])
            y_upper = y_lower + len(cl_sil)
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            axes[1, 1].barh(range(y_lower, y_upper), cl_sil,
                            height=1.0, color=color, edgecolor="none")
            y_lower = y_upper + 2
        axes[1, 1].set_yticks([])
        axes[1, 1].set_xlabel("Silhouette")
        axes[1, 1].set_title("Silhouette Diagram")
    except Exception:
        axes[1, 1].text(0.5, 0.5, "Could not compute silhouette",
                        ha="center", va="center", fontsize=8,
                        transform=axes[1, 1].transAxes)

    # --- [1,2] Empty or placeholder ---
    axes[1, 2].text(0.5, 0.5, f"Method: {method_name}\nk = {len(unique)}",
                    ha="center", va="center", fontsize=12,
                    transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[1, 2].axis("off")

    fig.suptitle(f"Clustering Summary — {method_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, save)
    return fig, axes


# ======================================================================
# §21  SPATIO-TEMPORAL FIELD — small multiples on unfolded coordinates
# ======================================================================

def plot_spatiotemporal_field(
    unfolded_coords: np.ndarray,
    faces: np.ndarray,
    H: np.ndarray,
    t_values: Optional[np.ndarray] = None,
    *,
    n_panels: int = 8,
    t_indices: Optional[List[int]] = None,
    cmap: str = "magma",
    log_norm: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    subfield_labels: Optional[np.ndarray] = None,
    boundary_color: str = "k",
    boundary_width: float = 0.6,
    descriptor_name: str = "HKS",
    xlabel: str = "AP coordinate",
    ylabel: str = "PD coordinate",
    n_cols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, np.ndarray]:
    """Small-multiples grid of a descriptor field on unfolded surface.

    Renders per-vertex spectral descriptor values (HKS, WKS) on the
    2D unfolded coordinate system (e.g., HippUnfold's AP × PD sheet)
    at log-spaced diffusion times, producing a publication-ready
    panel that shows the local-to-global progression of HKS.

    Each panel is rendered with ``tripcolor`` directly on the mesh
    triangulation — no interpolation to a regular grid is needed,
    preserving vertex-exact values.

    Parameters
    ----------
    unfolded_coords : (V, 2) array
        Per-vertex 2D coordinates in unfolded space.
        Column 0 = AP (or u), column 1 = PD (or v).
    faces : (F, 3) array
        Triangle indices (same mesh topology as the folded surface).
    H : (V, T) array
        Per-vertex descriptor matrix across T scales.
    t_values : (T,) array or None
        Scale parameter values for axis labels. None → integer indices.
    n_panels : int
        Number of panels to show (overridden by t_indices).
    t_indices : list of int or None
        Specific column indices into H. None → geometrically spaced.
    cmap : str
        Colourmap for the descriptor field.
    log_norm : bool
        Use ``LogNorm`` for colour scaling (recommended for HKS).
    vmin, vmax : float or None
        Colour range. None → robust 2nd/98th percentiles of full H.
    subfield_labels : (V,) int array or None
        If provided, overlay subfield boundaries via ``tricontour``.
    boundary_color : str
        Colour for subfield boundary lines.
    boundary_width : float
        Width of boundary lines.
    descriptor_name : str
        Label for colourbar (e.g. ``"HKS"``, ``"WKS"``).
    xlabel, ylabel : str
        Axis labels for the unfolded coordinate axes.
    n_cols : int
        Number of columns in the panel grid.
    figsize : tuple or None
        Figure size. None → auto from n_panels.
    save : PathLike or None

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    _apply_style()
    import matplotlib.tri as mtri
    from matplotlib.colors import LogNorm, Normalize

    H = np.asarray(H, dtype=np.float64)
    uv = np.asarray(unfolded_coords, dtype=np.float64)
    fcs = np.asarray(faces, dtype=np.int64)
    V, T = H.shape

    # --- select scale indices ---
    if t_indices is not None:
        sel = list(t_indices)
    else:
        # geometrically spaced indices for log-spaced t
        sel = np.unique(
            np.geomspace(1, T, n_panels, dtype=int).clip(0, T - 1)
        ).tolist()
    n_panels = len(sel)

    if t_values is None:
        t_values = np.arange(T, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)

    # --- build triangulation ---
    tri = mtri.Triangulation(uv[:, 0], uv[:, 1], fcs)

    # --- colour normalisation (shared across all panels) ---
    if vmin is None:
        vmin = float(np.nanpercentile(H[:, sel], 2))
    if vmax is None:
        vmax = float(np.nanpercentile(H[:, sel], 98))

    if log_norm:
        vmin_safe = max(vmin, 1e-12)
        norm = LogNorm(vmin=vmin_safe, vmax=max(vmax, vmin_safe * 2))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # --- layout ---
    n_rows = (n_panels + n_cols - 1) // n_cols
    if figsize is None:
        # 2:1 aspect ratio per panel (AP is typically ~2× PD)
        figsize = (4.0 * n_cols, 2.5 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    # --- render each panel ---
    tpc = None
    for pi, t_idx in enumerate(sel):
        ax = axes_flat[pi]
        scalar = H[:, t_idx]

        tpc = ax.tripcolor(
            tri, scalar,
            shading="gouraud",
            cmap=cmap,
            norm=norm,
            rasterized=True,  # keeps PDF/SVG file sizes small
        )

        # subfield boundaries
        if subfield_labels is not None:
            labels_f = np.asarray(subfield_labels, dtype=np.float64)
            ax.tricontour(
                tri, labels_f,
                levels=np.arange(labels_f.max()) + 0.5,
                colors=boundary_color,
                linewidths=boundary_width,
            )

        # scale label
        t_val = t_values[t_idx]
        ax.set_title(f"t = {t_val:.2g}", fontsize=8)
        ax.set_aspect("equal")

    # hide unused panels
    for pi in range(n_panels, len(axes_flat)):
        axes_flat[pi].set_visible(False)

    # axis labels on edge panels only
    for ax in axes_flat:
        if ax.get_visible():
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

    # shared colourbar
    if tpc is not None:
        cbar = fig.colorbar(
            tpc, ax=axes_flat[:n_panels].tolist(),
            location="right", shrink=0.8, pad=0.02,
        )
        cbar.set_label(
            f"{descriptor_name} ({'log scale' if log_norm else 'linear'})",
            fontsize=8,
        )
        cbar.ax.tick_params(labelsize=6)

    _savefig(fig, save)
    return fig, axes


# ======================================================================
# §22  SPATIO-TEMPORAL ANIMATION — GIF / MP4 across scales
# ======================================================================

def plot_spatiotemporal_animation(
    unfolded_coords: np.ndarray,
    faces: np.ndarray,
    H: np.ndarray,
    t_values: Optional[np.ndarray] = None,
    *,
    cmap: str = "magma",
    log_norm: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    subfield_labels: Optional[np.ndarray] = None,
    descriptor_name: str = "HKS",
    fps: int = 8,
    figsize: Tuple[float, float] = (6, 3.5),
    save: Optional[PathLike] = None,
) -> Optional[Any]:
    """Animate descriptor field across scales on the unfolded surface.

    Produces a GIF or MP4 (determined by file extension of ``save``)
    showing how HKS/WKS evolves from fine-scale (local curvature)
    to coarse-scale (global topology) as the diffusion time increases.

    Uses ``FuncAnimation`` with a single ``tripcolor`` artist and
    ``set_array`` updates for efficient rendering.

    Parameters
    ----------
    unfolded_coords : (V, 2) array
    faces : (F, 3) array
    H : (V, T) array
    t_values : (T,) array or None
    cmap : str
    log_norm : bool
    vmin, vmax : float or None
    subfield_labels : (V,) int array or None
    descriptor_name : str
    fps : int
        Frames per second for the output animation.
    figsize : tuple
    save : PathLike or None
        Output path. Extension determines format:
        ``.gif`` → Pillow writer, ``.mp4`` → ffmpeg writer.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        The animation object (for notebook display). None if save-only.
    """
    import matplotlib.tri as mtri
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm, Normalize

    _apply_style()

    H = np.asarray(H, dtype=np.float64)
    uv = np.asarray(unfolded_coords, dtype=np.float64)
    fcs = np.asarray(faces, dtype=np.int64)
    V, T = H.shape

    if t_values is None:
        t_values = np.arange(T, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)

    tri = mtri.Triangulation(uv[:, 0], uv[:, 1], fcs)

    # --- normalisation ---
    if vmin is None:
        vmin = float(np.nanpercentile(H, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(H, 98))

    if log_norm:
        vmin_safe = max(vmin, 1e-12)
        norm = LogNorm(vmin=vmin_safe, vmax=max(vmax, vmin_safe * 2))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # --- setup figure with first frame ---
    fig, ax = plt.subplots(figsize=figsize)
    tpc = ax.tripcolor(
        tri, H[:, 0], shading="gouraud",
        cmap=cmap, norm=norm, rasterized=True,
    )
    fig.colorbar(tpc, ax=ax, label=descriptor_name, shrink=0.8)

    # static subfield boundaries (drawn once)
    if subfield_labels is not None:
        labels_f = np.asarray(subfield_labels, dtype=np.float64)
        ax.tricontour(
            tri, labels_f,
            levels=np.arange(labels_f.max()) + 0.5,
            colors="k", linewidths=0.5,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("AP coordinate", fontsize=8)
    ax.set_ylabel("PD coordinate", fontsize=8)
    title_text = ax.set_title(
        f"{descriptor_name}  t = {t_values[0]:.2g}", fontsize=9,
    )

    # --- animation update ---
    def _update(frame_idx):
        # tripcolor stores face-averaged values for flat shading
        # and vertex values for gouraud — set_array on the collection
        tpc.set_array(H[:, frame_idx])
        title_text.set_text(
            f"{descriptor_name}  t = {t_values[frame_idx]:.2g}"
        )
        return (tpc, title_text)

    ani = animation.FuncAnimation(
        fig, _update, frames=T,
        interval=1000 // fps,
        blit=False,  # tripcolor + blit can produce blank frames
    )

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        ext = save.suffix.lower()
        if ext == ".gif":
            ani.save(str(save), writer="pillow", fps=fps, dpi=200)
        elif ext in (".mp4", ".avi", ".mov"):
            ani.save(str(save), writer="ffmpeg", fps=fps, dpi=200)
        else:
            ani.save(str(save), fps=fps, dpi=200)
        logger.info("Saved animation → %s", save)

    return ani


# ======================================================================
# §23  HOVMÖLLER DIAGRAM — position × scale 2D heatmap
# ======================================================================

def plot_hovmoller(
    unfolded_coords: np.ndarray,
    H: np.ndarray,
    t_values: Optional[np.ndarray] = None,
    *,
    axis: Literal["AP", "PD"] = "AP",
    n_bins: int = 100,
    cmap: str = "viridis",
    log_norm: bool = True,
    log_t: bool = True,
    descriptor_name: str = "HKS",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 4),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Hovmöller diagram: averaged descriptor along one spatial axis × scale.

    Collapses the orthogonal spatial axis by averaging, producing a
    2D heatmap of (position along AP or PD) × (diffusion scale t).
    Reveals how the multi-scale spectral signature varies along the
    hippocampal long axis (AP) or proximal-distal axis (PD).

    Parameters
    ----------
    unfolded_coords : (V, 2) array
        Column 0 = AP, column 1 = PD.
    H : (V, T) array
        Per-vertex descriptor matrix.
    t_values : (T,) array or None
    axis : str
        Which spatial axis to retain: ``"AP"`` (column 0) or ``"PD"``
        (column 1). The other is averaged out.
    n_bins : int
        Number of bins along the retained spatial axis.
    cmap : str
    log_norm : bool
    log_t : bool
        Log-scale the t-axis.
    descriptor_name : str
    title : str or None
    figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    from matplotlib.colors import LogNorm, Normalize

    H = np.asarray(H, dtype=np.float64)
    uv = np.asarray(unfolded_coords, dtype=np.float64)
    V, T = H.shape

    if t_values is None:
        t_values = np.arange(T, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)

    # --- select spatial coordinate ---
    col_idx = 0 if axis == "AP" else 1
    pos = uv[:, col_idx]

    # --- bin vertices along the chosen axis ---
    bin_edges = np.linspace(pos.min(), pos.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    digitized = np.digitize(pos, bin_edges) - 1
    digitized = np.clip(digitized, 0, n_bins - 1)

    # average H within each spatial bin → (n_bins, T)
    H_binned = np.zeros((n_bins, T), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    for i in range(V):
        b = digitized[i]
        H_binned[b] += H[i]
        counts[b] += 1.0
    counts[counts == 0] = 1.0
    H_binned /= counts[:, None]

    # --- normalisation ---
    vmin = float(np.nanpercentile(H_binned[H_binned > 0], 2)) if log_norm else float(H_binned.min())
    vmax = float(np.nanpercentile(H_binned, 98))
    if log_norm:
        norm = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 2))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    mesh_plot = ax.pcolormesh(
        t_values, bin_centers, H_binned,
        shading="auto", cmap=cmap, norm=norm, rasterized=True,
    )

    if log_t:
        ax.set_xscale("log")

    ax.set_xlabel("Diffusion scale t (log)" if log_t else "Scale t")
    ax.set_ylabel(f"{axis} coordinate")
    ax.set_title(title or f"Hovmöller — {descriptor_name} along {axis}")

    cbar = fig.colorbar(mesh_plot, ax=ax, pad=0.02)
    cbar.set_label(descriptor_name)

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §24  KYMOGRAPH — 1D line through the surface × scale
# ======================================================================

def plot_kymograph(
    unfolded_coords: np.ndarray,
    faces: np.ndarray,
    H: np.ndarray,
    t_values: Optional[np.ndarray] = None,
    *,
    line_axis: Literal["AP", "PD"] = "AP",
    line_position: float = 0.5,
    n_samples: int = 200,
    cmap: str = "viridis",
    log_norm: bool = True,
    log_t: bool = True,
    descriptor_name: str = "HKS",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 4),
    save: Optional[PathLike] = None,
) -> Tuple[Figure, Axes]:
    """Kymograph: descriptor values along a 1D line × scale.

    Unlike the Hovmöller diagram (which averages over the orthogonal
    axis), the kymograph traces a single line through the unfolded
    surface — e.g. the midline of PD — and plots the descriptor
    along that line for each scale t.

    Parameters
    ----------
    unfolded_coords : (V, 2) array
    faces : (F, 3) array
    H : (V, T) array
    t_values : (T,) array or None
    line_axis : str
        Axis along which the line runs. ``"AP"`` → horizontal line
        at fixed PD = ``line_position``; ``"PD"`` → vertical line
        at fixed AP.
    line_position : float
        Position on the orthogonal axis (0–1 in normalised coords).
    n_samples : int
        Number of sample points along the line.
    cmap, log_norm, log_t : str, bool, bool
    descriptor_name, title, figsize, save

    Returns
    -------
    (Figure, Axes)
    """
    _apply_style()
    from scipy.interpolate import LinearNDInterpolator
    from matplotlib.colors import LogNorm, Normalize

    H = np.asarray(H, dtype=np.float64)
    uv = np.asarray(unfolded_coords, dtype=np.float64)
    V, T = H.shape

    if t_values is None:
        t_values = np.arange(T, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)

    # --- build interpolator (Delaunay computed once) ---
    interp = LinearNDInterpolator(uv, H[:, 0])

    # --- define the 1D line ---
    u_range = uv[:, 0]
    v_range = uv[:, 1]

    if line_axis == "AP":
        line_u = np.linspace(u_range.min(), u_range.max(), n_samples)
        line_v = np.full_like(line_u, line_position)
        spatial_coord = line_u
        spatial_label = "AP coordinate"
    else:
        line_v = np.linspace(v_range.min(), v_range.max(), n_samples)
        line_u = np.full_like(line_v, line_position)
        spatial_coord = line_v
        spatial_label = "PD coordinate"

    line_pts = np.column_stack([line_u, line_v])

    # --- interpolate each scale ---
    kymo = np.empty((n_samples, T), dtype=np.float64)
    for k in range(T):
        # reuse the same Delaunay but update values
        interp.values = H[:, k:k + 1]
        kymo[:, k] = interp(line_pts).squeeze()

    # fill NaN from outside convex hull with nearest valid
    for k in range(T):
        nans = np.isnan(kymo[:, k])
        if nans.any() and not nans.all():
            valid = ~nans
            kymo[nans, k] = np.interp(
                np.where(nans)[0], np.where(valid)[0], kymo[valid, k],
            )

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    vmin_val = float(np.nanpercentile(kymo[kymo > 0], 2)) if log_norm else float(np.nanmin(kymo))
    vmax_val = float(np.nanpercentile(kymo, 98))
    if log_norm:
        norm = LogNorm(vmin=max(vmin_val, 1e-12), vmax=max(vmax_val, vmin_val * 2))
    else:
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)

    mesh_plot = ax.pcolormesh(
        t_values, spatial_coord, kymo,
        shading="auto", cmap=cmap, norm=norm, rasterized=True,
    )

    if log_t:
        ax.set_xscale("log")

    ortho_name = "PD" if line_axis == "AP" else "AP"
    ax.set_xlabel("Diffusion scale t")
    ax.set_ylabel(spatial_label)
    ax.set_title(
        title or f"Kymograph — {descriptor_name} along {line_axis} "
                 f"at {ortho_name}={line_position:.2f}"
    )

    cbar = fig.colorbar(mesh_plot, ax=ax, pad=0.02)
    cbar.set_label(descriptor_name)

    fig.tight_layout()
    _savefig(fig, save)
    return fig, ax


# ======================================================================
# §25  WARPED SURFACE — 3D with height = descriptor value (vedo)
# ======================================================================

def plot_warped_surface(
    unfolded_coords: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    *,
    warp_factor: float = 0.5,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    descriptor_name: str = "HKS",
    views: Optional[List[str]] = None,
    lighting: str = "default",
    bg: str = _DEFAULT_BG,
    size: Optional[Tuple[int, int]] = None,
    scale: int = _DEFAULT_SCALE,
    save: Optional[PathLike] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """3D warped surface: unfolded (u, v) + height from descriptor.

    Creates a 3D surface where x = AP, y = PD, and z = warp_factor ×
    descriptor_value. The surface is coloured by the same descriptor.
    Useful as a "hero figure" showing the spatial distribution of
    a spectral descriptor.

    Parameters
    ----------
    unfolded_coords : (V, 2) array
    faces : (F, 3) array
    scalars : (V,) array
        Descriptor values to warp by and colour with.
    warp_factor : float
        Height scaling factor.
    cmap : str
    vmin, vmax : float or None
    descriptor_name : str
    views : list of str or None
        Camera presets. None → ``["oblique_left"]``.
    lighting, bg, size, scale, save

    Returns
    -------
    (Path, dict)
    """
    vedo = _get_vedo()

    uv = np.asarray(unfolded_coords, dtype=np.float64)
    fcs = np.asarray(faces, dtype=np.int64)
    sc = np.asarray(scalars, dtype=np.float64)

    # normalise scalars for z-displacement
    sc_norm = sc - sc.min()
    sc_max = sc_norm.max()
    if sc_max > 0:
        sc_norm /= sc_max

    # build 3D vertices: (u, v, warp_factor * normalised_scalar)
    verts_3d = np.column_stack([
        uv[:, 0],
        uv[:, 1],
        warp_factor * sc_norm,
    ])

    if views is None:
        views = ["oblique_left"]
    n_views = len(views)
    if size is None:
        size = (600 * n_views, 600)

    if vmin is None:
        vmin = float(np.nanpercentile(sc, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(sc, 99))

    plt_obj = vedo.Plotter(
        shape=(1, n_views), offscreen=True, size=size, bg=bg,
    )

    for vi, view_name in enumerate(views):
        mesh = _build_vedo_mesh(verts_3d, fcs, vedo)
        mesh.pointdata[descriptor_name] = sc
        mesh.cmap(cmap, descriptor_name, vmin=vmin, vmax=vmax)
        mesh.add_scalarbar(title=descriptor_name)
        mesh.lighting(lighting)

        preset = CAMERA_PRESETS.get(view_name, {"azimuth": -45, "elevation": 30})
        plt_obj.at(vi).show(
            mesh,
            title=f"{descriptor_name} (warped)",
            viewup="z", zoom=1.0,
            **{k: v for k, v in preset.items() if k in ("azimuth", "elevation")},
        )

    meta = {
        "warp_factor": warp_factor,
        "scalar_range": (vmin, vmax),
        "views": views,
    }
    out = _save_screenshot(plt_obj, save, scale=scale)
    return out, meta


# ======================================================================
# §26  DESCRIPTOR EVOLUTION COMPARISON — HKS vs WKS on unfolded
# ======================================================================

def plot_descriptor_evolution_comparison(
    unfolded_coords: np.ndarray,
    faces: np.ndarray,
    H_hks: np.ndarray,
    H_wks: np.ndarray,
    t_values_hks: Optional[np.ndarray] = None,
    e_values_wks: Optional[np.ndarray] = None,
    *,
    n_scales: int = 4,
    cmap_hks: str = "inferno",
    cmap_wks: str = "cividis",
    log_norm: bool = True,
    subfield_labels: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[PathLike] = None,
) -> Tuple[Figure, np.ndarray]:
    """Side-by-side HKS vs WKS evolution on the unfolded surface.

    Two-row layout: top = HKS at selected t, bottom = WKS at matched
    energies. Each with its own colourmap and independent colourbar.

    Parameters
    ----------
    unfolded_coords : (V, 2) array
    faces : (F, 3) array
    H_hks : (V, T_h) array
    H_wks : (V, T_w) array
    t_values_hks : (T_h,) or None
    e_values_wks : (T_w,) or None
    n_scales : int
        Number of scale panels per descriptor.
    cmap_hks, cmap_wks : str
    log_norm : bool
    subfield_labels : (V,) or None
    figsize, save

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    _apply_style()
    import matplotlib.tri as mtri
    from matplotlib.colors import LogNorm, Normalize

    H_h = np.asarray(H_hks, dtype=np.float64)
    H_w = np.asarray(H_wks, dtype=np.float64)
    uv = np.asarray(unfolded_coords, dtype=np.float64)
    fcs = np.asarray(faces, dtype=np.int64)

    tri = mtri.Triangulation(uv[:, 0], uv[:, 1], fcs)

    T_h = H_h.shape[1]
    T_w = H_w.shape[1]

    sel_h = np.unique(np.geomspace(1, T_h, n_scales, dtype=int).clip(0, T_h - 1))
    sel_w = np.unique(np.geomspace(1, T_w, n_scales, dtype=int).clip(0, T_w - 1))
    n_h = len(sel_h)
    n_w = len(sel_w)
    n_cols = max(n_h, n_w)

    if t_values_hks is None:
        t_values_hks = np.arange(T_h, dtype=np.float64)
    if e_values_wks is None:
        e_values_wks = np.arange(T_w, dtype=np.float64)
    t_h = np.asarray(t_values_hks)
    e_w = np.asarray(e_values_wks)

    if figsize is None:
        figsize = (4.0 * n_cols, 5.0)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize,
                              sharex=True, sharey=True,
                              constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # --- helper to get norm ---
    def _make_norm(H_block, indices):
        vals = H_block[:, indices]
        vmin = float(np.nanpercentile(vals[vals > 0], 2)) if log_norm else float(vals.min())
        vmax = float(np.nanpercentile(vals, 98))
        if log_norm:
            return LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 2))
        return Normalize(vmin=vmin, vmax=vmax)

    norm_h = _make_norm(H_h, sel_h)
    norm_w = _make_norm(H_w, sel_w)

    tpc_h = tpc_w = None

    # --- top row: HKS ---
    for pi, ti in enumerate(sel_h):
        ax = axes[0, pi]
        tpc_h = ax.tripcolor(tri, H_h[:, ti], shading="gouraud",
                              cmap=cmap_hks, norm=norm_h, rasterized=True)
        ax.set_title(f"t = {t_h[ti]:.2g}", fontsize=7)
        ax.set_aspect("equal")
        if subfield_labels is not None:
            ax.tricontour(tri, subfield_labels.astype(float),
                          levels=np.arange(subfield_labels.max()) + 0.5,
                          colors="k", linewidths=0.4)

    # --- bottom row: WKS ---
    for pi, ei in enumerate(sel_w):
        ax = axes[1, pi]
        tpc_w = ax.tripcolor(tri, H_w[:, ei], shading="gouraud",
                              cmap=cmap_wks, norm=norm_w, rasterized=True)
        ax.set_title(f"E = {e_w[ei]:.2g}", fontsize=7)
        ax.set_aspect("equal")
        if subfield_labels is not None:
            ax.tricontour(tri, subfield_labels.astype(float),
                          levels=np.arange(subfield_labels.max()) + 0.5,
                          colors="k", linewidths=0.4)

    # hide unused panels
    for pi in range(n_h, n_cols):
        axes[0, pi].set_visible(False)
    for pi in range(n_w, n_cols):
        axes[1, pi].set_visible(False)

    # row labels
    axes[0, 0].set_ylabel("HKS\nPD", fontsize=8)
    axes[1, 0].set_ylabel("WKS\nPD", fontsize=8)

    # colourbars
    if tpc_h is not None:
        fig.colorbar(tpc_h, ax=axes[0, :].tolist(), location="right",
                     shrink=0.7, pad=0.02, label="HKS")
    if tpc_w is not None:
        fig.colorbar(tpc_w, ax=axes[1, :].tolist(), location="right",
                     shrink=0.7, pad=0.02, label="WKS")

    _savefig(fig, save)
    return fig, axes
