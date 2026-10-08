"""Geometric eigenmodes of a surface -- the LBO basis itself, as a field.

The Laplace-Beltrami eigenfunctions ``psi_k`` of a surface are its natural
geometric modes (Pang et al., Nature 2023; Cao et al., HBM 2024, *Mode-Based
Morphometry*). This module computes them on the very mesh they will be shown
on, so that the vertex-to-value correspondence is guaranteed, and provides the
small spectral utilities that interpret them.

- :func:`compute_geometric_eigenmodes` -- FEM eigenmodes via LaPy (the solver
  used in the MBM and Reuter-lab pipelines), from a path, a ``(vertices,
  faces)`` pair or a :class:`~spectralbrain.core.meshes.BrainMesh`.
- :func:`eigenmode_wavelength` -- spatial wavelength ``2*pi / sqrt(lambda)``.

Rendering lives in :mod:`spectralbrain.viz.spectral`.

The sign of an eigenvector is arbitrary: what carries meaning are its nodal
lines (``psi_k = 0``), not whether a lobe is positive or negative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectralbrain.runtime import PathLike

_GENERIC_MESH_SUFFIXES = {".ply", ".obj", ".stl", ".vtk", ".vtp"}


def _resolve_surface(surface: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` from a path, a pair or a BrainMesh."""
    from spectralbrain.core.meshes import BrainMesh

    if isinstance(surface, BrainMesh):
        return np.asarray(surface.vertices), np.asarray(surface.faces)
    if isinstance(surface, (tuple, list)) and len(surface) == 2:
        return np.asarray(surface[0]), np.asarray(surface[1])
    if isinstance(surface, (str, Path)):
        from spectralbrain.io.loaders import (
            load_freesurfer_surface,
            load_gifti_surface,
            load_mesh,
        )

        path = Path(surface)
        name = path.name.lower()
        if name.endswith(".gii"):
            return load_gifti_surface(path)
        if path.suffix.lower() in _GENERIC_MESH_SUFFIXES:
            return load_mesh(path)
        return load_freesurfer_surface(path)
    raise TypeError(
        "surface must be a path, a (vertices, faces) pair or a BrainMesh; "
        f"got {type(surface).__name__}."
    )


def compute_geometric_eigenmodes(
    surface: PathLike | tuple[np.ndarray, np.ndarray] | Any,
    n_modes: int = 50,
    use_lumped_mass: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geometric (Laplace-Beltrami) eigenmodes on the render mesh, via LaPy.

    Solves the FEM weak form of ``Delta psi = -lambda psi`` on the surface.
    Because the modes are solved on the same mesh that will be plotted, the
    vertex-to-value correspondence is guaranteed.

    Parameters
    ----------
    surface : path, (vertices, faces) or BrainMesh
        GIfTI (``.surf.gii``), FreeSurfer geometry, or a generic mesh file
        (``.ply``, ``.obj``, ``.stl``, ``.vtk``, ``.vtp``); or the arrays
        directly.
    n_modes : int
        Number of eigenpairs (including the constant mode).
    use_lumped_mass : bool
        Lumped (diagonal) mass matrix, as in the MBM pipeline.

    Returns
    -------
    evals : ndarray, shape (n_modes,)
        Eigenvalues, ascending.
    evecs : ndarray, shape (N, n_modes)
        Eigenvectors, columns ordered as ``evals``.
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (F, 3)
    """
    try:
        from lapy import Solver, TriaMesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LaPy is required for compute_geometric_eigenmodes: pip install lapy"
        ) from exc

    verts, faces = _resolve_surface(surface)
    tria = TriaMesh(verts, faces)
    solver = Solver(tria, lump=use_lumped_mass)
    evals, evecs = solver.eigs(k=int(n_modes))
    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs, dtype=float)
    order = np.argsort(evals)
    return evals[order], evecs[:, order], verts, faces


def eigenmode_wavelength(eigenvalue: float) -> float:
    """Spatial wavelength of a mode: ``2 * pi / sqrt(lambda)`` (inf for lambda ~ 0)."""
    lam = float(eigenvalue)
    if lam <= 1e-12:
        return np.inf
    return 2.0 * np.pi / np.sqrt(lam)


__all__: list[str] = [
    "compute_geometric_eigenmodes",
    "eigenmode_wavelength",
]
