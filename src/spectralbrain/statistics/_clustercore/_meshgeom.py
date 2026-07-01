"""Mesh-adjacency and edge-distance helpers for the contiguous ddCRP cores.

Ported from brainmosaic.core.geometry so the vendored cluster cores are
self-contained. SpectralBrain's :class:`spectralbrain.core.meshes.BrainMesh`
exposes richer geometry; these are the minimal pieces the ddCRP sampler needs:
a per-vertex neighbour list and the aligned Euclidean edge lengths that let the
decay kernel modulate contiguity by physical distance.
"""

from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp


def adjacency_list_from_faces(faces: np.ndarray, n_vertices: int) -> List[np.ndarray]:
    """Per-vertex neighbour-index arrays from a triangle list."""
    faces = np.asarray(faces, dtype=np.int64)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    data = np.ones(e.shape[0], dtype=np.int8)
    A = sp.coo_matrix((data, (e[:, 0], e[:, 1])),
                      shape=(n_vertices, n_vertices)).tocsr()
    A.data[:] = 1
    return [A.indices[A.indptr[v]:A.indptr[v + 1]].copy() for v in range(n_vertices)]


def edge_distances(vertices: np.ndarray, adjacency_list) -> List[np.ndarray]:
    """Per-neighbour Euclidean edge lengths aligned to ``adjacency_list``.

    For adjacent mesh vertices the Euclidean edge length closely approximates the
    geodesic distance, so these are the natural ``distances`` for the ddCRP decay
    kernel: with them, ``decay_kind="exponential"``/``"logistic"`` modulate
    contiguity physically instead of treating every neighbour as equidistant.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3); got {vertices.shape}.")
    if len(adjacency_list) != vertices.shape[0]:
        raise ValueError("adjacency_list length must equal the number of vertices.")
    out: List[np.ndarray] = []
    for v, neigh in enumerate(adjacency_list):
        neigh = np.asarray(neigh, dtype=np.int64)
        if neigh.size == 0:
            out.append(np.zeros(0, dtype=np.float64))
        else:
            out.append(np.linalg.norm(vertices[neigh] - vertices[v], axis=1))
    return out


def adjacency_list_from_sparse(adjacency) -> List[np.ndarray]:
    """Per-vertex neighbour arrays from a scipy sparse adjacency matrix."""
    A = sp.csr_matrix(adjacency)
    n = A.shape[0]
    return [A.indices[A.indptr[v]:A.indptr[v + 1]].copy() for v in range(n)]
