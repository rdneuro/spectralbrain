"""
Heat Kernel Signature on a surface
==================================

Compute the Laplace–Beltrami decomposition of a small synthetic surface and
render its Heat Kernel Signature at one time scale. Replace the icosphere with
``sb.load_freesurfer_surface("lh.pial")`` to run on a real cortical surface.
"""

import numpy as np
import pyvista as pv

import spectralbrain as sb

# %%
# Build a small synthetic surface (a perturbed sphere) so the example is light.
sphere = pv.Icosphere(nsub=4, radius=1.0)
warp = 0.15 * np.sin(4 * sphere.points[:, 2])
sphere.points[:, 0] += warp
vertices = np.asarray(sphere.points)
faces = sphere.faces.reshape(-1, 4)[:, 1:]  # PyVista (n,4) -> (n,3)

# %%
# Decompose once, then read the descriptor off the decomposition.
mesh = sb.BrainMesh(vertices, faces)
decomp = mesh.decompose(k=80)
hks = sb.compute_hks(decomp, t_values=np.geomspace(0.01, 1.0, 3))

# %%
# Render the mid-scale HKS channel on the surface.
surf = pv.PolyData(vertices, sphere.faces)
surf["HKS"] = hks[:, 1]

plotter = pv.Plotter(off_screen=True, window_size=(600, 600))
plotter.add_mesh(surf, scalars="HKS", cmap="magma", smooth_shading=True,
                 scalar_bar_args={"title": "HKS (t mid)"})
plotter.view_isometric()
plotter.show()
