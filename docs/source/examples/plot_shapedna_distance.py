"""
ShapeDNA: comparing two shapes
==============================

Compute the ShapeDNA fingerprint (the Laplace–Beltrami spectrum) of two
synthetic surfaces and quantify how different they are with the ShapeDNA
distance. This example uses only matplotlib, so it is fast and dependency-light.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import spectralbrain as sb


def _fingerprint(mesh_pv, k=60):
    v = np.asarray(mesh_pv.points)
    f = mesh_pv.faces.reshape(-1, 4)[:, 1:]
    decomp = sb.BrainMesh(v, f).decompose(k=k)
    return sb.compute_shapedna(decomp), decomp


# %%
# Two shapes: a sphere and an ellipsoid (the sphere stretched along z).
sphere = pv.Icosphere(nsub=4)
ellipsoid = pv.Icosphere(nsub=4)
ellipsoid.points[:, 2] *= 1.8

sdna_a, dec_a = _fingerprint(sphere)
sdna_b, dec_b = _fingerprint(ellipsoid)

# %%
# Compare the fingerprints with the ShapeDNA distance.
dist = sb.shapedna_distance(sdna_a, sdna_b)

# %%
# Plot the two spectra; the gap between them is what the distance summarizes.
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sdna_a, "-o", ms=3, label="sphere")
ax.plot(sdna_b, "-s", ms=3, label="ellipsoid")
ax.set_xlabel("eigenvalue index $i$")
ax.set_ylabel(r"normalized $\lambda_i$")
ax.set_title(f"ShapeDNA spectra — distance = {dist:.3f}")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
