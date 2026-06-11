# API Reference

Generated from the source docstrings. The reference is intentionally terse — the
*why* lives in {doc}`Learn <../learn/index>`, the *how* in
{doc}`Tutorials <../tutorials/index>` and the
{doc}`Gallery <../auto_examples/index>`.

Everything below is importable from the top-level `spectralbrain` namespace
(e.g. `sb.compute_hks`), with the full module paths shown for reference.

## Subpackages

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   spectralbrain.core
   spectralbrain.io
   spectralbrain.spectral
   spectralbrain.statistics
   spectralbrain.viz
   spectralbrain.utils
   spectralbrain.backends
   spectralbrain.runtime
```

## Most-used entry points

```{eval-rst}
.. currentmodule:: spectralbrain

.. rubric:: Geometry

.. autosummary::
   :toctree: _autosummary

   BrainMesh
   BrainPointCloud
   SpectralDecomposition

.. rubric:: Spectral descriptors

.. autosummary::
   :toctree: _autosummary

   compute_shapedna
   compute_hks
   compute_si_hks
   compute_wks
   compute_gps
   compute_bates_signatures
   compute_bks
   compute_ibks
   sgw_transform
   sgw_descriptor
   compute_all_descriptors
   anisotropic_laplacian

.. rubric:: Correspondence & distances

.. autosummary::
   :toctree: _autosummary

   compute_functional_map
   shape_difference_operator
   wesd
   wesd_matrix
   shapedna_distance
   biharmonic_distance
   commute_time_distance
   diffusion_distance
   descriptor_distance
   build_geometric_connectome

.. rubric:: I/O

.. autosummary::
   :toctree: _autosummary

   load
   load_freesurfer_surface
   load_gifti_surface
   load_nifti
   load_group
   parcellate
   apply_parcellation
```
