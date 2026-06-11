# Installation

SpectralBrain runs on **Python 3.11–3.12**.

## From PyPI

```bash
pip install spectralbrain
```

The base install gives you the full spectral-geometry core (Laplace–Beltrami
decomposition, all descriptors, distances), input-agnostic I/O, and the 2D/3D
rendering core.

## Optional extras

Install only what your study needs:

```bash
pip install "spectralbrain[bayesian]"   # PyMC, nutpie, NumPyro, BlackJAX, ArviZ
pip install "spectralbrain[viz]"        # vedo, scienceplots, hippunfold_plot, …
pip install "spectralbrain[gpu]"        # torch, CuPy, JAX (CUDA)
pip install "spectralbrain[neuro]"      # nilearn, dipy, pybids, templateflow, …
pip install "spectralbrain[full]"       # everything above
```

:::{admonition} GPU is optional
:class: note
The CPU backend (NumPy/SciPy) computes the Laplace–Beltrami eigenpairs and every
descriptor on its own. The `gpu` extra accelerates large batches but is never
required — nothing in the core depends on a CUDA device.
:::

:::{admonition} FreeSurfer is optional
:class: note
SpectralBrain *reads* FreeSurfer and HippUnfold output but does not run them. If
your surfaces already exist (`lh.pial`, hippocampal `.surf.gii`, …), you do not
need FreeSurfer installed to analyze them.
:::

## Development install

```bash
git clone https://github.com/rdneuro/spectralbrain
cd spectralbrain
pip install -e ".[full]"
```
