# How-to guides

Task-focused recipes for things you already understand and just want to *do*.
For the concepts behind them, see {doc}`Learn <../learn/index>`; for the full
signatures, see the {doc}`API <../api/index>`.

## Accelerate a batch on the GPU

The CPU backend is the default and computes everything on its own. For large
cohorts, install the `gpu` extra and select the GPU backend; nothing about the
analysis code changes.

```python
import spectralbrain as sb
# sb.NumpyBackend is the default; the gpu extra adds CUDA-backed solvers.
```

## Harmonize across scanners before group statistics

When a reference or patient cohort spans multiple sites, harmonize descriptors
with ComBat / ComBat-GAM so that group differences reflect biology, not scanner.
This is the dual sensitivity question — always report harmonized *and* raw where
batch and group are confounded.

```python
from spectralbrain.statistics import analysis   # harmonization + group tests
```

## Vertex-wise statistics with real family-wise error control

Use max-statistic permutation for genuine FWE control, with FDR and TFCE as
alternatives, plus partial correlations with the correct degrees of freedom.

```python
from spectralbrain.statistics import analysis
```

## Fit a Bayesian model

Six PyMC models ship with the `bayesian` extra, with posterior and diagnostic
plots in `viz`.

```python
from spectralbrain.statistics import bayesian
from spectralbrain.viz import bayes as bayes_viz
```

## The MTLE-HS hippocampus workflow

The library's primary use case: load HippUnfold hippocampal surfaces, compute
spectral descriptors, build a normative reference, and z-score patients,
lateralizing left vs. right. Tutorials 06–10 assemble this end to end.

:::{seealso}
{doc}`../learn/normative_modeling`, and tutorials
`08_cohorts_and_vertexwise_stats` through `10_bayesian_and_visualization`.
:::
