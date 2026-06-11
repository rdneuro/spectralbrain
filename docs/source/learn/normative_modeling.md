# Normative modeling

Spectral descriptors become clinically useful when you can say how *unusual* a
given brain is. Normative modeling answers that by learning the expected
distribution of a descriptor across a healthy reference cohort, then scoring each
new subject against it.

## Why normative, not matched controls

Matched control groups force a binary, case-vs-control framing and carry a
control-selection confound. A normative model instead gives a **dimensional**
z-score per subject (and per vertex or region), models its uncertainty
explicitly, and removes the need to hand-pick a matched group:

$$
z(x) = \frac{\, y(x) - \hat{\mu}(x)\,}{\hat{\sigma}(x)} ,
$$

where $\hat\mu$ and $\hat\sigma$ are the normative mean and dispersion estimated
from the reference cohort at location $x$. A subject's deviation map is then a
continuous field of how far each location departs from normative expectation.

## In SpectralBrain

The `statistics.normative` module estimates the reference model and produces
z-deviation maps for new subjects; pair it with the harmonization tools
(ComBat / ComBat-GAM) when the reference cohort spans multiple sites, so that
deviations reflect biology rather than scanner.

```python
from spectralbrain.statistics import normative
# fit on a reference cohort, then z-score new subjects against it
```

:::{seealso}
Tutorials `08_cohorts_and_vertexwise_stats` and
`09_effectsizes_classification_harmonization` show the cohort, statistics, and
harmonization pieces end to end. See also the {doc}`How-to guides <../howto/index>`.
:::
