# Contributing

Contributions are welcome. The authoritative guide is
[`CONTRIBUTING.md`](https://github.com/rdneuro/spectralbrain/blob/main/CONTRIBUTING.md)
in the repository; the essentials:

- **Issues and pull requests** go through the
  [GitHub repository](https://github.com/rdneuro/spectralbrain).
- **Code style** is enforced by `pre-commit` (see `.pre-commit-config.yaml`).
- **Tests** live under `tests/`; run them before opening a PR.

## Rebuilding the docs locally

```bash
pip install -r docs/requirements.txt
pip install -e .
make -C docs html        # output in docs/build/html/
```

## Rebuilding the tutorials

The tutorial notebooks are produced by the `gen_nb*.py` generators in
`tutorials/`. The documentation renders the committed notebooks **without
re-executing them**, so figures appear exactly as generated on a machine with
the data and (where needed) a GPU. To refresh a notebook, re-run its generator
and commit the updated `.ipynb`.
