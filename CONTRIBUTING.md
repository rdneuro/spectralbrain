# Contributing to SpectralBrain

Thank you for your interest in improving SpectralBrain. This document
explains how to set up a development environment, the quality bar for
contributions, and how to propose changes.

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report bugs** by opening an issue with a minimal reproducible example
  (mesh/point-cloud size, function called, traceback, package versions).
- **Suggest features** or new spectral descriptors, ideally with a
  reference to the method in the literature.
- **Improve documentation**, examples, or docstrings.
- **Submit code** via pull request, following the guidelines below.

## Development setup

SpectralBrain uses [uv](https://docs.astral.sh/uv/) for fast, reproducible
environments, but plain `pip` works too.

```bash
git clone https://github.com/rdneuro/spectralbrain.git
cd spectralbrain

# With uv (recommended)
uv sync --all-extras --group dev

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]" --group dev   # pip >= 25; otherwise: pip install -e . then the dev tools

# Enable the pre-commit hooks
pre-commit install
```

## Quality bar

All pull requests must pass CI, which runs:

```bash
ruff check src/          # lint
ruff format --check src/ # formatting
pytest tests/            # tests (Python 3.11 and 3.12)
```

Before pushing, run the same checks locally:

```bash
make lint      # ruff check + ruff format --check
make format    # apply ruff formatting
make test      # run the test suite
make typecheck # mypy (optional but encouraged)
```

### Code style

- **Python only** (the runtime is pure Python; Julia/bash helper scripts
  live outside the package).
- Follow the existing **NumPy-style docstrings** and the conventions in
  the surrounding module.
- Scientific notation in docstrings (λ, φ, ×, ≈, …) is welcome and is
  explicitly allowed by the linter configuration.
- Keep heavy or optional dependencies (torch, pymc, vedo, open3d, …)
  **lazily imported** inside the functions that need them, so that
  `import spectralbrain` stays light. Use the `_require_*` helper pattern
  for clear "please install X" errors.
- Validate inputs (shapes, dtypes) at function boundaries and fail loudly.

### Tests

- Add tests under `tests/` for any new behavior. Numerical code should be
  validated against a known property or analytic benchmark (see
  `tests/test_spectral.py`, which uses the analytic spectrum of the unit
  sphere) rather than only checking that a call returns without error.
- The test configuration treats warnings as errors; keep new code free of
  deprecation warnings.

## Pull request process

1. Fork the repository and create a feature branch from `develop`
   (or `main` for small fixes).
2. Make your change, add tests and documentation, and update
   `CHANGELOG.md` under the `[Unreleased]` section.
3. Ensure `make lint` and `make test` pass.
4. Open a pull request describing the motivation and the change. Link any
   related issue.

## Reporting security issues

Please do not open public issues for security-sensitive problems. Instead,
email the maintainer at r.debona@posgrad.ufsc.br.
