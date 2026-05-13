# Makefile — SpectralBrain development shortcuts
# Usage: make <target>
#
# This Makefile wraps common uv/pytest/ruff commands so you don't have
# to remember the exact invocations.  It's optional — you can always
# run the underlying commands directly.

.PHONY: help install install-full test lint format build clean \
        publish-test publish docs check-version

# Default target: show available commands.
help:
	@echo ""
	@echo "SpectralBrain — development commands"
	@echo "──────────────────────────────────────────────────"
	@echo "  make install       Install core + dev dependencies"
	@echo "  make install-full  Install everything (gpu, viz, bayesian, neuro)"
	@echo "  make test          Run test suite"
	@echo "  make test-fast     Run tests without slow markers"
	@echo "  make lint          Check code style (ruff)"
	@echo "  make format        Auto-format code (ruff)"
	@echo "  make build         Build sdist + wheel"
	@echo "  make check-version Show current package version"
	@echo "  make publish-test  Publish to TestPyPI (manual, for debugging)"
	@echo "  make clean         Remove build artifacts"
	@echo "  make docs          Build documentation"
	@echo ""

# ── Installation ──────────────────────────────────────────────────────
install:
	uv sync --group dev

install-full:
	uv sync --all-extras --group dev

# ── Testing ───────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v --tb=short

test-fast:
	uv run pytest tests/ -v --tb=short -m "not slow"

test-cov:
	uv run pytest tests/ --cov=spectralbrain --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────
lint:
	uv run ruff check spectralbrain/ tests/
	uv run ruff format --check spectralbrain/ tests/

format:
	uv run ruff format spectralbrain/ tests/
	uv run ruff check --fix spectralbrain/ tests/

typecheck:
	uv run mypy spectralbrain/

# ── Build & release ───────────────────────────────────────────────────
build: clean
	uv build
	@echo ""
	@echo "Built artifacts:"
	@ls -lh dist/

check-version:
	@uv run python -c "import spectralbrain; print(f'Version: {spectralbrain.__version__}')"

# Publish to TestPyPI using uv (for manual debugging; normally the
# GitHub Actions workflow handles this automatically).
publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/ --trusted-publishing always

# ── Documentation ─────────────────────────────────────────────────────
docs:
	uv run sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation: docs/_build/html/index.html"

# ── Cleanup ───────────────────────────────────────────────────────────
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts."
