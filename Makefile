# Makefile — SpectralBrain development shortcuts

.PHONY: help install install-full test lint format build clean \
        publish-test publish docs check-version

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
	@echo "  make publish-test  Publish to TestPyPI"
	@echo "  make clean         Remove build artifacts"
	@echo "  make docs          Build documentation"
	@echo ""

install:
	uv sync --group dev

install-full:
	uv sync --all-extras --group dev

test:
	uv run pytest tests/ -v --tb=short

test-fast:
	uv run pytest tests/ -v --tb=short -m "not slow"

test-cov:
	uv run pytest tests/ --cov=spectralbrain --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint:
	uv run ruff check src/spectralbrain/ tests/
	uv run ruff format --check src/spectralbrain/ tests/

format:
	uv run ruff format src/spectralbrain/ tests/
	uv run ruff check --fix src/spectralbrain/ tests/

typecheck:
	uv run mypy src/spectralbrain/

build: clean
	uv build
	@echo ""
	@echo "Built artifacts:"
	@ls -lh dist/

check-version:
	@uv run python -c "import spectralbrain; print(f'Version: {spectralbrain.__version__}')"

publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/ --trusted-publishing always

docs:
	uv run sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation: docs/_build/html/index.html"

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts."
