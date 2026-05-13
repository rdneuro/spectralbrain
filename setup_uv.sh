#!/usr/bin/env bash
# setup_uv.sh — Bootstrap a SpectralBrain development environment using uv.
#
# Usage:
#   chmod +x setup_uv.sh
#   ./setup_uv.sh              # Core deps only (fast, for library dev)
#   ./setup_uv.sh --full       # All optional extras (viz, bayesian, gpu, neuro)
#   ./setup_uv.sh --notebooks  # Core + notebook extras (for running examples)
#
# Requirements:
#   - Internet connection (first run downloads uv + Python 3.11)
#   - ~2 GB disk for core, ~8 GB for --full (GPU wheels are large)
#
# Note: uv manages pip-installable packages only.  For system-level
# dependencies (FreeSurfer, CUDA toolkit, ANTs binaries), use conda
# or your system package manager.  You can also activate a conda env
# first, then run this script inside it — uv respects the active
# Python interpreter.

set -euo pipefail

# ── Colour helpers ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Colour

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[fail]${NC}  $*"; exit 1; }

# ── Parse arguments ──
EXTRAS=""
case "${1:-}" in
    --full)       EXTRAS="--all-extras" ;;
    --notebooks)  EXTRAS="--extra notebooks --extra viz" ;;
    --gpu)        EXTRAS="--extra gpu" ;;
    --bayesian)   EXTRAS="--extra bayesian" ;;
    --viz)        EXTRAS="--extra viz" ;;
    --neuro)      EXTRAS="--extra neuro" ;;
    --help|-h)
        echo "Usage: $0 [--full|--notebooks|--gpu|--bayesian|--viz|--neuro]"
        echo ""
        echo "  (no args)    Core dependencies only (library development)"
        echo "  --full       All optional extras"
        echo "  --notebooks  Core + viz + notebook extras"
        echo "  --gpu        Core + PyTorch/CuPy/JAX"
        echo "  --bayesian   Core + PyMC/ArviZ/Bambi"
        echo "  --viz        Core + vedo/open3d/yabplot/scienceplots"
        echo "  --neuro      Core + nilearn/MNE/DiPy/ANTsPy"
        exit 0
        ;;
    "")           EXTRAS="" ;;
    *)            warn "Unknown option '$1', installing core only." ;;
esac

# ── Step 1: Install uv (idempotent) ──
if command -v uv >/dev/null 2>&1; then
    info "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed: $(uv --version)"
fi

# ── Step 2: Ensure Python 3.11 is available ──
info "Ensuring Python 3.11 is available..."
uv python install 3.11 2>/dev/null || true

# Write .python-version so uv auto-selects 3.11 for this project
echo "3.11" > .python-version
ok "Python 3.11 pinned in .python-version"

# ── Step 3: Create virtual environment ──
if [ -d ".venv" ]; then
    info "Virtual environment .venv/ already exists, reusing it."
else
    info "Creating virtual environment..."
    uv venv --python 3.11
    ok "Virtual environment created at .venv/"
fi

# ── Step 4: Resolve and install ──
info "Resolving dependencies and installing..."
if [ -n "$EXTRAS" ]; then
    info "  Extras: $EXTRAS"
    uv sync --group dev $EXTRAS
else
    info "  Core dependencies + dev tools only."
    uv sync --group dev
fi

# ── Step 5: Show lockfile status ──
if [ -f "uv.lock" ]; then
    ok "Lockfile uv.lock exists ($(wc -l < uv.lock) lines)"
    info "  Commit uv.lock to version control for reproducibility."
else
    warn "No uv.lock found — run 'uv lock' to generate one."
fi

# ── Step 6: Verify installation ──
info "Verifying SpectralBrain installation..."
.venv/bin/python -c "
import spectralbrain as sb
print(f'  SpectralBrain {sb.__version__}')
print(f'  Modules: runtime, core, io, spectral, statistics, backends, utils, viz')
" 2>/dev/null && ok "SpectralBrain imported successfully!" || warn "Import check failed — some optional deps may be missing (this is OK for core-only installs)."

# ── Step 7: Print activation instructions ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Environment ready!${NC}"
echo ""
echo "  Activate:   source .venv/bin/activate"
echo "  Run tests:  uv run pytest"
echo "  Run linter: uv run ruff check spectralbrain/"
echo "  Jupyter:    uv run jupyter lab"
echo ""
echo "  To add a single package:"
echo "    uv add nibabel         # adds to [project.dependencies]"
echo "    uv add --dev pytest    # adds to [dependency-groups.dev]"
echo ""
echo "  To update the lockfile after editing pyproject.toml:"
echo "    uv lock"
echo "    uv sync"
echo ""
echo "  For reproducible CI installs:"
echo "    uv sync --locked --all-extras --group test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
