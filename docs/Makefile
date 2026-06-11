# Makefile for SpectralBrain documentation.
# Usage:
#   make html      build the HTML site into build/html
#   make clean     remove the build directory and copied notebooks
#   make linkcheck check external links
#   make serve     build, then serve locally at http://localhost:8000

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
TUTORIALS_SRC = ../tutorials
TUTORIALS_DST = source/tutorials

.PHONY: help html clean linkcheck serve notebooks

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Copy the committed tutorial notebooks into the docs tree before building.
notebooks:
	@mkdir -p "$(TUTORIALS_DST)"
	@cp $(TUTORIALS_SRC)/*.ipynb "$(TUTORIALS_DST)/" 2>/dev/null || \
		echo "note: no notebooks found in $(TUTORIALS_SRC) (skipping)"

html: notebooks
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Built: $(BUILDDIR)/html/index.html"

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

serve: html
	@python -m http.server 8000 --directory "$(BUILDDIR)/html"

clean:
	@rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/auto_examples" "$(SOURCEDIR)/api/_autosummary"
	@rm -f $(TUTORIALS_DST)/*.ipynb
	@echo "Cleaned build artifacts."
