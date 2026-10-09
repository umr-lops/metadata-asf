# Makefile for building the Sphinx documentation of metadata-asf.
#
# The build is driven by Sphinx's "mode" router (-M), so any builder Sphinx
# supports is available as a target: `make html`, `make latexpdf`, `make clean`, ...
# The output always lands in docs/_build/<builder>/ (gitignored).

#* Variables
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
SPHINXOPTS    ?=

#* Help
.PHONY: help
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#* Catch-all: every other target is delegated to the Sphinx mode router.
#    `make html` -> sphinx-build -M html . _build ; `make clean` -> -M clean ...
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#* Convenience: `make docs` builds the HTML in-place.
.PHONY: docs
docs: html
