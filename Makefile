# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= pdm run sphinx-build
SOURCEDIR     = source
BUILDDIR      = .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

###############################################################################
# COMMANDS
###############################################################################
.PHONY: install-pdm
## install pdm before environment setup
install-pdm:
	python -m pip install -U pip setuptools wheel pdm

.PHONY: update-lock
## update pdm.lock
update-lock:
	pdm update --no-sync

.PHONY: update-setup
## update setup.py
update-setup:
	pdm export -f setuppy -o setup.py --pyprojec t

.PHONY: deploy-dev
## deploy dev environment
deploy-dev:
	pdm sync -G dev --clean

.PHONY: clean
## Clean python cache file.
clean:
	find dist ! -name '.nojekyll' -delete