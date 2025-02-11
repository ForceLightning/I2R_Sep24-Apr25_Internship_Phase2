# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = ../src

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

MK_APIDOC = "$(SPHINXAPIDOC)" -o "docs/source/api" "$(PACKAGEDIR)" $(SPHINXOPTS) $(O)
apidoc: Makefile
	$(MK_APIDOC)

define SPHINX_CMD
	$(SPHINXBUILD) -M $(1) "$(SOURCEDIR)" "$(BUILDDIR)" $SPHINXOPTS $(O) -j auto
endef

docs:
	$(MK_APIDOC)
	$(call SPHINX_CMD,html)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
