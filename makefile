SHELL := /bin/bash
CONDA_ENV_NAME = jb
ACTIVATE_CONDA = source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV_NAME)

MD_DIRECTORY := ./dsbook
MD_FILES := $(shell find $(MD_DIRECTORY) -name "*.md" -not -path "$(MD_DIRECTORY)/_build/*")
IPYNB_FILES := $(MD_FILES:.md=.ipynb)

# Notebooks whose outputs are committed (exempt from nbstripout in .gitattributes).
# These are re-executed whenever their .md changes, so figures stay in sync with
# the source. All other .ipynb are synced without execution and stripped at commit.
EXEC_IPYNB := dsbook/network/fba.ipynb \
              dsbook/network/gem_ecoli.ipynb \
              dsbook/network/gsea.ipynb \
              dsbook/statistics/testing.ipynb \
              dsbook/statistics/qvalue.ipynb \
              dsbook/unsupervised/VAEofCarcinomas.ipynb

.PHONY: all build

all: build

# Convert any .md to .ipynb when .md is newer; execute the committed-output ones
%.ipynb: %.md
	@echo "Syncing $< -> $@"
	@$(ACTIVATE_CONDA) && jupytext --to ipynb --update --output $@ $<
	@if echo " $(EXEC_IPYNB) " | grep -q " $@ "; then \
	  echo "Executing $@"; \
	  $(ACTIVATE_CONDA) && jupyter nbconvert --to notebook --execute --inplace $@; \
	fi

# Build the book, but only after notebooks are up-to-date
build: $(IPYNB_FILES)
	@echo "Building Jupyter Book..."
	@$(ACTIVATE_CONDA) && cd dsbook && jupyter book build --html --execute
