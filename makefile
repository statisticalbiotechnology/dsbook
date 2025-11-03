SHELL := /bin/bash
CONDA_ENV_NAME = jb
ACTIVATE_CONDA = source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV_NAME)

MD_DIRECTORY := ./dsbook
MD_FILES := $(shell find $(MD_DIRECTORY) -name "*.md" -not -path "$(MD_DIRECTORY)/_build/*")
IPYNB_FILES := $(MD_FILES:.md=.ipynb)

.PHONY: all build

all: build

# Convert any .md to .ipynb when .md is newer
%.ipynb: %.md
	@echo "Syncing $< -> $@"
	@$(ACTIVATE_CONDA) && jupytext --to ipynb --update --output $@ $<

# Build the book, but only after notebooks are up-to-date
build: $(IPYNB_FILES)
	@echo "Building Jupyter Book..."
	@$(ACTIVATE_CONDA) && jupyter-book build dsbook/