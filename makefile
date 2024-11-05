# Use bash as the shell for this Makefile
SHELL := /bin/bash

# Define the Conda environment name
CONDA_ENV_NAME = jb

# Define the commands to activate the conda environment in bash
ACTIVATE_CONDA = source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV_NAME)

# Directory containing your Markdown files
MD_DIRECTORY = ./dsbook

# Find all Markdown files
MD_FILES = $(shell if [ -d $(MD_DIRECTORY) ]; then find $(MD_DIRECTORY) -name "*.md" -not -path "$(MD_DIRECTORY)/_build/*"; fi)

.PHONY: all build activate-env convert-md2ipynb

# Main target that runs everything
all: activate-env convert-md2ipynb build

# Target to activate the conda environment
activate-env:
	@echo "Activating conda environment: $(CONDA_ENV_NAME)"
	@$(ACTIVATE_CONDA)

# Target to convert Markdown files to Jupyter notebooks only if .md is newer than .ipynb
convert-md2ipynb: activate-env
	@echo "Converting newer Markdown files to notebooks..."
	@$(ACTIVATE_CONDA) && $(foreach md, $(MD_FILES), \
		ipynb=$$(echo $(md) | sed 's/\.md$$/.ipynb/'); \
		if [ ! -f $$ipynb ] || [ $(md) -nt $$ipynb ]; then \
			echo "Converting $(md) to $$ipynb"; \
			jupytext --to notebook $(md); \
		fi;)

# Target to build the Jupyter Book
build: activate-env
	@echo "Building Jupyter Book..."
	@$(ACTIVATE_CONDA) && jupyter-book build dsbook/
