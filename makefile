# Define the Conda environment name
CONDA_ENV_NAME = jb

# Define the commands
ACTIVATE_CONDA = source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV_NAME)

.PHONY: all build activate-env convert-md2ipynb

# Main target that runs everything
all: activate-env convert-md2ipynb build

# Target to activate the conda environment
activate-env:
	@echo "Activating conda environment: $(CONDA_ENV_NAME)"
	@$(ACTIVATE_CONDA)

# Target to convert Markdown files to Jupyter notebooks
convert-md2ipynb: activate-env
	@echo "Converting Markdown files to notebooks..."
	@$(ACTIVATE_CONDA) && python md2ipynb.py

# Target to build the Jupyter Book
build: activate-env
	@echo "Building Jupyter Book..."
	@$(ACTIVATE_CONDA) && jupyter-book build dsbook/
