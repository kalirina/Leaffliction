VENV = ~/goinfre/leaf_venv
PYTHON = $(VENV)/bin/python

.PHONY: all install clean fclean

all: install

install:
	uv python install 3.12
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PYTHON) -r requirements.txt
	uv pip install --python $(PYTHON) plantcv
	uv pip install --python $(PYTHON) torch torchvision --index-url https://download.pytorch.org/whl/cpu

clean:
	rm -rf graphs/*
	rm -rf data/stats.json
	rm -rf data/augmented_directory

fclean: clean
	rm -rf $(VENV)