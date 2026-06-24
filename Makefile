PYTHON = ~/goinfre/leaf_venv/bin/python
VENV = ~/goinfre/leaf_venv

.PHONY: all install clean

all: install

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

clean:
	rm -rf graphs/*
	rm -rf data/stats.json
	rm -rf data/augmented_directory

fclean: clean
	rm -rf $(VENV)
