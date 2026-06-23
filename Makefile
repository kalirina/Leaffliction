PYTHON = python3
VENV = .venv

.PHONY: all install clean

all: install

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -r requirements.txt

clean:
	rm -rf graphs/*

fclean: clean
	rm -rf $(VENV)
