# Makefile
SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands."
	@echo "venv  : Creates a virtual environment."
	@echo "style : Executes style formating."
	@echo "clean : Cleans all uncessary files."
	@echo "test  : Executes test on code, data and models."
	
# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .

# Environment
.ONESHELL:
venv:
	python3 -m venv inn \
	source inn/bin/activate && \
	python3 -m pip install --upgrade pip setuptools wheel && \
	python3 -m pip install -e ".[dev]" && \
	pre-commit install && \
	pre-commit autoupdate
	
# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage
	
# Test
.PHONY: test
test:
	pytest -m "not training"
