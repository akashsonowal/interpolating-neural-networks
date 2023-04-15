# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .
	
# Environment
.ONESHELL:
venv:
	conda create --name inn python=3.8 \
	conda activate inn && \
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
