# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .
	
# Test
.PHONY: test
test:
	pytest -m "not training"
