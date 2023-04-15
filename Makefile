.PHONY: test
test:
  pytest -m "not training"
