.PHONY: install test lint benchmark check clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests benchmarks

benchmark:
	$(PYTHON) benchmarks/benchmark_binomial.py

check:
	$(PYTHON) -m compileall src
	$(MAKE) test
	$(MAKE) lint

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
