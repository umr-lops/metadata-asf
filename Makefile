#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`

#* Installation
.PHONY: install
install:
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	ruff check --fix .
	black ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) pytest -c pyproject.toml tests/

.PHONY: check-codestyle
check-codestyle:
	ruff check .
	black --check ./

.PHONY: mypy
mypy:
	mypy --strict src/tests

.PHONY: lint
lint: test check-codestyle mypy

#* Documentation
.PHONY: docs
docs:
	sphinx-build -b html docs docs/_build/html

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypy-cache-remove:
	find . | grep -E ".mypy_cache|.ruff_cache" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/ dist/ *.egg-info src/*.egg-info

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypy-cache-remove pytestcache-remove coverage-remove

.PHONY: coverage-remove
coverage-remove:
	rm -f .coverage
