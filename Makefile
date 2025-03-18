# Load environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export $(shell sed 's/=.*//' .env)
else
    $(warning .env file not found. Environment variables not loaded.)
endif

# Variables
PYTHON      ?= python3
PIP         ?= pip3
DEP_MNGR    ?= poetry
DOCS_DIR   ?= docs

# Directories and files to clean
CACHE_DIRS  = .mypy_cache .pytest_cache .ruff_cache
COVERAGE    = .coverage htmlcov coverage.xml
DIST_DIRS   = dist junit
TMP_DIRS   = site

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show help for all targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup & Installation
.PHONY: setup
setup: ## Install system dependencies and dependency manager (e.g., Poetry)
	sudo apt-get update
	sudo apt-get install -y python3-pip
	$(PIP) install --upgrade pip
	$(PIP) install $(DEP_MNGR)

.PHONY: install
install: ## Install Python dependencies
	$(DEP_MNGR) install --all-extras --no-interaction

# Quality & Testing
.PHONY: test
test: ## Run tests
	$(DEP_MNGR) run pytest

.PHONY: lint
lint: ## Run linter checks
	$(DEP_MNGR) run ruff check --fix

.PHONY: format
format: ## Format code
	$(DEP_MNGR) run ruff format

.PHONY: typecheck
typecheck: ## Typecheck code
	$(DEP_MNGR) run mypy .

.PHONY: precommit
precommit: ## Run pre-commit hooks
	$(DEP_MNGR) run pre-commit run --all-files

.PHONY: precommit-install
precommit-install: ## Install pre-commit hooks
	$(DEP_MNGR) run pre-commit install

# Documentation
.PHONY: docs
docs: ## Build documentation
	$(DEP_MNGR) run mkdocs build

# Build & Publish
.PHONY: build
build: ## Build distributions
	$(DEP_MNGR) build

.PHONY: publish
publish: ## Publish to PyPI (requires PYPI_TOKEN)
	$(DEP_MNGR) config pypi-token.pypi $(PYPI_TOKEN)
	$(DEP_MNGR) publish --build

# Maintenance
.PHONY: clean
clean: ## Remove caches and build artifacts
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf $(CACHE_DIRS) $(COVERAGE) $(DIST_DIRS) $(TMP_DIRS)
