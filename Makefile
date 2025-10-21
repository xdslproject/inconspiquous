MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# allow overriding the name of the venv directory
VENV_DIR ?= .venv

# use activated venv if any
export UV_PROJECT_ENVIRONMENT=$(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),$(VENV_DIR))

# allow overriding which extras are installed
VENV_EXTRAS ?= --all-extras
VENV_GROUPS ?= --all-groups

# default lit options
LIT_OPTIONS ?= -v --order=smart

# make tasks run all commands in a single shell
.ONESHELL:

.PHONY: uv-installed
uv-installed:
	@command -v uv &> /dev/null ||\
		(echo "UV doesn't seem to be installed, try the following instructions:" &&\
		echo "https://docs.astral.sh/uv/getting-started/installation/" && false)

# set up the venv with all dependencies for development
.PHONY: ${VENV_DIR}/
${VENV_DIR}/: uv-installed
	uv sync ${VENV_EXTRAS} ${VENV_GROUPS}

# make sure `make venv` also works correctly
.PHONY: venv
venv: ${VENV_DIR}/

# remove all caches
.PHONY: clean-caches
clean-caches:
	rm -rf .pytest_cache *.egg-info .coverage.*
	find . -type f -name "*.cover" -delete

# remove all caches and the venv
.PHONY: clean
clean: clean-caches
	rm -rf ${VENV_DIR}

# run filecheck tests
.PHONY: filecheck
filecheck: uv-installed
	uv run lit ${LIT_OPTIONS} tests/filecheck

# run pytest tests
.PHONY: pytest
pytest: uv-installed
	uv run pytest tests -W error -vv

# run all tests
.PHONY: tests
tests: pytest filecheck pyright
	@echo All tests done.

# set up all precommit hooks
.PHONY: precommit-install
precommit-install: uv-installed
	uv run pre-commit install

# run all precommit hooks and apply them
.PHONY: precommit
precommit: uv-installed
	uv run pre-commit run --all

# run pyright on all files in the current git commit
.PHONY: pyright
pyright: uv-installed
	uv run pyright $(shell git diff --staged --name-only  -- '*.py')
