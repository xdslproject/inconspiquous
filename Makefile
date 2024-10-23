MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# make tasks run all commands in a single shell
.ONESHELL:

# these targets don't produce files:
.PHONY: ${VENV_DIR}/ venv clean clean-caches filecheck pytest tests

# remove all caches
clean-caches:
	rm -rf .pytest_cache *.egg-info .coverage.*
	find . -type f -name "*.cover" -delete

# remove all caches and the venv
clean: clean-caches
	rm -rf ${VENV_DIR}

# run filecheck tests
filecheck:
	uv run lit -vv tests/filecheck --order=smart --timeout=20

# run pytest tests
pytest:
	uv run pytest tests -W error -vv

# run all tests
tests: pytest filecheck pyright
	@echo All tests done.

# set up all precommit hooks
precommit-install:
	uv run pre-commit install

# run all precommit hooks and apply them
precommit:
	uv run pre-commit run --all

# run pyright on all files in the current git commit
pyright:
	uv run pyright $(shell git diff --staged --name-only  -- '*.py')
