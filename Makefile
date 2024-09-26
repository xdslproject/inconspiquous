MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# allow overriding the name of the venv directory
VENV_DIR ?= venv

# make tasks run all commands in a single shell
.ONESHELL:

# these targets don't produce files:
.PHONY: ${VENV_DIR}/ venv clean clean-caches filecheck pytest tests

# set up the venv with all dependencies for development
${VENV_DIR}/: requirements.txt
	python3 -m venv ${VENV_DIR}
	. ${VENV_DIR}/bin/activate
	python3 -m pip --require-virtualenv install -r requirements.txt

# make sure `make venv` always works no matter what $VENV_DIR is
venv: ${VENV_DIR}/

# remove all caches
clean-caches:
	rm -rf .pytest_cache *.egg-info .coverage.*
	find . -type f -name "*.cover" -delete

# remove all caches and the venv
clean: clean-caches
	rm -rf ${VENV_DIR}

# run filecheck tests
filecheck:
	lit -vv tests/filecheck --order=smart --timeout=20

# run pytest tests
pytest:
	pytest tests -W error -vv

# run all tests
tests: pytest filecheck pyright
	@echo All tests done.

# set up all precommit hooks
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
precommit:
	pre-commit run --all

# run pyright on all files in the current git commit
pyright:
	pyright $(shell git diff --staged --name-only  -- '*.py')
