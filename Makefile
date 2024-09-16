MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# allow overriding the name of the venv directory
VENV_DIR ?= venv

# use a default prefix for coverage data files
COVERAGE_FILE ?= .coverage

# use different coverage data file per coverage run, otherwise combine complains
TESTS_COVERAGE_FILE = ${COVERAGE_FILE}.tests

# make tasks run all commands in a single shell
.ONESHELL:

# these targets don't produce files:
.PHONY: ${VENV_DIR}/ venv clean clean-caches filecheck pytest pytest-nb tests-toy tests
.PHONY: coverage-report-html coverage-report-md

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
tests: pytest tests-toy filecheck pytest-nb tests-marimo tests-marimo-mlir pyright
	@echo All tests done.

# set up all precommit hooks
precommit-install:
	pre-commit install

# run all precommit hooks and apply them
precommit:
	pre-commit run --all

# run pyright on all files in the current git commit
pyright:
    # We make sure to generate the python typing stubs before running pyright
	xdsl-stubgen
	pyright $(shell git diff --staged --name-only  -- '*.py')

# run coverage over all tests and combine data files
coverage: coverage-tests coverage-filecheck-tests
	coverage combine --append

# run coverage over tests
coverage-tests:
	COVERAGE_FILE=${TESTS_COVERAGE_FILE} pytest -W error --cov --cov-config=.coveragerc

# run coverage over filecheck tests
coverage-filecheck-tests:
	lit -v tests/filecheck/ -DCOVERAGE

# generate html coverage report
coverage-report-html:
	coverage html

# generate markdown coverage report
coverage-report-md:
	coverage report --format=markdown