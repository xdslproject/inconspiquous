# inconspiquous
A testing ground for quantum computing compilation ideas in xdsl

## Installation
This tool uses [uv](https://docs.astral.sh/uv/) to build. The tool can be installed by cloning and running:
```bash
uv sync
```
The cli tool can be run by `uv run quopt` or by entering the generated virtual environment.

```bash
source .venv\bin\activate
```

For tests to work the dev dependencies should be installed:
```bash
uv sync --dev
```

To run tests:
```bash
make tests
```