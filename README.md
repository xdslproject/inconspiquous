# inconspiquous
A testing ground for quantum computing compilation ideas in [xdsl](https://xdsl.dev)

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

## Usage
Inconspiquous is built on top of [xdsl](https://github.com/xdslproject/xdsl), which is itself based on (and broadly compatible with) the MLIR compiler framework. Tutorials for this framework can be found at these respective projects.

This repository adds extra dialects and transformations for working with quantum programs. The repository is primarily tested using filecheck tests, found in the [test directory](tests/filecheck). These tests also serve as examples of the intermediate representations defined by this tool. The examples can be run using `quopt`, which can be run via `uv run quopt`, or by entering the virtual environment created by `uv`. It is also possible to view transformations using the interactive `quopt-gui` tool.

## Contributing
Contributions should be submitted by github pull requests. Code must be formatted and linted by `ruff`, typecheckable by `pyright`, and have accompanying tests. We recommend installing `pre-commit` (which can be done by `make precommit-install`) to ensure code remains properly formatted.

## Discussion
For any discussion or help, feel free to create a topic/send a message on the [xdsl zulip](https://xdsl.zulipchat.com/).
