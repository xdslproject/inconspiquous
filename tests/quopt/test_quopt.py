from contextlib import redirect_stdout
from io import StringIO
from inconspiquous.tools.quopt_main import QuoptMain


def test_empty_program():
    filename = "tests/quopt/empty_program.mlir"
    opt = QuoptMain(args=[filename])

    f = StringIO("")
    with redirect_stdout(f):
        opt.run()

    with open(filename) as file:
        expected = file.read()
        assert f.getvalue().strip() == expected.strip()
