from typing import Any

try:
    import pyqir
except ImportError:
    pyqir = None  # type: ignore
from inconspiquous.dialects.qir import HOp, XOp, CNOTOp, MeasureOp


def print_qir_assembly(qir_module: Any) -> str:
    """
    Convert a qir dialect module to QIR assembly using PyQIR.
    """
    if pyqir is None:
        raise ImportError("pyqir is not installed")
    module = getattr(pyqir, "QirModule")(
        "qir_module", num_qubits=4, num_results=4
    )  # type: ignore
    builder = getattr(pyqir, "QirBuilder")(module)  # type: ignore

    # Walk the qir_module and emit QIR ops using builder
    for op in getattr(qir_module, 'ops', []):
        if isinstance(op, HOp):
            getattr(builder, "h")(0)  # type: ignore
        elif isinstance(op, XOp):
            getattr(builder, "x")(0)  # type: ignore
        elif isinstance(op, CNOTOp):
            getattr(builder, "cx")(0, 1)  # type: ignore
        elif isinstance(op, MeasureOp):
            getattr(builder, "measure")(0, 0)  # type: ignore
        # ...add more ops as needed...

    # Return the QIR assembly as a string
    return getattr(module, "ir")()  # type: ignore
