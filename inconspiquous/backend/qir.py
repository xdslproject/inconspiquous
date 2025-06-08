# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
from __future__ import annotations

from io import StringIO
from typing import Any, Dict, TYPE_CHECKING

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation

if TYPE_CHECKING:
    from pyqir import BasicQisBuilder, SimpleModule

try:
    from pyqir import BasicQisBuilder, SimpleModule

    pyqir_available = True
except ImportError:
    pyqir_available = False
    BasicQisBuilder = None  # type: ignore
    SimpleModule = None  # type: ignore

from inconspiquous.dialects import qir


class QIRError(Exception):
    """Base class for QIR-related errors"""

    pass


class QIRBackendError(QIRError):
    """Error in QIR backend"""

    pass


class QIRUnsupportedOperationError(QIRError):
    """Operation not supported in QIR"""

    pass


def print_qir(module: ModuleOp, output: StringIO) -> None:
    """Print QIR assembly to output stream"""
    if not pyqir_available:
        output.write("; PyQIR not available. Install with: pip install pyqir\n")
        return

    backend = QIRBackend()
    try:
        qir_assembly = backend.emit_module(module)
        output.write(qir_assembly)
    except QIRError as e:
        output.write(f"; Error generating QIR: {e}\n")
        output.write("; Module structure:\n")
        for op in module.walk():
            if hasattr(op, "name"):
                output.write(f";   {op.name}\n")
    except Exception as e:
        output.write(f"; Unexpected error: {e}\n")
        output.write("; Module structure:\n")
        for op in module.walk():
            if hasattr(op, "name"):
                output.write(f";   {op.name}\n")


class QIRBackend:
    """Backend for converting QIR dialect to QIR assembly using PyQIR"""

    def __init__(self):
        if not pyqir_available:
            raise ImportError("PyQIR is required for QIR backend")

    def emit_module(self, module: ModuleOp) -> str:
        """Convert a module containing QIR dialect operations to QIR assembly"""
        try:
            if not pyqir_available or SimpleModule is None or BasicQisBuilder is None:
                raise ImportError("PyQIR is required for QIR backend")

            qir_module = SimpleModule("qir_circuit", num_qubits=10, num_results=10)
            builder = BasicQisBuilder(qir_module.builder)

            qubit_map: Dict[Any, int] = {}
            result_map: Dict[Any, int] = {}
            qubit_counter = 0
            result_counter = 0

            for op in module.walk():
                self._emit_operation(
                    op,
                    builder,
                    qubit_map,
                    result_map,
                    qir_module,
                    qubit_counter,
                    result_counter,
                )

                if isinstance(op, qir.QubitAllocateOp):
                    qubit_counter += 1
                elif isinstance(op, qir.MeasureOp):
                    result_counter += 1

            return str(qir_module.ir())
        except Exception as e:
            raise QIRBackendError(f"Failed to emit QIR: {e}")

    def _emit_operation(
        self,
        op: Operation,
        builder,  # type: ignore
        qubit_map: Dict[Any, int],
        result_map: Dict[Any, int],
        qir_module,  # type: ignore
        qubit_idx: int,
        result_idx: int,
    ) -> None:
        """Emit a single operation"""
        try:
            match op:
                case qir.QubitAllocateOp():
                    qubit_map[op.result] = qubit_idx

                case qir.QubitReleaseOp():
                    pass

                case qir.HGateOp():
                    if op.qubit in qubit_map:
                        qubit_idx = qubit_map[op.qubit]
                        builder.h(qir_module.qubits[qubit_idx])
                    else:
                        raise QIRBackendError(f"Qubit not found for H gate: {op.qubit}")

                case qir.XGateOp():
                    if op.qubit in qubit_map:
                        qubit_idx = qubit_map[op.qubit]
                        builder.x(qir_module.qubits[qubit_idx])
                    else:
                        raise QIRBackendError(f"Qubit not found for X gate: {op.qubit}")

                case qir.YGateOp():
                    if op.qubit in qubit_map:
                        qubit_idx = qubit_map[op.qubit]
                        builder.y(qir_module.qubits[qubit_idx])
                    else:
                        raise QIRBackendError(f"Qubit not found for Y gate: {op.qubit}")

                case qir.ZGateOp():
                    if op.qubit in qubit_map:
                        qubit_idx = qubit_map[op.qubit]
                        builder.z(qir_module.qubits[qubit_idx])
                    else:
                        raise QIRBackendError(f"Qubit not found for Z gate: {op.qubit}")

                case qir.SGateOp():
                    if op.qubit in qubit_map:
                        qubit_idx = qubit_map[op.qubit]
                        builder.s(qir_module.qubits[qubit_idx])
                    else:
                        raise QIRBackendError(f"Qubit not found for S gate: {op.qubit}")

                case qir.TGateOp():
                    if op.qubit in qubit_map:
                        qubit_idx = qubit_map[op.qubit]
                        builder.t(qir_module.qubits[qubit_idx])
                    else:
                        raise QIRBackendError(f"Qubit not found for T gate: {op.qubit}")

                case qir.CXGateOp():
                    if op.control in qubit_map and op.target in qubit_map:
                        control_idx = qubit_map[op.control]
                        target_idx = qubit_map[op.target]
                        builder.cx(
                            qir_module.qubits[control_idx],
                            qir_module.qubits[target_idx],
                        )
                    else:
                        raise QIRBackendError(
                            f"Qubits not found for CX gate: {op.control}, {op.target}"
                        )

                case qir.CZGateOp():
                    if op.control in qubit_map and op.target in qubit_map:
                        control_idx = qubit_map[op.control]
                        target_idx = qubit_map[op.target]
                        builder.cz(
                            qir_module.qubits[control_idx],
                            qir_module.qubits[target_idx],
                        )
                    else:
                        raise QIRBackendError(
                            f"Qubits not found for CZ gate: {op.control}, {op.target}"
                        )

                case qir.MeasureOp():
                    if op.qubit in qubit_map:
                        qubit_idx = qubit_map[op.qubit]
                        result: Any = builder.mz(  # type: ignore
                            qir_module.qubits[qubit_idx], qir_module.results[result_idx]
                        )
                        result_map[op.result] = result
                    else:
                        raise QIRBackendError(
                            f"Qubit not found for measurement: {op.qubit}"
                        )

                case qir.ReadResultOp():
                    # Result reading is handled automatically by PyQIR
                    pass

                case _:
                    # Skip operations we don't handle
                    pass
        except Exception as e:
            raise QIRBackendError(f"Failed to emit operation {op.name}: {e}")
