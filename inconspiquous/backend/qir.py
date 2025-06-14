from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, TypeVar, TYPE_CHECKING, Protocol, runtime_checkable, IO

from xdsl.dialects.builtin import ModuleOp, FuncOp
from xdsl.ir import Operation, SSAValue, Region, Attribute

from inconspiquous.dialects import qir

# Import PyQIR first
from pyqir import BasicQisBuilder, SimpleModule

# Check if PyQIR is available
pyqir_available = importlib.util.find_spec("pyqir") is not None

if not pyqir_available:
    raise ImportError(
        "PyQIR is not installed. Please install it with 'pip install pyqir' or 'pip install inconspicuous[qir]'"
    )

# Then do TYPE_CHECKING imports
if TYPE_CHECKING:
    pass

__all__ = ["print_qir", "QIRBackend", "emit_qir_module", "pyqir_available"]

T = TypeVar("T")


class QIRBuilder(Protocol):
    def h(self, qubit: Any) -> None: ...
    def x(self, qubit: Any) -> None: ...
    def y(self, qubit: Any) -> None: ...
    def z(self, qubit: Any) -> None: ...
    def s(self, qubit: Any) -> None: ...
    def t(self, qubit: Any) -> None: ...
    def cx(self, control: Any, target: Any) -> None: ...
    def cz(self, control: Any, target: Any) -> None: ...
    def mz(self, qubit: Any, result: Any) -> Any: ...


@runtime_checkable
class QIRModule(Protocol):
    @property
    def qubits(self) -> list[Any]: ...
    @property
    def results(self) -> list[Any]: ...
    @property
    def builder(self) -> Any: ...
    def ir(self) -> str: ...


@dataclass
class QIRBackend:
    """Backend for converting QIR dialect to QIR assembly using PyQIR"""

    qubit_map: Dict[SSAValue, int] = field(default_factory=dict)
    result_map: Dict[SSAValue, Any] = field(default_factory=dict)
    qubit_counter: int = 0
    result_counter: int = 0

    def emit_module(self, module: ModuleOp) -> str:
        """Convert a module containing QIR dialect operations to QIR assembly"""
        qir_module = self.emit_qir_module(module)
        return str(qir_module.ir())

    def emit_qir_module(self, module: ModuleOp) -> QIRModule:
        """Convert a module containing QIR dialect operations to a QIR module object"""
        if not pyqir_available:
            raise ImportError(
                "PyQIR is not installed. Please install it with 'pip install pyqir' or 'pip install inconspicuous[qir]'"
            )

        # Check that module has exactly one function
        funcs = [op for op in module.ops if isinstance(op, FuncOp)]
        if len(funcs) != 1:
            raise ValueError("Module must contain exactly one function")
        func = funcs[0]

        # Check that function has exactly one block
        if not isinstance(func.body, Region):
            raise ValueError("Function body must be a region")
        if len(func.body.blocks) != 1:
            raise ValueError("Function must contain exactly one block")
        block = func.body.blocks[0]

        # Count qubits and results needed
        num_qubits = sum(1 for op in block.ops if isinstance(op, qir.QubitAllocateOp))
        num_results = sum(1 for op in block.ops if isinstance(op, qir.MeasureOp))

        # Get function name from attributes
        func_name = func.name.data if isinstance(func.name, Attribute) else "main"
        qir_module = SimpleModule(
            func_name, num_qubits=num_qubits, num_results=num_results
        )
        builder = BasicQisBuilder(qir_module.builder)

        # Reset state
        self.qubit_map.clear()
        self.result_map.clear()
        self.qubit_counter = 0
        self.result_counter = 0

        # Emit operations
        for op in block.ops:
            self._emit_operation(op, builder, qir_module)

        return qir_module

    def _emit_operation(
        self,
        op: Operation,
        builder: QIRBuilder,
        qir_module: QIRModule,
    ) -> None:
        """Emit a single operation"""
        match op:
            case qir.QubitAllocateOp():
                self.qubit_map[op.result] = self.qubit_counter
                self.qubit_counter += 1

            case qir.QubitReleaseOp():
                pass

            case qir.HGateOp():
                if op.qubit in self.qubit_map:
                    qubit_idx = self.qubit_map[op.qubit]
                    builder.h(qir_module.qubits[qubit_idx])

            case qir.XGateOp():
                if op.qubit in self.qubit_map:
                    qubit_idx = self.qubit_map[op.qubit]
                    builder.x(qir_module.qubits[qubit_idx])

            case qir.YGateOp():
                if op.qubit in self.qubit_map:
                    qubit_idx = self.qubit_map[op.qubit]
                    builder.y(qir_module.qubits[qubit_idx])

            case qir.ZGateOp():
                if op.qubit in self.qubit_map:
                    qubit_idx = self.qubit_map[op.qubit]
                    builder.z(qir_module.qubits[qubit_idx])

            case qir.SGateOp():
                if op.qubit in self.qubit_map:
                    qubit_idx = self.qubit_map[op.qubit]
                    builder.s(qir_module.qubits[qubit_idx])

            case qir.TGateOp():
                if op.qubit in self.qubit_map:
                    qubit_idx = self.qubit_map[op.qubit]
                    builder.t(qir_module.qubits[qubit_idx])

            case qir.CXGateOp():
                if op.control in self.qubit_map and op.target in self.qubit_map:
                    control_idx = self.qubit_map[op.control]
                    target_idx = self.qubit_map[op.target]
                    builder.cx(
                        qir_module.qubits[control_idx], qir_module.qubits[target_idx]
                    )

            case qir.CZGateOp():
                if op.control in self.qubit_map and op.target in self.qubit_map:
                    control_idx = self.qubit_map[op.control]
                    target_idx = self.qubit_map[op.target]
                    builder.cz(
                        qir_module.qubits[control_idx], qir_module.qubits[target_idx]
                    )

            case qir.MeasureOp():
                if op.qubit in self.qubit_map:
                    qubit_idx = self.qubit_map[op.qubit]
                    result = builder.mz(
                        qir_module.qubits[qubit_idx],
                        qir_module.results[self.result_counter],
                    )
                    self.result_map[op.result] = result
                    self.result_counter += 1

            case qir.ReadResultOp():
                # Result reading is handled automatically by PyQIR
                pass

            case _:
                raise ValueError(f"Unsupported operation: {op.name}")


def emit_qir_module(module: ModuleOp) -> QIRModule:
    """Convert a module containing QIR dialect operations to a QIR module object"""
    if not pyqir_available:
        raise ImportError(
            "PyQIR is not installed. Please install it with 'pip install pyqir' or 'pip install inconspicuous[qir]'"
        )

    backend = QIRBackend()
    return backend.emit_qir_module(module)


def print_qir(module: ModuleOp, output: IO[str]) -> None:
    """Print QIR assembly to output stream"""
    try:
        qir_module = emit_qir_module(module)
        output.write(str(qir_module.ir()))
    except Exception as e:
        output.write(f"; Error generating QIR: {e}\n")
        output.write("; Module structure:\n")
        for op in module.walk():
            if hasattr(op, "name"):
                output.write(f";   {op.name}\n")
