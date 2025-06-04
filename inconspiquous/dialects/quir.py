# quir.py - Skeleton for QUIR dialect (ported from TableGen)
from typing import Iterator
from xdsl.ir import Dialect, Attribute
from xdsl.irdl import irdl_op_definition, IRDLOperation, Operand, result_def

class QubitType(Attribute):
    name = "quir.qubit"

class BitType(Attribute):
    name = "quir.bit"

@irdl_op_definition
class AllocateQubit(IRDLOperation):
    name = "quir.alloc_qubit"
    result = result_def(QubitType)

@irdl_op_definition
class Measure(IRDLOperation):
    name = "quir.measure"
    qubit = Operand(QubitType)
    result = result_def(BitType)

class QUIRDialect(Dialect):
    @property
    def name(self) -> str:
        return "quir"

    @property
    def operations(self) -> Iterator[type[IRDLOperation]]:
        return iter([AllocateQubit, Measure])

    @property
    def types(self) -> Iterator[type[Attribute]]:
        return iter([QubitType, BitType])

    @property
    def attributes(self) -> Iterator[type[Attribute]]:
        return iter([])
