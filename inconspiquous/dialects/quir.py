# quir.py - Skeleton for QUIR dialect (ported from TableGen)
from xdsl.ir import Dialect, Operation, Attribute
from xdsl.irdl import irdl_op_definition, Operand, result_def

class QubitType(Attribute):
    name = "quir.qubit"

class BitType(Attribute):
    name = "quir.bit"

@irdl_op_definition
class AllocateQubit(Operation):
    name = "quir.alloc_qubit"
    result = result_def(QubitType)

@irdl_op_definition
class Measure(Operation):
    name = "quir.measure"
    qubit = Operand(QubitType)
    result = result_def(BitType)

class QUIRDialect(Dialect):
    @property
    def name(self):
        return "quir"

    @property
    def operations(self):
        return [AllocateQubit, Measure]

    @property
    def types(self):
        return [QubitType, BitType]

    @property
    def attributes(self):
        return []
