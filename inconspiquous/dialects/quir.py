# quir.py - Skeleton for QUIR dialect (ported from TableGen)
from xdsl.dialects.builtin import Dialect, Operation, Attribute
from xdsl.irdl import irdl_op_definition, Operand, result_def

# Define quir.qubit and quir.bit types
class QubitType(Attribute):
    name = "quir.qubit"

class BitType(Attribute):
    name = "quir.bit"

# quir.alloc_qubit operation: returns a qubit
@irdl_op_definition
class AllocateQubit(Operation):
    name = "quir.alloc_qubit"
    result = result_def(QubitType)

# quir.measure operation: takes a qubit, returns a bit
@irdl_op_definition
class Measure(Operation):
    name = "quir.measure"
    qubit = Operand(QubitType)
    result = result_def(BitType)

class QUIRDialect(Dialect):
    name = "quir"
    operations = [AllocateQubit, Measure]
    types = [QubitType, BitType]
    # attributes = []

# Register the dialect if needed
