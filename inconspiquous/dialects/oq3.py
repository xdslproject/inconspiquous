from typing import Iterator
from xdsl.ir import Dialect, Attribute
from xdsl.irdl import irdl_op_definition, IRDLOperation, Operand, result_def

# Define oq3.qubit and oq3.bit types
class QubitType(Attribute):
    name = "oq3.qubit"

class BitType(Attribute):
    name = "oq3.bit"

# oq3.gate operation: takes a qubit, returns a qubit
@irdl_op_definition
class Gate(IRDLOperation):
    name = "oq3.gate"
    qubit = Operand(QubitType())
    result = result_def(QubitType())

# oq3.measure operation: takes a qubit, returns a bit
@irdl_op_definition
class Measure(IRDLOperation):
    name = "oq3.measure"
    qubit = Operand(QubitType())
    result = result_def(BitType())

# oq3.reset operation: takes a qubit, returns a qubit (reset to |0>)
@irdl_op_definition
class Reset(IRDLOperation):
    name = "oq3.reset"
    qubit = Operand(QubitType())
    result = result_def(QubitType())

# oq3.barrier operation: takes a qubit, returns a qubit (barrier for scheduling)
@irdl_op_definition
class Barrier(IRDLOperation):
    name = "oq3.barrier"
    qubit = Operand(QubitType())
    result = result_def(QubitType())

# Example attribute: classical condition for conditional gate
class ConditionAttr(Attribute):
    name = "oq3.condition"
    # In practice, logic to store bit/classical value needs to be added

# oq3.cond_gate operation: conditional gate (if bit==1 then apply gate)
@irdl_op_definition
class CondGate(IRDLOperation):
    name = "oq3.cond_gate"
    bit = Operand(BitType())
    qubit = Operand(QubitType())
    result = result_def(QubitType())
    # cond attribute logic to be implemented if needed

class OQ3Dialect(Dialect):
    @property
    def name(self) -> str:
        return "oq3"

    @property
    def operations(self) -> Iterator[type[IRDLOperation]]:
        return iter([Gate, Measure, Reset, Barrier, CondGate])

    @property
    def types(self) -> Iterator[type[Attribute]]:
        return iter([QubitType, BitType])

    @property
    def attributes(self) -> Iterator[type[Attribute]]:
        return iter([ConditionAttr])
    # Register the dialect if needed
