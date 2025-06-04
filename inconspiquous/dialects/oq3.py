# oq3.py - Skeleton for OQ3 dialect (ported from TableGen)
from xdsl.ir import Dialect, Operation, Attribute
from xdsl.irdl import irdl_op_definition, Operand, result_def

# Define oq3.qubit and oq3.bit types
class QubitType(Attribute):
    name = "oq3.qubit"

class BitType(Attribute):
    name = "oq3.bit"

# oq3.gate operation: takes a qubit, returns a qubit
@irdl_op_definition
class Gate(Operation):
    name = "oq3.gate"
    qubit = Operand(QubitType)
    result = result_def(QubitType)

# oq3.measure operation: takes a qubit, returns a bit
@irdl_op_definition
class Measure(Operation):
    name = "oq3.measure"
    qubit = Operand(QubitType)
    result = result_def(BitType)

# oq3.reset operation: takes a qubit, returns a qubit (reset to |0>)
@irdl_op_definition
class Reset(Operation):
    name = "oq3.reset"
    qubit = Operand(QubitType)
    result = result_def(QubitType)

# oq3.barrier operation: takes a qubit, returns a qubit (barrier for scheduling)
@irdl_op_definition
class Barrier(Operation):
    name = "oq3.barrier"
    qubit = Operand(QubitType)
    result = result_def(QubitType)

# Example attribute: classical condition for conditional gate
class ConditionAttr(Attribute):
    name = "oq3.condition"
    # In practice, logic to store bit/classical value needs to be added

# oq3.cond_gate operation: conditional gate (if bit==1 then apply gate)
@irdl_op_definition
class CondGate(Operation):
    name = "oq3.cond_gate"
    bit = Operand(BitType)
    qubit = Operand(QubitType)
    result = result_def(QubitType)
    # cond attribute logic to be implemented if needed

class OQ3Dialect(Dialect):
    @property
    def name(self):
        return "oq3"

    @property
    def operations(self):
        return [Gate, Measure, Reset, Barrier, CondGate]

    @property
    def types(self):
        return [QubitType, BitType]

    @property
    def attributes(self):
        return [ConditionAttr]
    # Register the dialect if needed
