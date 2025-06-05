from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    irdl_attr_definition,
    operand_def,
    result_def,
)
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute, SSAValue, Operation

# Qubit type for QIR
define_qir_type = True


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    name = "qir.qubit"


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    name = "qir.result"


# QIR Operations
@irdl_op_definition
class HOp(IRDLOperation):
    name = "qir.h"
    qubit = operand_def(QubitType)

    def __init__(self, qubit: SSAValue | Operation | None):
        super().__init__(operands=[qubit])


@irdl_op_definition
class XOp(IRDLOperation):
    name = "qir.x"
    qubit = operand_def(QubitType)

    def __init__(self, qubit: SSAValue | Operation | None):
        super().__init__(operands=[qubit])


@irdl_op_definition
class CNOTOp(IRDLOperation):
    name = "qir.cnot"
    control = operand_def(QubitType)
    target = operand_def(QubitType)

    def __init__(self, control: SSAValue | Operation | None, target: SSAValue | Operation | None):
        super().__init__(operands=[control, target])


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qir.measure"
    qubit = operand_def(QubitType)
    result = result_def(ResultType)

    def __init__(self, qubit: SSAValue | Operation | None):
        super().__init__(operands=[qubit], result_types=[ResultType()])


QIR = Dialect(
    "qir",
    [HOp, XOp, CNOTOp, MeasureOp],
    [QubitType, ResultType],
)
