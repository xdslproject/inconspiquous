from __future__ import annotations

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    """QIR qubit type"""

    name = "qir.qubit"


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    """QIR result type"""

    name = "qir.result"


@irdl_op_definition
class QubitAllocateOp(IRDLOperation):
    """Allocate a qubit"""

    name = "qir.qubit_allocate"

    result = result_def(QubitType)

    assembly_format = "attr-dict"
    traits = traits_def(Pure())

    def __init__(self):
        super().__init__(result_types=(QubitType(),))


@irdl_op_definition
class QubitReleaseOp(IRDLOperation):
    """Release a qubit"""

    name = "qir.qubit_release"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class HGateOp(IRDLOperation):
    """Hadamard gate"""

    name = "qir.h"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"
    traits = traits_def(Pure())

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class XGateOp(IRDLOperation):
    """Pauli-X gate"""

    name = "qir.x"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"
    traits = traits_def(Pure())

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class YGateOp(IRDLOperation):
    """Pauli-Y gate"""

    name = "qir.y"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"
    traits = traits_def(Pure())

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class ZGateOp(IRDLOperation):
    """Pauli-Z gate"""

    name = "qir.z"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"
    traits = traits_def(Pure())

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class SGateOp(IRDLOperation):
    """S gate (phase gate)"""

    name = "qir.s"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"
    traits = traits_def(Pure())

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class TGateOp(IRDLOperation):
    """T gate"""

    name = "qir.t"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"
    traits = traits_def(Pure())

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class CXGateOp(IRDLOperation):
    """CNOT gate"""

    name = "qir.cx"

    control = operand_def(QubitType)
    target = operand_def(QubitType)

    assembly_format = "$control `,` $target attr-dict"
    traits = traits_def(Pure())

    def __init__(self, control: SSAValue | Operation, target: SSAValue | Operation):
        super().__init__(operands=(control, target))


@irdl_op_definition
class CZGateOp(IRDLOperation):
    """Controlled-Z gate"""

    name = "qir.cz"

    control = operand_def(QubitType)
    target = operand_def(QubitType)

    assembly_format = "$control `,` $target attr-dict"
    traits = traits_def(Pure())

    def __init__(self, control: SSAValue | Operation, target: SSAValue | Operation):
        super().__init__(operands=(control, target))


@irdl_op_definition
class MeasureOp(IRDLOperation):
    """Measure a qubit"""

    name = "qir.measure"

    qubit = operand_def(QubitType)
    result = result_def(ResultType)

    assembly_format = "$qubit attr-dict"

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,), result_types=(ResultType(),))


@irdl_op_definition
class ReadResultOp(IRDLOperation):
    """Read measurement result as boolean"""

    name = "qir.read_result"

    result_val = operand_def(ResultType)
    output = result_def(IntegerType(1))

    assembly_format = "$result_val attr-dict"
    traits = traits_def(Pure())

    def __init__(self, result_val: SSAValue | Operation):
        super().__init__(operands=(result_val,), result_types=(IntegerType(1),))


QIR = Dialect(
    "qir",
    [
        QubitAllocateOp,
        QubitReleaseOp,
        HGateOp,
        XGateOp,
        YGateOp,
        ZGateOp,
        SGateOp,
        TGateOp,
        CXGateOp,
        CZGateOp,
        MeasureOp,
        ReadResultOp,
    ],
    [
        QubitType,
        ResultType,
    ],
)
