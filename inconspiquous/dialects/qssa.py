from typing import ClassVar
from xdsl.dialects.builtin import i1
from xdsl.ir import Dialect, Operation, SSAValue, VerifyException
from xdsl.irdl import (
    EqAttrConstraint,
    IRDLOperation,
    RangeConstraint,
    RangeOf,
    RangeVarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)

from inconspiquous.gates import GateAttr
from inconspiquous.dialects.qubit import BitType


@irdl_op_definition
class GateOp(IRDLOperation):
    name = "qssa.gate"

    gate = prop_def(GateAttr)

    _T: ClassVar[RangeConstraint] = RangeVarConstraint(
        "T", RangeOf(EqAttrConstraint(BitType()))
    )

    ins = var_operand_def(_T)

    outs = var_result_def(_T)

    assembly_format = "`<` $gate `>` $ins attr-dict `:` type($ins)"

    def __init__(self, gate: GateAttr, *ins: SSAValue | Operation):
        super().__init__(
            operands=[ins],
            properties={
                "gate": gate,
            },
            result_types=tuple(BitType() for _ in ins),
        )

    def verify_(self) -> None:
        if len(self.ins) != self.gate.num_qubits:
            raise VerifyException(
                f"Gate {self.gate} expected {self.gate.num_qubits} input qubits but got {len(self.ins)}."
            )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qssa.measure"

    in_qubit = operand_def(BitType())

    out = result_def(i1)

    out_qubit = result_def(BitType())

    assembly_format = "$in_qubit attr-dict"

    def __init__(self, in_qubit: SSAValue | Operation):
        super().__init__(
            operands=(in_qubit,),
        )


Qssa = Dialect(
    "qssa",
    [
        GateOp,
        MeasureOp,
    ],
    [],
)
