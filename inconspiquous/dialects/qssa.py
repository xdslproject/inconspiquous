from typing import ClassVar
from xdsl.dialects.builtin import i1
from xdsl.ir import Dialect, Operation, SSAValue
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
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait

from inconspiquous.gates import GateAttr
from inconspiquous.gates.constraints import DynGateConstraint, GateConstraint
from inconspiquous.dialects.qubit import BitType


@irdl_op_definition
class GateOp(IRDLOperation):
    name = "qssa.gate"

    _Q: ClassVar[RangeConstraint] = RangeVarConstraint(
        "Q", RangeOf(EqAttrConstraint(BitType()))
    )

    gate = prop_def(GateConstraint(_Q))

    ins = var_operand_def(_Q)

    outs = var_result_def(_Q)

    assembly_format = "`<` $gate `>` $ins attr-dict `:` type($ins)"

    def __init__(self, gate: GateAttr, *ins: SSAValue | Operation):
        super().__init__(
            operands=[ins],
            properties={
                "gate": gate,
            },
            result_types=(tuple(BitType() for _ in ins),),
        )


class DynGateOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qssa import (
            DynGateConst,
            DynGateCompose,
        )

        return (DynGateConst(), DynGateCompose())


@irdl_op_definition
class DynGateOp(IRDLOperation):
    name = "qssa.dyn_gate"

    _Q: ClassVar[RangeConstraint] = RangeVarConstraint(
        "Q", RangeOf(EqAttrConstraint(BitType()))
    )

    ins = var_operand_def(_Q)

    # Operands must be in this order for verification
    gate = operand_def(DynGateConstraint(_Q))

    outs = var_result_def(_Q)

    assembly_format = "`<` $gate `>` $ins attr-dict `:` type($ins)"

    traits = traits_def(DynGateOpHasCanonicalizationPatterns())

    def __init__(self, gate: SSAValue | Operation, *ins: SSAValue | Operation):
        super().__init__(
            operands=[ins, gate],
            result_types=(tuple(BitType() for _ in ins),),
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
            result_types=(BitType(),),
        )


Qssa = Dialect(
    "qssa",
    [
        GateOp,
        DynGateOp,
        MeasureOp,
    ],
    [],
)
