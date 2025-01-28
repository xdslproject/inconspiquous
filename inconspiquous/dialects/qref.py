from typing import ClassVar
from xdsl.dialects.builtin import i1
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyInt,
    IRDLOperation,
    IntVarConstraint,
    RangeOf,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    eq,
)
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait

from inconspiquous.dialects.gate import GateType
from inconspiquous.gates import GateAttr
from inconspiquous.dialects.qubit import BitType
from inconspiquous.constraints import SizedAttributeConstraint


@irdl_op_definition
class GateOp(IRDLOperation):
    name = "qref.gate"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = prop_def(SizedAttributeConstraint(GateAttr, _I))

    ins = var_operand_def(RangeOf(eq(BitType()), length=_I))

    assembly_format = "`<` $gate `>` $ins attr-dict"

    def __init__(self, gate: GateAttr, *ins: SSAValue | Operation):
        super().__init__(
            operands=[ins],
            properties={
                "gate": gate,
            },
        )


class DynGateOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qref import DynGateConst

        return (DynGateConst(),)


@irdl_op_definition
class DynGateOp(IRDLOperation):
    name = "qref.dyn_gate"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = operand_def(GateType.constr(_I))

    ins = var_operand_def(RangeOf(eq(BitType()), length=_I))

    assembly_format = "`<` $gate `>` $ins attr-dict"

    traits = traits_def(DynGateOpHasCanonicalizationPatterns())

    def __init__(self, gate: SSAValue | Operation, *ins: SSAValue | Operation):
        super().__init__(
            operands=[gate, ins],
        )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qref.measure"

    in_qubit = operand_def(BitType())

    out = result_def(i1)

    assembly_format = "$in_qubit attr-dict"

    def __init__(self, in_qubit: SSAValue | Operation):
        super().__init__(
            operands=(in_qubit,),
            result_types=(i1,),
        )


Qref = Dialect(
    "qref",
    [
        GateOp,
        DynGateOp,
        MeasureOp,
    ],
    [],
)
