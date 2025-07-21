from typing import ClassVar
from xdsl.dialects.builtin import i1
from xdsl.ir import Dialect, Operation, SSAValue, Block, Region
from xdsl.irdl import (
    AnyInt,
    IRDLOperation,
    IntVarConstraint,
    RangeOf,
    irdl_op_definition,
    operand_def,
    prop_def,
    traits_def,
    var_operand_def,
    var_result_def,
    result_def,
    region_def,
    eq,
)
from xdsl.traits import IsTerminator, HasParent
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait

from inconspiquous.dialects.gate import GateType
from inconspiquous.dialects.measurement import CompBasisMeasurementAttr, MeasurementType
from inconspiquous.gates import GateAttr
from inconspiquous.dialects.qu import BitType
from inconspiquous.constraints import SizedAttributeConstraint
from inconspiquous.measurement import MeasurementAttr


class GateOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qssa import GateIdentity

        return (GateIdentity(),)


@irdl_op_definition
class GateOp(IRDLOperation):
    name = "qssa.gate"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = prop_def(SizedAttributeConstraint(GateAttr, _I))

    ins = var_operand_def(RangeOf(eq(BitType())).of_length(_I))

    outs = var_result_def(RangeOf(eq(BitType())).of_length(_I))

    assembly_format = "`<` $gate `>` $ins attr-dict"

    traits = traits_def(GateOpHasCanonicalizationPatterns())

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

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = operand_def(GateType.constr(_I))

    ins = var_operand_def(RangeOf(eq(BitType())).of_length(_I))

    outs = var_result_def(RangeOf(eq(BitType())).of_length(_I))

    assembly_format = "`<` $gate `>` $ins attr-dict"

    traits = traits_def(DynGateOpHasCanonicalizationPatterns())

    def __init__(self, gate: SSAValue | Operation, *ins: SSAValue | Operation):
        super().__init__(
            operands=[gate, ins],
            result_types=(tuple(BitType() for _ in ins),),
        )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qssa.measure"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = prop_def(
        SizedAttributeConstraint(MeasurementAttr, _I),
        default_value=CompBasisMeasurementAttr(),
    )

    in_qubits = var_operand_def(RangeOf(eq(BitType())).of_length(_I))

    outs = var_result_def(RangeOf(eq(i1)).of_length(_I))

    assembly_format = "(`` `<` $measurement^ `>`)? $in_qubits attr-dict"

    def __init__(
        self,
        *in_qubits: SSAValue | Operation,
        measurement: MeasurementAttr = CompBasisMeasurementAttr(),
    ):
        super().__init__(
            properties={
                "measurement": measurement,
            },
            operands=(in_qubits,),
            result_types=((i1,) * len(in_qubits)),
        )


class DynMeasureOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qssa import (
            DynMeasureConst,
        )

        return (DynMeasureConst(),)


@irdl_op_definition
class DynMeasureOp(IRDLOperation):
    name = "qssa.dyn_measure"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = operand_def(MeasurementType.constr(_I))

    in_qubits = var_operand_def(RangeOf(eq(BitType())).of_length(_I))

    outs = var_result_def(RangeOf(eq(i1)).of_length(_I))

    assembly_format = "`<` $measurement `>` $in_qubits attr-dict"

    traits = traits_def(DynMeasureOpHasCanonicalizationPatterns())

    def __init__(
        self,
        *in_qubits: SSAValue | Operation,
        measurement: SSAValue | Operation,
    ):
        super().__init__(
            operands=[measurement, in_qubits],
            result_types=(tuple(i1 for _ in in_qubits),),
        )


@irdl_op_definition
class CircuitOp(IRDLOperation):
    name = "qssa.circuit"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    body = region_def("single_block", entry_args=RangeOf(eq(BitType())).of_length(_I))
    result = result_def(GateType.constr(_I))

    assembly_format = "`(` `)` `(` $body `)` `:` `(` `)` `->` type($result) attr-dict"

    def __init__(self, num_qubits: int, region: Region | None = None):
        if region is None:
            region = Region(Block(arg_types=[BitType() for _ in range(num_qubits)]))

        super().__init__(
            regions=(region,),
            result_types=(GateType(num_qubits),),
        )

    def verify_(self):
        # Check terminator
        entry_block = self.body.blocks[0]
        if entry_block.ops:
            terminator = entry_block.last_op
            assert isinstance(terminator, ReturnOp), (
                "qssa.circuit must be terminated by qssa.return"
            )
            # Check that qssa.return has the correct number of operands
            expected_num_qubits = len(entry_block.args)
            actual_num_operands = len(terminator.args)
            assert actual_num_operands == expected_num_qubits, (
                f"qssa.return must have {expected_num_qubits} operands "
                f"but has {actual_num_operands}"
            )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "qssa.return"

    args = var_operand_def(BitType)

    traits = traits_def(HasParent(CircuitOp), IsTerminator())
    assembly_format = "$args attr-dict"

    def __init__(self, *operands: SSAValue | Operation):
        super().__init__(
            operands=(operands,),
        )


Qssa = Dialect(
    "qssa",
    [
        GateOp,
        DynGateOp,
        MeasureOp,
        DynMeasureOp,
        CircuitOp,
        ReturnOp,
    ],
    [],
)
