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
    eq,
    var_result_def,
    result_def,
    region_def,
)
from xdsl.traits import IsTerminator
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait

from inconspiquous.dialects.gate import GateType
from inconspiquous.dialects.measurement import CompBasisMeasurementAttr, MeasurementType
from inconspiquous.gates import GateAttr
from inconspiquous.dialects.qu import BitType
from inconspiquous.constraints import SizedAttributeConstraint
from inconspiquous.measurement import MeasurementAttr


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

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = prop_def(
        SizedAttributeConstraint(MeasurementAttr, _I),
        default_value=CompBasisMeasurementAttr(),
    )

    in_qubits = var_operand_def(RangeOf(eq(BitType()), length=_I))

    outs = var_result_def(RangeOf(eq(i1), length=_I))

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
        from inconspiquous.transforms.canonicalization.qref import (
            DynMeasureConst,
        )

        return (DynMeasureConst(),)


@irdl_op_definition
class DynMeasureOp(IRDLOperation):
    name = "qref.dyn_measure"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = operand_def(MeasurementType.constr(_I))

    in_qubits = var_operand_def(RangeOf(eq(BitType()), length=_I))

    outs = var_result_def(RangeOf(eq(i1), length=_I))

    assembly_format = "`<` $measurement `>` $in_qubits attr-dict"

    traits = traits_def(DynMeasureOpHasCanonicalizationPatterns())

    def __init__(
        self, *in_qubits: SSAValue | Operation, measurement: SSAValue | Operation
    ):
        super().__init__(
            operands=[measurement, in_qubits],
            result_types=(tuple(i1 for _ in in_qubits),),
        )


@irdl_op_definition
class CircuitOp(IRDLOperation):
    name = "qref.circuit"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    body = region_def("single_block", entry_args=RangeOf(eq(BitType()), length=_I))
    result = result_def(GateType.constr(_I))

    def __init__(self, num_qubits: int, region: Region | None = None):
        if region is None:
            region = Region(Block(arg_types=[BitType() for _ in range(num_qubits)]))

        super().__init__(
            regions=(region,),
            result_types=(GateType(num_qubits),),
        )

    def verify_(self):
        assert len(self.body.blocks) == 1, "Circuit must have exactly one block"

        entry_block = self.body.blocks[0]
        num_args = len(entry_block.args)
        expected_qubits = self.result.type.num_qubits.value.data

        assert num_args == expected_qubits, f"Expected {expected_qubits} block arguments, got {num_args}"

        # Check that all block arguments are qubit types
        for arg in entry_block.args:
            assert isinstance(arg.type, BitType), f"Block argument must be !qu.bit, got {arg.type}"

        # Check terminator
        if entry_block.ops:
            terminator = entry_block.last_op
            assert isinstance(terminator, ReturnOp), "Circuit must be terminated by qref.return"


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "qref.return"

    traits = traits_def(IsTerminator())

    def __init__(self):
        super().__init__()


Qref = Dialect(
    "qref",
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
