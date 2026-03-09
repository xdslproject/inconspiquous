from typing import ClassVar

from xdsl.dialects.builtin import i1
from xdsl.interfaces import HasCanonicalizationPatternsInterface
from xdsl.ir import Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    EqIntConstraint,
    IntVarConstraint,
    IRDLOperation,
    RangeOf,
    eq,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasParent, IsTerminator

from inconspiquous.constraints import SizedAttributeConstraint
from inconspiquous.dialects.instrument import (
    InstrumentAttr,
    InstrumentConstraint,
    InstrumentType,
)
from inconspiquous.dialects.measurement import CompBasisMeasurementAttr, MeasurementType
from inconspiquous.dialects.qu import BitType
from inconspiquous.measurement import MeasurementAttr


@irdl_op_definition
class ApplyOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    name = "qref.apply"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = prop_def(
        InstrumentConstraint(_I, RangeOf(AnyAttr()).of_length(EqIntConstraint(0)))
    )

    ins = var_operand_def(RangeOf(eq(BitType())).of_length(_I))

    assembly_format = "`<` $gate `>` $ins attr-dict"

    def __init__(self, gate: InstrumentAttr, *ins: SSAValue | Operation):
        super().__init__(
            operands=[ins],
            properties={
                "gate": gate,
            },
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qref import GateIdentity

        return (GateIdentity(),)


@irdl_op_definition
class DynApplyOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    name = "qref.dyn_apply"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = operand_def(
        InstrumentType.constr(_I, RangeOf(AnyAttr()).of_length(EqIntConstraint(0)))
    )

    ins = var_operand_def(RangeOf(eq(BitType())).of_length(_I))

    assembly_format = "`<` $gate `>` $ins attr-dict"

    def __init__(self, gate: SSAValue | Operation, *ins: SSAValue | Operation):
        super().__init__(
            operands=[gate, ins],
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qref import (
            DynGateCompose,
            DynGateConst,
        )

        return (DynGateConst(), DynGateCompose())


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qref.measure"

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


@irdl_op_definition
class DynMeasureOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    name = "qref.dyn_measure"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = operand_def(MeasurementType.constr(_I))

    in_qubits = var_operand_def(RangeOf(eq(BitType())).of_length(_I))

    outs = var_result_def(RangeOf(eq(i1)).of_length(_I))

    assembly_format = "`<` $measurement `>` $in_qubits attr-dict"

    def __init__(
        self, *in_qubits: SSAValue | Operation, measurement: SSAValue | Operation
    ):
        super().__init__(
            operands=[measurement, in_qubits],
            result_types=(tuple(i1 for _ in in_qubits),),
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qref import (
            DynMeasureConst,
        )

        return (DynMeasureConst(),)


@irdl_op_definition
class CircuitOp(IRDLOperation):
    name = "qref.circuit"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    body = region_def("single_block", entry_args=RangeOf(eq(BitType())).of_length(_I))
    result = result_def(
        InstrumentType.constr(_I, RangeOf(AnyAttr()).of_length(EqIntConstraint(0)))
    )

    assembly_format = "`(` `)` `(` $body `)` `:` `(` `)` `->` type($result) attr-dict"

    def __init__(self, num_qubits: int, region: Region | None = None):
        if region is None:
            region = Region(Block(arg_types=[BitType() for _ in range(num_qubits)]))

        super().__init__(
            regions=(region,),
            result_types=(InstrumentType(num_qubits),),
        )

    def verify_(self):
        # Check terminator
        entry_block = self.body.blocks[0]
        if entry_block.ops:
            terminator = entry_block.last_op
            assert isinstance(terminator, ReturnOp), (
                "qref.circuit must be terminated by qref.return"
            )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "qref.return"

    traits = traits_def(HasParent(CircuitOp), IsTerminator())
    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__()


Qref = Dialect(
    "qref",
    [
        ApplyOp,
        DynApplyOp,
        MeasureOp,
        DynMeasureOp,
        CircuitOp,
        ReturnOp,
    ],
    [],
)
