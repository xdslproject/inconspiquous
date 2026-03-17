from typing import ClassVar

from xdsl.dialects.builtin import i1
from xdsl.interfaces import HasCanonicalizationPatternsInterface
from xdsl.ir import Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    IntVarConstraint,
    IRDLOperation,
    RangeOf,
    SameVariadicResultSize,
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
from inconspiquous.dialects.measurement import (
    CompBasisMeasurementAttr,
    MeasurementAttr,
    MeasurementType,
)
from inconspiquous.dialects.qu import BitType


@irdl_op_definition
class GateOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    name = "qssa.gate"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = prop_def(InstrumentConstraint(_I, RangeOf(AnyAttr()).of_length(0)))

    in_qubits = var_operand_def(RangeOf(BitType()).of_length(_I))

    out_qubits = var_result_def(RangeOf(BitType()).of_length(_I))

    assembly_format = "`<` $gate `>` $in_qubits attr-dict"

    def __init__(self, gate: InstrumentAttr, *in_qubits: SSAValue | Operation):
        super().__init__(
            operands=(in_qubits,),
            properties={
                "gate": gate,
            },
            result_types=((BitType(),) * len(in_qubits),),
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qssa import GateIdentity

        return (GateIdentity(),)


@irdl_op_definition
class DynGateOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    name = "qssa.dyn_gate"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = operand_def(InstrumentType.constr(_I, RangeOf(AnyAttr()).of_length(0)))

    in_qubits = var_operand_def(RangeOf(BitType()).of_length(_I))

    out_qubits = var_result_def(RangeOf(BitType()).of_length(_I))

    assembly_format = "`<` $gate `>` $in_qubits attr-dict"

    def __init__(self, gate: SSAValue | Operation, *in_qubits: SSAValue | Operation):
        super().__init__(
            operands=(gate, in_qubits),
            result_types=((BitType(),) * len(in_qubits),),
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qssa import (
            DynGateCompose,
            DynGateConst,
        )

        return (DynGateConst(), DynGateCompose())


@irdl_op_definition
class MeasureOp(IRDLOperation):
    name = "qssa.measure"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = prop_def(
        SizedAttributeConstraint(MeasurementAttr, _I),
        default_value=CompBasisMeasurementAttr(),
    )

    in_qubits = var_operand_def(RangeOf(BitType()).of_length(_I))

    out_qubits = var_result_def(RangeOf(BitType()).of_length(_I))

    outs = var_result_def(RangeOf(i1).of_length(_I))

    assembly_format = "(`` `<` $measurement^ `>`)? $in_qubits attr-dict"

    irdl_options = (SameVariadicResultSize(),)

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
            result_types=((BitType(),) * len(in_qubits), (i1,) * len(in_qubits)),
        )


@irdl_op_definition
class DynMeasureOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    name = "qssa.dyn_measure"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = operand_def(MeasurementType.constr(_I))

    in_qubits = var_operand_def(RangeOf(BitType()).of_length(_I))

    out_qubits = var_result_def(RangeOf(BitType()).of_length(_I))

    outs = var_result_def(RangeOf(i1).of_length(_I))

    assembly_format = "`<` $measurement `>` $in_qubits attr-dict"

    irdl_options = (SameVariadicResultSize(),)

    def __init__(
        self,
        *in_qubits: SSAValue | Operation,
        measurement: SSAValue | Operation,
    ):
        super().__init__(
            operands=[measurement, in_qubits],
            result_types=((BitType(),) * len(in_qubits), (i1,) * len(in_qubits)),
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qssa import (
            DynMeasureConst,
        )

        return (DynMeasureConst(),)


@irdl_op_definition
class CircuitOp(IRDLOperation):
    name = "qssa.circuit"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    body = region_def("single_block", entry_args=RangeOf(BitType()).of_length(_I))
    result = result_def(InstrumentType.constr(_I, RangeOf(AnyAttr()).of_length(0)))

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
