from __future__ import annotations

from abc import ABC, abstractmethod
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
    RangeVarConstraint,
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

from inconspiquous.dialects.instrument import (
    InstrumentAttr,
    InstrumentConstraint,
    InstrumentType,
)
from inconspiquous.dialects.measurement import (
    CompBasisMeasurementAttr,
)
from inconspiquous.dialects.qu import BitType


class QrefApplyInterface(IRDLOperation, ABC):
    """
    Operations inheriting this interface can be seen as instances of `qref.apply`
    """

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    in_qubits = var_operand_def(RangeOf(BitType()).of_length(_I))

    @abstractmethod
    def get_instrument(self) -> SSAValue[InstrumentType] | InstrumentAttr: ...

    @abstractmethod
    def get_outs(self) -> tuple[SSAValue, ...]: ...

    @staticmethod
    def create_op(
        instrument: SSAValue[InstrumentType] | InstrumentAttr,
        *in_qubits: SSAValue | Operation,
    ) -> QrefApplyInterface:
        if isinstance(instrument, InstrumentAttr):
            if not instrument.classical_results:
                return GateOp(instrument, *in_qubits)
            if len(instrument.classical_results) == instrument.num_qubits and all(
                x == i1 for x in instrument.classical_results
            ):
                return MeasureOp(*in_qubits, measurement=instrument)
            return ApplyOp(instrument, *in_qubits)
        if not instrument.type.classical_results:
            return DynGateOp(instrument, *in_qubits)
        if len(
            instrument.type.classical_results
        ) == instrument.type.num_qubits.data and all(
            x == i1 for x in instrument.type.classical_results
        ):
            return DynMeasureOp(*in_qubits, measurement=instrument)
        return DynApplyOp(instrument, *in_qubits)


@irdl_op_definition
class ApplyOp(QrefApplyInterface):
    name = "qref.apply"

    _T: ClassVar = RangeVarConstraint("T", RangeOf(AnyAttr()))

    instrument = prop_def(InstrumentConstraint(QrefApplyInterface._I, _T))

    def get_instrument(self) -> SSAValue[InstrumentType] | InstrumentAttr:
        return self.instrument

    outs = var_result_def(_T)

    def get_outs(self) -> tuple[SSAValue, ...]:
        return self.outs

    assembly_format = "`<` $instrument `>` $in_qubits (`:` type($outs)^)? attr-dict"

    def __init__(self, instrument: InstrumentAttr, *in_qubits: SSAValue | Operation):
        super().__init__(
            operands=(in_qubits,),
            properties={
                "instrument": instrument,
            },
            result_types=(instrument.classical_results,),
        )


@irdl_op_definition
class DynApplyOp(QrefApplyInterface):
    name = "qref.dyn_apply"

    _T: ClassVar = RangeVarConstraint("T", RangeOf(AnyAttr()))

    instrument = operand_def(InstrumentType.constr(QrefApplyInterface._I, _T))

    def get_instrument(self) -> SSAValue[InstrumentType] | InstrumentAttr:
        return self.instrument  # pyright: ignore[reportReturnType]

    outs = var_result_def(_T)

    def get_outs(self) -> tuple[SSAValue, ...]:
        return self.outs

    assembly_format = "`<` $instrument `>` $in_qubits (`:` type($outs)^)? attr-dict"

    def __init__(
        self, instrument: SSAValue[InstrumentType], *in_qubits: SSAValue | Operation
    ):
        super().__init__(
            operands=(
                instrument,
                in_qubits,
            ),
            result_types=(instrument.type.classical_results,),
        )


@irdl_op_definition
class GateOp(QrefApplyInterface, HasCanonicalizationPatternsInterface):
    name = "qref.gate"

    gate = prop_def(
        InstrumentConstraint(QrefApplyInterface._I, RangeOf(AnyAttr()).of_length(0))
    )

    def get_instrument(self) -> SSAValue[InstrumentType] | InstrumentAttr:
        return self.gate

    def get_outs(self) -> tuple[SSAValue, ...]:
        return ()

    assembly_format = "`<` $gate `>` $in_qubits attr-dict"

    def __init__(self, gate: InstrumentAttr, *in_qubits: SSAValue | Operation):
        super().__init__(
            operands=(in_qubits,),
            properties={
                "gate": gate,
            },
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qref import GateIdentity

        return (GateIdentity(),)


@irdl_op_definition
class DynGateOp(QrefApplyInterface, HasCanonicalizationPatternsInterface):
    name = "qref.dyn_gate"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    gate = operand_def(
        InstrumentType.constr(QrefApplyInterface._I, RangeOf(AnyAttr()).of_length(0))
    )

    def get_instrument(self) -> SSAValue[InstrumentType] | InstrumentAttr:
        return self.gate  # pyright: ignore[reportReturnType]

    def get_outs(self) -> tuple[SSAValue, ...]:
        return ()

    assembly_format = "`<` $gate `>` $in_qubits attr-dict"

    def __init__(self, gate: SSAValue | Operation, *in_qubits: SSAValue | Operation):
        super().__init__(
            operands=(gate, in_qubits),
        )

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.qref import (
            DynGateCompose,
            DynGateConst,
        )

        return (DynGateConst(), DynGateCompose())


@irdl_op_definition
class MeasureOp(QrefApplyInterface):
    name = "qref.measure"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    measurement = prop_def(
        InstrumentConstraint(
            QrefApplyInterface._I, RangeOf(i1).of_length(QrefApplyInterface._I)
        ),
        default_value=CompBasisMeasurementAttr(),
    )

    def get_instrument(self) -> SSAValue[InstrumentType] | InstrumentAttr:
        return self.measurement

    outs = var_result_def(RangeOf(i1).of_length(QrefApplyInterface._I))

    def get_outs(self) -> tuple[SSAValue, ...]:
        return self.outs

    assembly_format = "(`` `<` $measurement^ `>`)? $in_qubits attr-dict"

    def __init__(
        self,
        *in_qubits: SSAValue | Operation,
        measurement: InstrumentAttr = CompBasisMeasurementAttr(),
    ):
        super().__init__(
            properties={
                "measurement": measurement,
            },
            operands=(in_qubits,),
            result_types=((i1,) * len(in_qubits),),
        )


@irdl_op_definition
class DynMeasureOp(QrefApplyInterface, HasCanonicalizationPatternsInterface):
    name = "qref.dyn_measure"

    measurement = operand_def(
        InstrumentType.constr(
            QrefApplyInterface._I, RangeOf(i1).of_length(QrefApplyInterface._I)
        )
    )

    def get_instrument(self) -> SSAValue[InstrumentType] | InstrumentAttr:
        return self.measurement  # pyright: ignore[reportReturnType]

    outs = var_result_def(RangeOf(i1).of_length(QrefApplyInterface._I))

    def get_outs(self) -> tuple[SSAValue, ...]:
        return self.outs

    assembly_format = "`<` $measurement `>` $in_qubits attr-dict"

    def __init__(
        self, *in_qubits: SSAValue | Operation, measurement: SSAValue | Operation
    ):
        super().__init__(
            operands=(measurement, in_qubits),
            result_types=((i1,) * len(in_qubits),),
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
        GateOp,
        DynGateOp,
        MeasureOp,
        DynMeasureOp,
        CircuitOp,
        ReturnOp,
    ],
    [],
)
