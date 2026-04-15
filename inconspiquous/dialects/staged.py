from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import i1
from xdsl.ir import (
    Block,
    BlockArgument,
    Dialect,
    IsTerminator,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
    VerifyException,
)
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrConstraint,
    IntVarConstraint,
    IRDLAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    RangeOf,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
    var_result_def,
)

from inconspiquous.constraints import SizedAttributeConstraint
from inconspiquous.dialects.gate import GateAttr, GateType
from inconspiquous.dialects.measurement import CompBasisMeasurementAttr, MeasurementAttr
from inconspiquous.dialects.qu import BitType


@irdl_attr_definition
class LaterType(ParametrizedAttribute, TypeAttribute):
    """
    A wrapper for types which are not yet available in this block.
    """

    name = "staged.later"

    inner: TypeAttribute

    @staticmethod
    def constr(type: IRDLAttrConstraint) -> AttrConstraint[LaterType]:
        return ParamAttrConstraint(LaterType, (type,))


class StagedOperation(IRDLOperation):
    """
    Operation with qubit inputs which must live in the same block.
    """

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    in_qubits = var_operand_def(RangeOf(BitType()).of_length(_I))

    def verify_(self) -> None:
        parent = self.parent
        if parent is None:
            raise VerifyException("Not attached to block")
        for q in self.in_qubits:
            if isinstance(q, BlockArgument):
                raise VerifyException(f"Qubit input {q} cannot be a block argument")

            input_parent = q.owner.parent

            if input_parent is None:
                raise VerifyException(f"Qubit input {q} is not attached")

            if parent != input_parent:
                raise VerifyException(
                    f"Qubit input {q} originates in block {input_parent} but is used in {parent}"
                )


@irdl_op_definition
class GateOp(StagedOperation):
    name = "staged.gate"

    gate = prop_def(SizedAttributeConstraint(GateAttr, StagedOperation._I))

    assembly_format = "`<` $gate `>` $in_qubits attr-dict"

    def __init__(self, gate: GateAttr, *in_qubits: SSAValue | Operation):
        super().__init__(
            operands=(in_qubits,),
            properties={
                "gate": gate,
            },
        )


@irdl_op_definition
class DynGateOp(StagedOperation):
    name = "staged.dyn_gate"

    gate = operand_def(GateType.constr(StagedOperation._I))

    assembly_format = "`<` $gate `>` $in_qubits attr-dict"

    def __init__(self, gate: SSAValue | Operation, *in_qubits: SSAValue | Operation):
        super().__init__(
            operands=(gate, in_qubits),
        )


@irdl_op_definition
class MeasureOp(StagedOperation):
    name = "staged.measure"

    measurement = prop_def(
        SizedAttributeConstraint(MeasurementAttr, StagedOperation._I),
        default_value=CompBasisMeasurementAttr(),
    )

    outs = var_result_def(RangeOf(LaterType(i1)).of_length(StagedOperation._I))

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
            result_types=((LaterType(i1),) * len(in_qubits),),
        )


@irdl_op_definition
class PureOp(IRDLOperation):
    """
    Make a (classical) type available later.
    """

    name = "staged.pure"

    _T: ClassVar = VarConstraint("T", AnyAttr())

    arg = operand_def(_T)

    out = result_def(LaterType.constr(_T))

    def __init__(self, arg: SSAValue[TypeAttribute]):
        super().__init__(operands=(arg,), result_types=(LaterType(arg.type),))


@irdl_op_definition
class StepOp(IRDLOperation):
    """
    Progress a time step, completing the current circuit and unwrapping later types.
    """

    name = "staged.step"

    arguments = var_operand_def(LaterType)

    successor = successor_def()

    traits = traits_def(IsTerminator())

    assembly_format = "$successor (`(` $arguments^ `:` type($arguments) `)`)? attr-dict"

    def __init__(self, successor: Block, *args: Operation | SSAValue):
        super().__init__(operands=[args], successors=[successor])

    def verify_(self) -> None:
        arg_types = self.arguments.types
        succ_arg_types = tuple(self.successor.arg_types)
        if len(arg_types) != len(succ_arg_types):
            raise VerifyException(
                f"Operation has {len(arg_types)} arguments but target block has {len(succ_arg_types)} arguments"
            )

        for x, y in zip(arg_types, succ_arg_types):
            assert isinstance(x, LaterType)
            if x.inner != y:
                raise VerifyException(
                    f"Argument {x} does not match block argument of type {y}"
                )


Staged = Dialect(
    "staged",
    [
        GateOp,
        DynGateOp,
        MeasureOp,
        PureOp,
        StepOp,
    ],
    [
        LaterType,
    ],
)
