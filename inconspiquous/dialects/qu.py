from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import TypeVar
from xdsl.dialects.builtin import IntAttr, IntAttrConstraint
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyInt,
    AttrConstraint,
    ConstraintContext,
    IntConstraint,
    IntVarConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    RangeOf,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class BitType(ParametrizedAttribute, TypeAttribute):
    """
    Type for a single qubit
    """

    name = "qu.bit"


class AllocAttr(ParametrizedAttribute, ABC):
    @property
    @abstractmethod
    def num_qubits(self) -> int: ...


@dataclass(frozen=True)
class AllocConstraint(AttrConstraint[AllocAttr]):
    """
    Constraints an attribute to be a gate type with size given by an integer constraint.
    """

    size_constraint: IntConstraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, AllocAttr):
            raise VerifyException(
                f"attribute {attr} expected to be a allocation attribute"
            )
        self.size_constraint.verify(attr.num_qubits, constraint_context)

    def variables(self) -> set[str]:
        return self.size_constraint.variables()

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint[AllocAttr]:
        return AllocConstraint(self.size_constraint.mapping_type_vars(type_var_mapping))


@irdl_attr_definition
class AllocZeroAttr(AllocAttr):
    """
    Allocate a qubit in the zero computational basis state
    """

    name = "qu.zero"

    @property
    def num_qubits(self) -> int:
        return 1


@irdl_attr_definition
class RegisterType(ParametrizedAttribute, TypeAttribute):
    """Type for a register of qubits of a static size."""

    name = "qu.reg"

    size: IntAttr

    @classmethod
    def constr(cls, size: IntVarConstraint):
        return ParamAttrConstraint(cls, (IntAttrConstraint(size),))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.size.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr]:
        with parser.in_angle_brackets():
            return (IntAttr(parser.parse_integer()),)


@irdl_attr_definition
class AllocPlusAttr(AllocAttr):
    """
    Allocate a qubit in the plus state.
    """

    name = "qu.plus"

    @property
    def num_qubits(self) -> int:
        return 1


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "qu.alloc"

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    alloc = prop_def(AllocConstraint(_I), default_value=AllocZeroAttr())

    outs = var_result_def(RangeOf(BitType()).of_length(_I))

    assembly_format = "(`` `<` $alloc^ `>`)? attr-dict"

    def __init__(self, alloc: AllocAttr = AllocZeroAttr()):
        super().__init__(
            properties={
                "alloc": alloc,
            },
            result_types=((BitType(),) * alloc.num_qubits,),
        )


@irdl_op_definition
class ReleaseOp(IRDLOperation):
    name = "qu.release"

    in_qubit = operand_def(BitType)

    assembly_format = "$in_qubit attr-dict"

    def __init__(self, in_qubit: Operation | SSAValue):
        super().__init__(operands=(in_qubit,))


@irdl_op_definition
class FromBitsOp(IRDLOperation):
    """Converts a collection of input qubits to a register."""

    name = "qu.from_bits"
    _I: ClassVar = IntVarConstraint("I", AnyInt())
    qubits = var_operand_def(RangeOf(BitType()).of_length(_I))
    reg = result_def(RegisterType.constr(_I))

    assembly_format = "$qubits attr-dict `:` type($reg)"

    def __init__(self, *qubits: SSAValue):
        super().__init__(
            operands=[qubits], result_types=[RegisterType(IntAttr(len(qubits)))]
        )


@irdl_op_definition
class ToBitsOp(IRDLOperation):
    """Converts a register to individual qubit typed SSA values."""

    name = "qu.to_bits"
    _I: ClassVar = IntVarConstraint("I", AnyInt())
    reg = operand_def(RegisterType.constr(_I))
    qubits = var_result_def(RangeOf(BitType()).of_length(_I))

    assembly_format = "$reg attr-dict `:` type($reg)"

    def __init__(self, reg: SSAValue[RegisterType]):
        super().__init__(
            operands=[reg], result_types=[[BitType()] * reg.type.size.data]
        )


@irdl_op_definition
class CombineOp(IRDLOperation):
    """Concatenates two registers."""

    name = "qu.combine"
    reg1 = operand_def(RegisterType)
    reg2 = operand_def(RegisterType)
    res = result_def(RegisterType)

    assembly_format = (
        "$reg1 `,` $reg2 attr-dict `:` type($reg1) `,` type($reg2) `->` type($res)"
    )

    def __init__(self, reg1: SSAValue[RegisterType], reg2: SSAValue[RegisterType]):
        res_size = reg1.type.size.data + reg2.type.size.data
        super().__init__(
            operands=[reg1, reg2], result_types=[RegisterType(IntAttr(res_size))]
        )

    def verify_(self):
        assert isinstance(self.reg1.type, RegisterType)
        assert isinstance(self.reg2.type, RegisterType)

        if (
            self.reg1.type.size.data + self.reg2.type.size.data
            != self.res.type.size.data
        ):
            raise VerifyException(
                "Result register size must equal the sum of input register sizes."
            )


@irdl_op_definition
class SplitOp(IRDLOperation):
    """Splits a register into two parts."""

    name = "qu.split"
    reg = operand_def(RegisterType)
    res1 = result_def(RegisterType)
    res2 = result_def(RegisterType)

    assembly_format = "$reg attr-dict `:` type($reg) `->` type($res1) `,` type($res2)"

    def __init__(
        self,
        reg: SSAValue[RegisterType],
        res1_type: RegisterType,
        res2_type: RegisterType,
    ):
        super().__init__(operands=[reg], result_types=[res1_type, res2_type])

    def verify_(self):
        assert isinstance(self.reg.type, RegisterType)

        if (
            self.reg.type.size.data
            != self.res1.type.size.data + self.res2.type.size.data
        ):
            raise VerifyException(
                "Input register size must equal the sum of result register sizes."
            )


Qu = Dialect(
    "qu",
    [
        AllocOp,
        ReleaseOp,
        FromBitsOp,
        ToBitsOp,
        CombineOp,
        SplitOp,
    ],
    [
        BitType,
        RegisterType,
        AllocZeroAttr,
        AllocPlusAttr,
    ],
)
