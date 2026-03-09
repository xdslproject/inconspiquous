from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar

from typing_extensions import TypeVar
from xdsl.dialects.builtin import ArrayAttr, IntAttr
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
    VerifyException,
    dataclass,
)
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    IntConstraint,
    IntVarConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    RangeConstraint,
    RangeOf,
    RangeVarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import ConstantLike, Pure


class InstrumentAttr(ParametrizedAttribute, ABC):
    """
    In general most quantum operations are not operationally different,
    so we specify them by an attribute that is passed into a more generic
    operation.

    Gates are instruments with no classical results.
    """

    @property
    @abstractmethod
    def num_qubits(self) -> int: ...

    @property
    @abstractmethod
    def classical_results(self) -> tuple[TypeAttribute, ...]: ...


@dataclass(frozen=True)
class InstrumentConstraint(AttrConstraint[InstrumentAttr]):
    """
    Constraints an instrument attribute to have a given number of qubits and given classical results.
    """

    size_constraint: IntConstraint
    result_constraint: RangeConstraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, InstrumentAttr):
            raise VerifyException(
                f"attribute {attr} expected to be a instrument attribute"
            )
        self.size_constraint.verify(attr.num_qubits, constraint_context)
        self.result_constraint.verify(attr.classical_results, constraint_context)

    def variables(self) -> set[str]:
        return self.size_constraint.variables() | self.result_constraint.variables()

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint[InstrumentAttr]:
        return InstrumentConstraint(
            self.size_constraint.mapping_type_vars(type_var_mapping),
            self.result_constraint.mapping_type_vars(type_var_mapping),
        )


@irdl_attr_definition
class InstrumentType(ParametrizedAttribute, TypeAttribute):
    """
    A type for dynamic quantum instruments.
    """

    name = "instrument.type"

    num_qubits: IntAttr

    result_types: ArrayAttr[TypeAttribute]

    def __init__(
        self,
        num_qubits: int | IntAttr,
        result_types: ArrayAttr[TypeAttribute] | Sequence[TypeAttribute] = (),
    ):
        if isinstance(num_qubits, int):
            num_qubits = IntAttr(num_qubits)
        if not isinstance(result_types, ArrayAttr):
            result_types = ArrayAttr(result_types)
        super().__init__(num_qubits, result_types)

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> tuple[IntAttr, ArrayAttr[TypeAttribute]]:
        with parser.in_angle_brackets():
            i = parser.parse_integer(allow_boolean=False, allow_negative=False)
            if parser.parse_optional_punctuation(","):
                result_types = ArrayAttr(
                    parser.parse_comma_separated_list(
                        parser.Delimiter.NONE, parser.parse_type
                    )
                )
            else:
                result_types = ArrayAttr(())
            return (IntAttr(i), result_types)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.num_qubits.data)
            for ty in self.result_types:
                printer.print_string(", ")
                printer.print_attribute(ty)

    @staticmethod
    def constr(
        int_constraint: IntConstraint | None = None,
        result_constraint: RangeConstraint | None = None,
    ) -> AttrConstraint[InstrumentType]:
        if int_constraint is None and result_constraint is None:
            return BaseAttr(InstrumentType)
        return ParamAttrConstraint(
            InstrumentType,
            (IntAttr.constr(int_constraint), ArrayAttr.constr(result_constraint)),
        )


@irdl_op_definition
class ConstantInstrumentOp(IRDLOperation, HasFolderInterface):
    """
    Constant-like operation for producing measurement types from measurement attributes.
    """

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    T: ClassVar = RangeVarConstraint("T", RangeOf(AnyAttr()))

    name = "instrument.constant"

    instrument = prop_def(InstrumentConstraint(_I, T))

    out = result_def(InstrumentType.constr(_I, T))

    assembly_format = "$instrument attr-dict"

    traits = traits_def(Pure(), ConstantLike())

    def __init__(self, instrument: InstrumentAttr):
        super().__init__(
            properties={
                "instrument": instrument,
            },
            result_types=(
                InstrumentType(instrument.num_qubits, instrument.classical_results),
            ),
        )

    def fold(self) -> tuple[InstrumentAttr]:
        return (self.instrument,)


Instrument = Dialect(
    "instrument",
    [
        ConstantInstrumentOp,
    ],
    [
        InstrumentType,
    ],
)
