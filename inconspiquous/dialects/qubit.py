from typing import ClassVar, Sequence
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    RangeConstraint,
    RangeOf,
    RangeVarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    var_result_def,
)

from inconspiquous.alloc import AllocAttr, AllocConstraint


@irdl_attr_definition
class BitType(ParametrizedAttribute, TypeAttribute):
    """
    Type for a single qubit
    """

    name = "qubit.bit"


@irdl_attr_definition
class AllocZeroAttr(AllocAttr):
    """
    Allocate a qubit in the zero computational basis state
    """

    name = "qubit.zero"

    def get_types(self) -> Sequence[Attribute]:
        return (BitType(),)


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "qubit.alloc"

    _T: ClassVar[RangeConstraint] = RangeVarConstraint("T", RangeOf(AnyAttr()))

    alloc = prop_def(AllocConstraint(_T), default_value=AllocZeroAttr())

    outs = var_result_def(_T)

    assembly_format = "(`` `<` $alloc^ `>`)? attr-dict"

    def __init__(self, alloc: AllocAttr = AllocZeroAttr()):
        super().__init__(
            properties={
                "alloc": alloc,
            },
            result_types=[alloc.get_types()],
        )


Qubit = Dialect(
    "qubit",
    [
        AllocOp,
    ],
    [BitType, AllocZeroAttr],
)
