from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
)

from inconspiquous.alloc import AllocAttr


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


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "qubit.alloc"

    alloc = prop_def(AllocAttr, default_value=AllocZeroAttr())

    outs = result_def(BitType())

    assembly_format = "(`` `<` $alloc^ `>`)? attr-dict"

    def __init__(self, alloc: AllocAttr = AllocZeroAttr()):
        super().__init__(
            properties={
                "alloc": alloc,
            },
            result_types=[BitType()],
        )


Qubit = Dialect(
    "qubit",
    [
        AllocOp,
    ],
    [BitType, AllocZeroAttr],
)
