from typing import ClassVar
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyInt,
    IRDLOperation,
    IntVarConstraint,
    RangeOf,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    var_result_def,
    eq,
)

from inconspiquous.alloc import AllocAttr
from inconspiquous.constraints import SizedAttributeConstraint


@irdl_attr_definition
class BitType(ParametrizedAttribute, TypeAttribute):
    """
    Type for a single qubit
    """

    name = "qu.bit"


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

    alloc = prop_def(
        SizedAttributeConstraint(AllocAttr, _I), default_value=AllocZeroAttr()
    )

    outs = var_result_def(RangeOf(eq(BitType()), length=_I))

    assembly_format = "(`` `<` $alloc^ `>`)? attr-dict"

    def __init__(self, alloc: AllocAttr = AllocZeroAttr()):
        super().__init__(
            properties={
                "alloc": alloc,
            },
            result_types=((BitType(),) * alloc.num_qubits,),
        )


Qu = Dialect(
    "qu",
    [
        AllocOp,
    ],
    [BitType, AllocZeroAttr, AllocPlusAttr],
)
