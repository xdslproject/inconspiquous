from typing import ClassVar

from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyInt,
    IRDLOperation,
    IntVarConstraint,
    ParameterDef,
    RangeOf,
    eq,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
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
class QReg(ParametrizedAttribute, TypeAttribute):
    """
    Type for a register of qubits of a static size.
    """

    name = "qu.reg"
    size: ParameterDef[IntegerAttr]


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
class FromBitsOp(IRDLOperation):
    name = "qu.from_bits"
    qubits = var_operand_def(BitType)
    reg = result_def(QReg)


@irdl_op_definition
class ToBitsOp(IRDLOperation):
    name = "qu.to_bits"
    reg = operand_def(QReg)
    qubits = var_result_def(BitType)


@irdl_op_definition
class CombineOp(IRDLOperation):
    name = "qu.combine"
    reg1 = operand_def(QReg)
    reg2 = operand_def(QReg)
    res = result_def(QReg)


@irdl_op_definition
class SplitOp(IRDLOperation):
    name = "qu.split"
    reg = operand_def(QReg)
    split_index = prop_def(IntegerAttr)
    res1 = result_def(QReg)
    res2 = result_def(QReg)


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
        FromBitsOp,
        ToBitsOp,
        CombineOp,
        SplitOp,
    ],
    [
        BitType,
        AllocZeroAttr,
        AllocPlusAttr,
        QReg,
    ],
)
