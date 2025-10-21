from abc import abstractmethod
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.dialects.builtin import Float64Type, FloatAttr, i1


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    """
    QIR qubit type. Lowers to opaque pointer.
    """

    name = "qir.qubit"


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    """
    QIR result type. Lowers to opaque pointer.
    """

    name = "qir.result"


class QIROperation(IRDLOperation):
    """
    QIR operations are defined using opaque function.
    """

    @staticmethod
    @abstractmethod
    def get_func_name() -> str:
        """
        Return the name of the opaque function.
        """
        ...


@irdl_op_definition
class ResultGetOneOp(QIROperation):
    """
    MLIR equivalent of __quantum__rt__result_get_one
    """

    name = "qir.result_get_one"

    out = result_def(ResultType)

    assembly_format = "attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__rt__result_get_one"


@irdl_op_definition
class ResultEqualOp(QIROperation):
    """
    MLIR equivalent of __quantum__rt__result_equal
    """

    name = "qir.result_equal"

    lhs = operand_def(ResultType)
    rhs = operand_def(ResultType)

    out = result_def(i1)

    assembly_format = "$lhs `,` $rhs attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__rt__result_equal"


@irdl_op_definition
class QubitAllocateOp(QIROperation):
    """
    MLIR equivalent of __quantum__rt__qubit_allocate
    """

    name = "qir.qubit_allocate"

    out = result_def(QubitType)

    assembly_format = "attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__rt__qubit_allocate"


@irdl_op_definition
class MeasureOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__m__body
    """

    name = "qir.m"

    qubit = operand_def(QubitType)
    res = result_def(ResultType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__m__body"


@irdl_op_definition
class ReleaseOp(QIROperation):
    """
    MLIR equivalent of __quantum__rt__qubit_release
    """

    name = "qir.qubit_release"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__rt__qubit_release"


@irdl_op_definition
class CNotOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__cnot__body
    """

    name = "qir.cnot"

    control = operand_def(QubitType)
    target = operand_def(QubitType)

    assembly_format = "$control `,` $target attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__cnot__body"


@irdl_op_definition
class CZOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__cz__body
    """

    name = "qir.cz"

    control = operand_def(QubitType)
    target = operand_def(QubitType)

    assembly_format = "$control `,` $target attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__cz__body"


@irdl_op_definition
class HOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__h__body
    """

    name = "qir.h"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__h__body"


@irdl_op_definition
class RXOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__rx__body
    """

    name = "qir.rx"

    angle = prop_def(FloatAttr[Float64Type])
    qubit = operand_def(QubitType)

    assembly_format = "`` `<` $angle `>` $qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__rx__body"


@irdl_op_definition
class RYOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__ry__body
    """

    name = "qir.ry"

    angle = prop_def(FloatAttr[Float64Type])
    qubit = operand_def(QubitType)

    assembly_format = "`` `<` $angle `>` $qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__ry__body"


@irdl_op_definition
class RZOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__rz__body
    """

    name = "qir.rz"

    angle = prop_def(FloatAttr[Float64Type])
    qubit = operand_def(QubitType)

    assembly_format = "`` `<` $angle `>` $qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__rz__body"


@irdl_op_definition
class SOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__s__body
    """

    name = "qir.s"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__s__body"


@irdl_op_definition
class SAdjOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__s_adj
    """

    name = "qir.s_adj"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__s__adj"


@irdl_op_definition
class TOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__t__body
    """

    name = "qir.t"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__t__body"


@irdl_op_definition
class TAdjOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__t__adj
    """

    name = "qir.t_adj"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__t__adj"


@irdl_op_definition
class XOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__x__body
    """

    name = "qir.x"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__x__body"


@irdl_op_definition
class YOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__y__body
    """

    name = "qir.y"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__y__body"


@irdl_op_definition
class ZOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__z__body
    """

    name = "qir.z"

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__z__body"


QIR = Dialect(
    "qir",
    [
        ResultGetOneOp,
        ResultEqualOp,
        QubitAllocateOp,
        MeasureOp,
        ReleaseOp,
        CNotOp,
        CZOp,
        HOp,
        RXOp,
        RYOp,
        RZOp,
        SOp,
        SAdjOp,
        TOp,
        TAdjOp,
        XOp,
        YOp,
        ZOp,
    ],
    [
        QubitType,
        ResultType,
    ],
)
