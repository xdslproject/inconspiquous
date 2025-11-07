from abc import ABC, abstractmethod
from xdsl.dialects import llvm
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.dialects.builtin import Float64Type, i1


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


class QIROperation(IRDLOperation, ABC):
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

    @classmethod
    def get_func_type(cls) -> llvm.LLVMFunctionType:
        """
        Return the llvm function type for this function.
        If this function has inputs/outputs which are not llvm pointers,
        then this function must be overriden.
        """
        irdl_def = cls.get_irdl_definition()
        operands = len(irdl_def.operands)
        results = len(irdl_def.results)

        return llvm.LLVMFunctionType(
            (llvm.LLVMPointerType(),) * operands,
            llvm.LLVMPointerType() if results else None,
        )


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

    def __init__(self):
        super().__init__(result_types=(ResultType(),))


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

    @classmethod
    def get_func_type(cls) -> llvm.LLVMFunctionType:
        return llvm.LLVMFunctionType(
            (llvm.LLVMPointerType(), llvm.LLVMPointerType()), i1
        )

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        super().__init__(operands=(lhs, rhs), result_types=(i1,))


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

    def __init__(self):
        super().__init__(result_types=(QubitType(),))


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

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,), result_types=(ResultType(),))


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

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


class ControlledOperation(QIROperation, ABC):
    """
    Base class for operations with control and target qubits.
    """

    control = operand_def(QubitType)
    target = operand_def(QubitType)

    assembly_format = "$control `,` $target attr-dict"

    def __init__(self, control: SSAValue | Operation, target: SSAValue | Operation):
        super().__init__(operands=(control, target))


@irdl_op_definition
class CNotOp(ControlledOperation):
    """
    MLIR equivalent of __quantum__qis__cnot__body
    """

    name = "qir.cnot"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__cnot__body"


@irdl_op_definition
class CZOp(ControlledOperation):
    """
    MLIR equivalent of __quantum__qis__cz__body
    """

    name = "qir.cz"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__cz__body"


class RotationOperation(QIROperation, ABC):
    """
    Base class for rotation gates
    """

    angle = operand_def(Float64Type)
    qubit = operand_def(QubitType)

    assembly_format = "`` `<` $angle `>` $qubit attr-dict"

    def __init__(self, angle: SSAValue | Operation, qubit: SSAValue | Operation):
        super().__init__(operands=(angle, qubit))

    @classmethod
    def get_func_type(cls) -> llvm.LLVMFunctionType:
        return llvm.LLVMFunctionType((Float64Type(), llvm.LLVMPointerType()))


@irdl_op_definition
class RXOp(RotationOperation):
    """
    MLIR equivalent of __quantum__qis__rx__body
    """

    name = "qir.rx"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__rx__body"


@irdl_op_definition
class RYOp(RotationOperation):
    """
    MLIR equivalent of __quantum__qis__ry__body
    """

    name = "qir.ry"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__ry__body"


@irdl_op_definition
class RZOp(RotationOperation):
    """
    MLIR equivalent of __quantum__qis__rz__body
    """

    name = "qir.rz"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__rz__body"


@irdl_op_definition
class RZZOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__rzz__body
    """

    name = "qir.rzz"

    angle = operand_def(Float64Type)
    qubit1 = operand_def(QubitType)
    qubit2 = operand_def(QubitType)

    assembly_format = "`` `<` $angle `>` $qubit1 `,` $qubit2 attr-dict"

    def __init__(
        self,
        angle: SSAValue | Operation,
        qubit1: SSAValue | Operation,
        qubit2: SSAValue | Operation,
    ):
        super().__init__(operands=(angle, qubit1, qubit2))

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__rzz__body"

    @classmethod
    def get_func_type(cls) -> llvm.LLVMFunctionType:
        return llvm.LLVMFunctionType(
            (Float64Type(), llvm.LLVMPointerType(), llvm.LLVMPointerType())
        )


class ControlledRotationOperation(QIROperation, ABC):
    """
    Base class for controlled rotation gates
    """

    angle = operand_def(Float64Type)
    control = operand_def(QubitType)
    target = operand_def(QubitType)

    assembly_format = "`` `<` $angle `>` $control `,` $target attr-dict"

    def __init__(
        self,
        angle: SSAValue | Operation,
        control: SSAValue | Operation,
        target: SSAValue | Operation,
    ):
        super().__init__(operands=(angle, control, target))

    @classmethod
    def get_func_type(cls) -> llvm.LLVMFunctionType:
        return llvm.LLVMFunctionType(
            (Float64Type(), llvm.LLVMPointerType(), llvm.LLVMPointerType())
        )


@irdl_op_definition
class CRXOp(ControlledRotationOperation):
    """
    MLIR equivalent of __quantum__qis__rx__ctl
    """

    name = "qir.crx"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__rx__ctl"


@irdl_op_definition
class CRYOp(ControlledRotationOperation):
    """
    MLIR equivalent of __quantum__qis__ry__ctl
    """

    name = "qir.cry"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__ry__ctl"


@irdl_op_definition
class CRZOp(ControlledRotationOperation):
    """
    MLIR equivalent of __quantum__qis__rz__ctl
    """

    name = "qir.crz"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__rz__ctl"


class SingleQubitOperation(QIROperation, ABC):
    """
    Base class for operations on a single qubit and no other operands.
    """

    qubit = operand_def(QubitType)

    assembly_format = "$qubit attr-dict"

    def __init__(self, qubit: SSAValue | Operation):
        super().__init__(operands=(qubit,))


@irdl_op_definition
class HOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__h__body
    """

    name = "qir.h"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__h__body"


@irdl_op_definition
class SOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__s__body
    """

    name = "qir.s"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__s__body"


@irdl_op_definition
class SAdjOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__s_adj
    """

    name = "qir.s_adj"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__s__adj"


@irdl_op_definition
class TOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__t__body
    """

    name = "qir.t"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__t__body"


@irdl_op_definition
class TAdjOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__t__adj
    """

    name = "qir.t_adj"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__t__adj"


@irdl_op_definition
class XOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__x__body
    """

    name = "qir.x"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__x__body"


@irdl_op_definition
class YOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__y__body
    """

    name = "qir.y"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__y__body"


@irdl_op_definition
class ZOp(SingleQubitOperation):
    """
    MLIR equivalent of __quantum__qis__z__body
    """

    name = "qir.z"

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__z__body"


@irdl_op_definition
class CCXOp(QIROperation):
    """
    MLIR equivalent of __quantum__qis__ccx__body
    """

    name = "qir.ccx"

    c1 = operand_def(QubitType)
    c2 = operand_def(QubitType)
    targ = operand_def(QubitType)

    assembly_format = "$c1 `,` $c2 `,` $targ attr-dict"

    def __init__(
        self,
        c1: SSAValue | Operation,
        c2: SSAValue | Operation,
        targ: SSAValue | Operation,
    ):
        super().__init__(operands=(c1, c2, targ))

    @staticmethod
    def get_func_name() -> str:
        return "__quantum__qis__ccx__body"


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
        RZZOp,
        CRXOp,
        CRYOp,
        CRZOp,
        SOp,
        SAdjOp,
        TOp,
        TAdjOp,
        XOp,
        YOp,
        ZOp,
        CCXOp,
    ],
    [
        QubitType,
        ResultType,
    ],
)
