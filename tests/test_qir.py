from xdsl.dialects.llvm import LLVMVoidType

from inconspiquous.dialects.qir import (
    QIR,
    QIROperation,
)


def test_qir_op_names():
    for op in QIR.operations:
        assert issubclass(op, QIROperation)
        assert op.__doc__ is not None
        assert op.get_func_name() in op.__doc__


def test_qir_op_types():
    for op in QIR.operations:
        assert issubclass(op, QIROperation)
        op_def = op.get_irdl_definition()
        ty = op.get_func_type()
        assert len(ty.inputs) == len(op_def.operands)
        if ty.output == LLVMVoidType():
            assert not op_def.results
        else:
            assert op_def.results
