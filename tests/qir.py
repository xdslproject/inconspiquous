from xdsl.dialects.llvm import LLVMPointerType, LLVMVoidType
from inconspiquous.dialects.qir import QIR, QIROperation, QubitType, ResultType


def test_qir_op_names():
    for op in QIR.operations:
        assert isinstance(op, QIROperation)
        assert op.__doc__ is not None
        assert op.get_func_name() in op.__doc__


def test_qir_op_types():
    for op in QIR.operations:
        assert isinstance(op, QIROperation)
        op_def = op.get_irdl_definition()
        ty = op.get_func_type()
        assert len(ty.inputs) == len(op_def.operands)
        for i, operand in zip(ty.inputs, op_def.operands):
            if i == LLVMPointerType():
                constr = operand[1].constr
                assert constr.verifies((QubitType(),)) or constr.verifies(
                    (ResultType(),)
                )
        if ty.output == LLVMVoidType():
            assert not op_def.results
        elif ty.output == LLVMPointerType():
            constr = op_def.results[0][1].constr
            assert constr.verifies((QubitType(),)) or constr.verifies((ResultType(),))
        else:
            assert op_def.results
