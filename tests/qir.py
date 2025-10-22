from inconspiquous.dialects.qir import QIR, QIROperation


def test_qir_op_names():
    for op in QIR.operations:
        assert isinstance(op, QIROperation)
        assert op.__doc__ is not None
        assert op.get_func_name() in op.__doc__
