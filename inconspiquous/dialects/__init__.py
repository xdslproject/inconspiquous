from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    def get_arith():
        from xdsl.dialects.arith import Arith

        return Arith

    def get_builtin():
        from xdsl.dialects.builtin import Builtin

        return Builtin

    def get_cf():
        from xdsl.dialects.cf import Cf

        return Cf

    def get_func():
        from xdsl.dialects.func import Func

        return Func

    def get_gate():
        from inconspiquous.dialects.gate import Gate

        return Gate

    def get_linalg():
        from xdsl.dialects.linalg import Linalg

        return Linalg

    def get_prob():
        from inconspiquous.dialects.prob import Prob

        return Prob

    def get_qec():
        from inconspiquous.dialects.qec import QEC

        return QEC

    def get_qref():
        from inconspiquous.dialects.qref import Qref

        return Qref

    def get_qubit():
        from inconspiquous.dialects.qubit import Qubit

        return Qubit

    def get_qssa():
        from inconspiquous.dialects.qssa import Qssa

        return Qssa

    def get_scf():
        from xdsl.dialects.scf import Scf

        return Scf

    def get_stim():
        from xdsl.dialects.stim import Stim

        return Stim

    def get_tensor():
        from xdsl.dialects.tensor import Tensor

        return Tensor

    def get_test():
        from xdsl.dialects.test import Test

        return Test

    def get_varith():
        from xdsl.dialects.varith import Varith

        return Varith

    return {
        "arith": get_arith,
        "builtin": get_builtin,
        "cf": get_cf,
        "func": get_func,
        "gate": get_gate,
        "linalg": get_linalg,
        "prob": get_prob,
        "qec": get_qec,
        "qref": get_qref,
        "qubit": get_qubit,
        "qssa": get_qssa,
        "scf": get_scf,
        "stim": get_stim,
        "tensor": get_tensor,
        "test": get_test,
        "varith": get_varith,
    }
