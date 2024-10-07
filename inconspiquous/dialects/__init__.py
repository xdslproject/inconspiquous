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

    def get_linalg():
        from xdsl.dialects.linalg import Linalg

        return Linalg

    def get_qssa():
        from inconspiquous.dialects.qssa import Qssa

        return Qssa

    def get_quantum():
        from inconspiquous.dialects.quantum import Quantum

        return Quantum

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

    return {
        "arith": get_arith,
        "builtin": get_builtin,
        "cf": get_cf,
        "func": get_func,
        "linalg": get_linalg,
        "qssa": get_qssa,
        "quantum": get_quantum,
        "scf": get_scf,
        "stim": get_stim,
        "tensor": get_tensor,
        "test": get_test,
    }
