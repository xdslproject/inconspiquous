from abc import ABC

from xdsl.ir import (
    ParametrizedAttribute,
)


class GateAttr(ParametrizedAttribute, ABC):
    """
    In general most gate operations are not operationally different, so differentiating between them
    may actually be better done via an attribute that can be attached to a gate operation.
    """

    def num_qubits(self) -> int: ...

    # Some other possible things:
    # get_matrix
    # is_hermitian
    # is_self_adjoint
    # control wires?
    # inverse gate?


# Helper classes
class SingleQubitGate(GateAttr):
    def num_qubits(self) -> int:
        return 1


class TwoQubitGate(GateAttr):
    def num_qubits(self) -> int:
        return 2
