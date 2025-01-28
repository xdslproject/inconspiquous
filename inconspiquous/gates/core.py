from abc import ABC, abstractmethod

from xdsl.ir import (
    ParametrizedAttribute,
)

from inconspiquous.constraints import SizedAttribute


class GateAttr(ParametrizedAttribute, SizedAttribute, ABC):
    """
    In general most gate operations are not operationally different, so differentiating between them
    may actually be better done via an attribute that can be attached to a gate operation.
    """

    @property
    @abstractmethod
    def num_qubits(self) -> int: ...

    @property
    def size(self) -> int:
        return self.num_qubits

    # Some other possible things:
    # get_matrix
    # is_hermitian
    # is_self_adjoint
    # control wires?
    # inverse gate?


# Helper classes
class SingleQubitGate(GateAttr):
    @property
    def num_qubits(self) -> int:
        return 1


class TwoQubitGate(GateAttr):
    @property
    def num_qubits(self) -> int:
        return 2
