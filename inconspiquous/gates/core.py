from abc import ABC

from xdsl.ir import (
    Attribute,
    ParametrizedAttribute,
)
from xdsl.irdl import (
    WithType,
)


class GateAttr(ParametrizedAttribute, WithType, ABC):
    """
    In general most gate operations are not operationally different, so differentiating between them
    may actually be better done via an attribute that can be attached to a gate operation.
    """

    @property
    def num_qubits(self) -> int: ...

    def get_type(self) -> Attribute:
        # avoid import loop
        from inconspiquous.dialects.gate import GateType

        return GateType(self.num_qubits)

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
