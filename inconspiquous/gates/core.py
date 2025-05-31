from abc import ABC, abstractmethod
from typing import Literal

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


class CliffordGateAttr(GateAttr, ABC):
    """
    Base class for Clifford gates that support Pauli propagation.
    """

    @abstractmethod
    def pauli_prop(
        self, input_idx: int, pauli_type: Literal["X", "Z"]
    ) -> tuple[tuple[bool, bool], ...]:
        """
        Compute Pauli propagation through this gate.

        Args:
            input_idx: The index of the input qubit where the Pauli gate is applied
            pauli_type: Either "X" or "Z" indicating the type of Pauli gate

        Returns:
            A tuple of (X, Z) pairs for each output qubit, where True indicates
            that the corresponding Pauli component should be applied to that output.

        For example, for Hadamard gate:
            - X propagates to Z: pauli_prop(0, "X") returns ((False, True),)
            - Z propagates to X: pauli_prop(0, "Z") returns ((True, False),)
        """
        pass
