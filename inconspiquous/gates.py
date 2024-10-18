from abc import ABC

from xdsl.ir import (
    Attribute,
    ParametrizedAttribute,
    VerifyException,
)
from xdsl.irdl import (
    BaseAttr,
    ConstraintContext,
    GenericAttrConstraint,
    RangeVarConstraint,
)


class GateAttr(ParametrizedAttribute, ABC):
    """
    In general most gate operations are not operationally different, so differentiating between them
    may actually be better done via an attribute that can be attached to a gate operation.
    """

    @property
    def num_qubits(self) -> int: ...

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


class GateConstraint(GenericAttrConstraint[GateAttr]):
    """
    Constrains a given range variable to have the correct size for the gate.
    """

    range_var: str
    gate_constraint: GenericAttrConstraint[GateAttr]

    def __init__(
        self,
        range_constraint: str | RangeVarConstraint[Attribute],
        gate_constraint: GenericAttrConstraint[GateAttr] = BaseAttr[GateAttr](GateAttr),
    ):
        if isinstance(range_constraint, str):
            self.range_var = range_constraint
        else:
            self.range_var = range_constraint.name
        self.gate_constraint = gate_constraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if self.range_var in constraint_context.range_variables:
            attrs = constraint_context.get_range_variable(self.range_var)
            assert isinstance(attr, GateAttr)
            if attr.num_qubits != len(attrs):
                raise VerifyException(
                    f"Gate {attr} expected {attr.num_qubits} qubits but got {len(attrs)}"
                )
