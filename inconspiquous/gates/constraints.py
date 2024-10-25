from xdsl.ir import Attribute, VerifyException
from xdsl.irdl import ConstraintContext, GenericAttrConstraint, RangeVarConstraint, base

from inconspiquous.dialects.gate import GateType
from inconspiquous.gates import GateAttr


class GateConstraint(GenericAttrConstraint[GateAttr]):
    """
    Constrains a given range variable to have the correct size for the gate.
    """

    range_var: str
    gate_constraint: GenericAttrConstraint[GateAttr]

    def __init__(
        self,
        range_constraint: str | RangeVarConstraint[Attribute],
        gate_constraint: GenericAttrConstraint[GateAttr] = base(GateAttr),
    ):
        if isinstance(range_constraint, str):
            self.range_var = range_constraint
        else:
            self.range_var = range_constraint.name
        self.gate_constraint = gate_constraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        self.gate_constraint.verify(attr, constraint_context)
        if self.range_var in constraint_context.range_variables:
            attrs = constraint_context.get_range_variable(self.range_var)
            assert isinstance(attr, GateAttr)
            if attr.num_qubits != len(attrs):
                raise VerifyException(
                    f"Gate {attr} expected {attr.num_qubits} qubits but got {len(attrs)}"
                )


class DynGateConstraint(GenericAttrConstraint[GateType]):
    """
    Constrains a given range variable to have the correct size for the dynamic gate.
    """

    range_var: str

    def __init__(
        self,
        range_constraint: str | RangeVarConstraint[Attribute],
    ):
        if isinstance(range_constraint, str):
            self.range_var = range_constraint
        else:
            self.range_var = range_constraint.name

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        base(GateType).verify(attr, constraint_context)
        if self.range_var in constraint_context.range_variables:
            attrs = constraint_context.get_range_variable(self.range_var)
            assert isinstance(attr, GateType)
            num_qubits = attr.num_qubits.value.data
            if num_qubits != len(attrs):
                raise VerifyException(
                    f"Gate input expected {num_qubits} qubits but got {len(attrs)}"
                )

    def can_infer(self, constraint_names: set[str]) -> bool:
        return self.range_var in constraint_names

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        types = constraint_context.get_range_variable(self.range_var)
        return GateType(len(types))
