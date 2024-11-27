from collections.abc import Set
from dataclasses import dataclass
from xdsl.ir import Attribute, VerifyException
from xdsl.irdl import (
    ConstraintContext,
    GenericAttrConstraint,
    GenericRangeConstraint,
    InferenceContext,
    RangeVarConstraint,
)

from inconspiquous.dialects import qubit
from inconspiquous.dialects.gate import GateType
from inconspiquous.gates import GateAttr


@dataclass(frozen=True)
class GateConstraint(GenericAttrConstraint[GateAttr]):
    """
    Constrains a given range variable to have the correct size for the gate.
    """

    range_constraint: GenericRangeConstraint[qubit.BitType]

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, GateAttr):
            raise VerifyException(f"attribute {attr} expected to be a gate")
        self.range_constraint.verify(
            (qubit.BitType(),) * attr.num_qubits, constraint_context
        )


@dataclass(frozen=True)
class DynGateConstraint(GenericAttrConstraint[GateType]):
    """
    Constrains a given range variable to have the correct size for the dynamic gate.
    """

    range_constraint: RangeVarConstraint[qubit.BitType]

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, GateType):
            raise VerifyException(f"type {attr} expected to be a gate type")
        self.range_constraint.verify(
            (qubit.BitType(),) * attr.num_qubits.value.data, constraint_context
        )

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.range_constraint.name in var_constraint_names

    def infer(self, context: InferenceContext) -> GateType:
        range_type = self.range_constraint.infer(
            0, context
        )  # We know a range constraint does not use the input length
        return GateType(len(range_type))
