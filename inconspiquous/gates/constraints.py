from dataclasses import dataclass
from xdsl.ir import Attribute, VerifyException
from xdsl.irdl import (
    ConstraintContext,
    ConstraintVariableType,
    GenericAttrConstraint,
    IntConstraint,
    VarExtractor,
)

from inconspiquous.gates import GateAttr


@dataclass(frozen=True)
class GateAttrSizeConstraint(GenericAttrConstraint[GateAttr]):
    """
    Constraints an attribute to be a gate type with size given by an integer constraint.
    """

    int_constraint: IntConstraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, GateAttr):
            raise VerifyException(f"attribute {attr} expected to be a gate")
        self.int_constraint.verify(attr.num_qubits, constraint_context)

    @dataclass(frozen=True)
    class _Extractor(VarExtractor[GateAttr]):
        inner: VarExtractor[int]

        def extract_var(self, a: GateAttr) -> ConstraintVariableType:
            return self.inner.extract_var(a.num_qubits)

    def get_variable_extractors(self) -> dict[str, VarExtractor[GateAttr]]:
        return {
            k: self._Extractor(v)
            for k, v in self.int_constraint.get_length_extractors().items()
        }
