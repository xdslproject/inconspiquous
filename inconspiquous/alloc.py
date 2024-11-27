from abc import ABC
from dataclasses import dataclass
from typing import Sequence
from xdsl.ir import Attribute, ParametrizedAttribute, VerifyException
from xdsl.irdl import (
    ConstraintContext,
    ConstraintVariableType,
    GenericAttrConstraint,
    RangeConstraint,
    VarExtractor,
)


class AllocAttr(ParametrizedAttribute, ABC):
    def get_types(self) -> Sequence[Attribute]: ...


@dataclass(frozen=True)
class AllocConstraint(GenericAttrConstraint[AllocAttr]):
    """
    Put a constraint on the result types of an alloc operation.
    """

    type_constraint: RangeConstraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, AllocAttr):
            raise VerifyException(
                f"attribute {attr} expected to be a allocation attribute"
            )
        self.type_constraint.verify(attr.get_types(), constraint_context)

    @dataclass(frozen=True)
    class _Extractor(VarExtractor[AllocAttr]):
        inner: VarExtractor[Sequence[Attribute]]

        def extract_var(self, a: AllocAttr) -> ConstraintVariableType:
            return self.inner.extract_var(a.get_types())

    def get_variable_extractors(self) -> dict[str, VarExtractor[AllocAttr]]:
        return {
            v: self._Extractor(r)
            for v, r in self.type_constraint.get_variable_extractors().items()
        }
