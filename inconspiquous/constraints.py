from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Set
from typing import TypeVar
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Attribute, VerifyException
from xdsl.irdl import (
    ConstraintContext,
    ConstraintVariableType,
    GenericAttrConstraint,
    InferenceContext,
    IntConstraint,
    VarExtractor,
)


class SizedAttribute(Attribute, ABC):
    @property
    @abstractmethod
    def size(self) -> int: ...


SizedAttributeCovT = TypeVar("SizedAttributeCovT", bound=SizedAttribute, covariant=True)
SizedAttributeT = TypeVar("SizedAttributeT", bound=SizedAttribute)


@dataclass(frozen=True)
class SizedAttributeConstraint(GenericAttrConstraint[SizedAttributeCovT]):
    """
    Constraints an attribute to be a gate type with size given by an integer constraint.
    """

    base_class: type[SizedAttributeCovT]
    size_constraint: IntConstraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, self.base_class):
            raise VerifyException(
                f"attribute {attr} expected to be a {self.base_class.name}"
            )
        self.size_constraint.verify(attr.size, constraint_context)

    @dataclass(frozen=True)
    class _Extractor(VarExtractor[SizedAttributeT]):
        inner: VarExtractor[int]

        def extract_var(self, a: SizedAttributeT) -> ConstraintVariableType:
            return self.inner.extract_var(a.size)

    def get_variable_extractors(self) -> dict[str, VarExtractor[SizedAttributeCovT]]:
        return {
            k: self._Extractor(v)
            for k, v in self.size_constraint.get_length_extractors().items()
        }


@dataclass(frozen=True)
class IntAttrConstraint(GenericAttrConstraint[IntAttr]):
    """
    Constrains the value of an IntAttr.
    """

    int_constraint: IntConstraint

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, IntAttr):
            raise VerifyException(f"attribute {attr} expected to be an IntAttr")
        self.int_constraint.verify(attr.data, constraint_context)

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.int_constraint.can_infer(var_constraint_names)

    def infer(self, context: InferenceContext) -> IntAttr:
        return IntAttr(self.int_constraint.infer(context))
