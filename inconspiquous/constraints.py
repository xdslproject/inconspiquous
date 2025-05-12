from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar
from xdsl.ir import Attribute, VerifyException
from xdsl.irdl import (
    ConstraintContext,
    ConstraintVariableType,
    GenericAttrConstraint,
    IntConstraint,
    VarExtractor,
)
from xdsl.utils.exceptions import PyRDLError


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
    class _Extractor(VarExtractor[Attribute]):
        inner: VarExtractor[int]

        def extract_var(self, a: Attribute) -> ConstraintVariableType:
            if not isinstance(a, SizedAttribute):
                raise PyRDLError(f"Inference expected {a} to be a SizedAttribute")
            return self.inner.extract_var(a.size)

    def get_variable_extractors(self) -> dict[str, VarExtractor[Attribute]]:
        return {
            k: self._Extractor(v)
            for k, v in self.size_constraint.get_length_extractors().items()
        }
