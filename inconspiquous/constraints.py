from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping
from typing_extensions import TypeVar
from xdsl.ir import Attribute, VerifyException
from xdsl.irdl import (
    AttrConstraint,
    ConstraintContext,
    IntConstraint,
)


class SizedAttribute(Attribute, ABC):
    @property
    @abstractmethod
    def size(self) -> int: ...


SizedAttributeCovT = TypeVar("SizedAttributeCovT", bound=SizedAttribute, covariant=True)
SizedAttributeT = TypeVar("SizedAttributeT", bound=SizedAttribute)


@dataclass(frozen=True)
class SizedAttributeConstraint(AttrConstraint[SizedAttributeCovT]):
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

    def variables(self) -> set[str]:
        return self.size_constraint.variables()

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint[SizedAttributeCovT]:
        return self
