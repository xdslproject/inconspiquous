from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AbstractSet, Mapping
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
        return SizedAttributeConstraint(
            self.base_class, self.size_constraint.mapping_type_vars(type_var_mapping)
        )


@dataclass(frozen=True)
class IncrementConstraint(IntConstraint):
    """
    Increments an int constraint by 1
    """

    constraint: IntConstraint

    def verify(self, i: int, constraint_context: ConstraintContext) -> None:
        self.constraint.verify(i - 1, constraint_context)

    def variables(self) -> set[str]:
        return self.constraint.variables()

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return self.constraint.can_infer(var_constraint_names)

    def infer(self, context: ConstraintContext) -> int:
        return self.constraint.infer(context) + 1

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        return IncrementConstraint(self.constraint.mapping_type_vars(type_var_mapping))
