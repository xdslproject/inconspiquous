from abc import ABC
from typing import Sequence
from xdsl.ir import Attribute, ParametrizedAttribute
from xdsl.irdl import WithRangeType


class AllocAttr(ParametrizedAttribute, WithRangeType, ABC):
    def get_types(self) -> Sequence[Attribute]: ...
