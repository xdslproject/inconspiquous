from abc import ABC, abstractmethod
from xdsl.ir import ParametrizedAttribute

from inconspiquous.constraints import SizedAttribute


class MeasurementAttr(ParametrizedAttribute, SizedAttribute, ABC):
    @property
    @abstractmethod
    def num_qubits(self) -> int: ...

    @property
    def size(self) -> int:
        return self.num_qubits
