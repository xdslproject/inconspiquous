from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Dialect
from xdsl.irdl import irdl_attr_definition, ParameterDef
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from inconspiquous.gates import GateAttr


@irdl_attr_definition
class QFTAttr(GateAttr):
    name = "qft.n"

    num_qubits: ParameterDef[IntAttr]

    @property
    def num_qubits(self) -> int:
        return self.num_qubits.data

    def __init__(self, num_qubits: int | IntAttr):
        if not isinstance(num_qubits, IntAttr):
            num_qubits = IntAttr(num_qubits)

        super().__init__((num_qubits,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr]:
        with parser.in_angle_brackets():
            num_qubits = parser.parse_integer(allow_negative=False, allow_boolean=False)
        return (IntAttr.new(num_qubits),)

    def print_parameters(self, printer: Printer) -> None:
        return self.num_qubits.print_parameter(printer)


QFT = Dialect(
    "qft",
    [],
    [QFTAttr],
)
