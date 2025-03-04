from abc import ABC


from inconspiquous.gates.core import SingleQubitGate, TwoQubitGate
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Dialect
from xdsl.irdl import irdl_attr_definition, ParameterDef
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from inconspiquous.gates import GateAttr


class Latexable(ABC):
    def return_latex(
        self,
    ) -> list[str]:
        raise NotImplementedError("Requires latex return")


@irdl_attr_definition
class HadamardGate(SingleQubitGate, Latexable):
    name = "qft.h"

    def return_latex(
        self,
    ):
        return ["\\gate{H}"]


@irdl_attr_definition
class SwapGate(TwoQubitGate, Latexable):
    name = "qft.swap"

    def return_latex(self, inq: int, outq: int):
        if inq < outq:
            return ["\\swapx", "\\swapx\\qw"]
        else:
            return ["\\swapx\\qw", "\\swapx"]


@irdl_attr_definition
class nthrootGate(TwoQubitGate, Latexable):
    name = "qft.pair"

    nthroot: ParameterDef[IntAttr]

    @property
    def nthroot(self) -> int:
        return self.nthroot.data

    def __init__(self, nthroot: int | IntAttr):
        if not isinstance(nthroot, IntAttr):
            nthroot = IntAttr(nthroot)

        super().__init__((nthroot,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr]:
        with parser.in_angle_brackets():
            nthroot = parser.parse_integer(allow_negative=False, allow_boolean=False)
        return (IntAttr.new(nthroot),)

    def print_parameters(self, printer: Printer) -> None:
        return self.nthroot.print_parameter(printer)

    def return_latex(self, inq: int, outq: int):
        return [
            "\\ctrl{" + str(outq - inq) + "}",
            "\\gate{Z^{1/" + str(self.nthroot) + "}}",
        ]


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
    [QFTAttr, HadamardGate, nthrootGate, SwapGate],
)
