from collections.abc import Sequence
from typing import Literal

from xdsl.dialects.builtin import I1, DenseArrayBase, i1
from xdsl.ir import Attribute, Dialect, TypeAttribute, VerifyException
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from inconspiquous.dialects.gate import GateAttr
from inconspiquous.dialects.instrument import InstrumentAttr


@irdl_attr_definition
class PerfectCode5QubitCorrectionAttr(GateAttr):
    name = "qec.perfect"

    @property
    def num_qubits(self) -> int:
        return 5


@irdl_attr_definition
class StabilizerAttr(InstrumentAttr):
    name = "qec.stabilizer"

    x_stabilizer: DenseArrayBase[I1]
    z_stabilizer: DenseArrayBase[I1]

    def __init__(self, *stabilizers: Literal["I", "X", "Y", "Z"]):
        x_stabilizer = DenseArrayBase.from_list(
            i1, tuple(c in ("X", "Y") for c in stabilizers)
        )
        z_stabilizer = DenseArrayBase.from_list(
            i1, tuple(c in ("Y", "Z") for c in stabilizers)
        )

        super().__init__(x_stabilizer, z_stabilizer)

    def verify(self) -> None:
        if len(self.x_stabilizer) != len(self.z_stabilizer):
            raise VerifyException(
                "x_stabilizer and z_stabilizer arrays should have same length, "
                f"got {len(self.x_stabilizer)} and {len(self.z_stabilizer)}"
            )

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            start_pos = parser.pos
            ident = parser.parse_optional_identifier() or ""
        for i, c in enumerate(ident):
            if c not in ("I", "X", "Y", "Z"):
                parser.raise_error(
                    "Stabilizer should be one of 'I', 'X', 'Y', or 'Z'",
                    at_position=start_pos + i,
                    end_position=start_pos + i + 1,
                )

        x_stabilizer = DenseArrayBase.from_list(
            i1, tuple(c in ("X", "Y") for c in ident)
        )
        z_stabilizer = DenseArrayBase.from_list(
            i1, tuple(c in ("Y", "Z") for c in ident)
        )

        return (x_stabilizer, z_stabilizer)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            for x, z in zip(
                self.x_stabilizer.iter_values(),
                self.z_stabilizer.iter_values(),
                strict=True,
            ):
                if x:
                    if z:
                        printer.print_string("Y")
                    else:
                        printer.print_string("X")
                else:
                    if z:
                        printer.print_string("Z")
                    else:
                        printer.print_string("I")

    @property
    def num_qubits(self) -> int:
        return len(self.x_stabilizer)

    @property
    def classical_results(self) -> tuple[TypeAttribute, ...]:
        return (i1,)


QEC = Dialect(
    "qec",
    [],
    [
        PerfectCode5QubitCorrectionAttr,
        StabilizerAttr,
    ],
)
