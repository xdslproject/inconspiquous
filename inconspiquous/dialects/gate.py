from __future__ import annotations
import math

from xdsl.dialects.builtin import FloatAttr, Float64Type
from xdsl.ir import Dialect, ParametrizedAttribute
from xdsl.irdl import ParameterDef, irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from inconspiquous.gates import GateAttr, SingleQubitGate, TwoQubitGate


@irdl_attr_definition
class AngleAttr(ParametrizedAttribute):
    """
    Attribute that wraps around a float attr, implicitly keeping it in the range
    [0, 2) and implicitly multiplying by pi
    """

    name = "gate.angle"
    data: ParameterDef[FloatAttr[Float64Type]]

    def __init__(self, f: float):
        f_attr: FloatAttr[Float64Type] = FloatAttr(f % 2, 64)
        super().__init__((f_attr,))

    @property
    def as_float_raw(self) -> float:
        return self.data.value.data

    @property
    def as_float(self) -> float:
        return self.as_float_raw * math.pi

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[FloatAttr[Float64Type]]:
        with parser.in_angle_brackets():
            is_negative = parser.parse_optional_punctuation("-") is not None
            f = parser.parse_optional_number()
            if f is None:
                f = 1.0
            if isinstance(f, int):
                f = float(f)
            if f == 0.0:
                parser.parse_optional_keyword("pi")
            else:
                parser.parse_keyword("pi")
            if is_negative:
                f = -f
            return (FloatAttr(f % 2, 64),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            f = self.as_float_raw
            if f == 0.0:
                printer.print_string("0")
            elif f == 1.0:
                printer.print_string("pi")
            else:
                printer.print_string(f"{f}pi")

    def __add__(self, other: AngleAttr) -> AngleAttr:
        return AngleAttr(self.data.value.data + other.data.value.data)

    def __sub__(self, other: AngleAttr) -> AngleAttr:
        return AngleAttr(self.data.value.data - other.data.value.data)

    def __neg__(self) -> AngleAttr:
        return AngleAttr(-self.data.value.data)


@irdl_attr_definition
class HadamardGate(SingleQubitGate):
    name = "gate.h"


@irdl_attr_definition
class XGate(SingleQubitGate):
    name = "gate.x"


@irdl_attr_definition
class YGate(SingleQubitGate):
    name = "gate.y"


@irdl_attr_definition
class ZGate(SingleQubitGate):
    name = "gate.z"


@irdl_attr_definition
class PhaseGate(SingleQubitGate):
    name = "gate.s"


@irdl_attr_definition
class TGate(SingleQubitGate):
    name = "gate.t"


@irdl_attr_definition
class RZGate(SingleQubitGate):
    name = "gate.rz"

    angle: ParameterDef[AngleAttr]

    def __init__(self, angle: float | AngleAttr):
        if not isinstance(angle, AngleAttr):
            angle = AngleAttr(angle)

        super().__init__((angle,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[AngleAttr]:
        return (AngleAttr.new(AngleAttr.parse_parameters(parser)),)

    def print_parameters(self, printer: Printer) -> None:
        return self.angle.print_parameters(printer)


@irdl_attr_definition
class CNotGate(TwoQubitGate):
    name = "gate.cnot"


@irdl_attr_definition
class CZGate(TwoQubitGate):
    name = "gate.cz"


@irdl_attr_definition
class ToffoliGate(GateAttr):
    name = "gate.toffoli"

    def num_qubits(self) -> int:
        return 3


Gate = Dialect(
    "gate",
    [],
    [
        AngleAttr,
        HadamardGate,
        XGate,
        YGate,
        ZGate,
        PhaseGate,
        TGate,
        RZGate,
        CNotGate,
        CZGate,
        ToffoliGate,
    ],
)
