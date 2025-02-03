from __future__ import annotations

from xdsl.ir import Dialect
from xdsl.irdl import (
    ParameterDef,
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from inconspiquous.dialects.gate import AngleAttr
from inconspiquous.measurement import MeasurementAttr


@irdl_attr_definition
class CompBasisMeasurementAttr(MeasurementAttr):
    """
    A computational basis measurement attribute.
    """

    name = "measurement.comp_basis"

    @property
    def num_qubits(self) -> int:
        return 1


@irdl_attr_definition
class XYMeasurementAttr(MeasurementAttr):
    """
    An XY plane measurement attribute with specified angle.
    """

    name = "measurement.xy"

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

    @property
    def num_qubits(self) -> int:
        return 1


Measurement = Dialect(
    "measurement",
    [],
    [
        CompBasisMeasurementAttr,
        XYMeasurementAttr,
    ],
)
