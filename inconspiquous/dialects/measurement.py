from __future__ import annotations

from abc import ABC

from xdsl.dialects.builtin import i1
from xdsl.interfaces import HasCanonicalizationPatternsInterface
from xdsl.ir import Dialect, Operation, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import Pure

from inconspiquous.dialects.angle import AngleAttr, AngleType
from inconspiquous.dialects.instrument import InstrumentAttr, InstrumentType


class MeasurementAttr(InstrumentAttr, ABC):
    """
    Helper for instruments which act as traditional "measurements", i.e. have one
    boolean output for each qubit.
    """

    @property
    def classical_results(self) -> tuple[TypeAttribute, ...]:
        return (i1,) * self.num_qubits


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
class XBasisMeasurementAttr(MeasurementAttr):
    """
    An X basis measurement attribute.
    """

    name = "measurement.x_basis"

    @property
    def num_qubits(self) -> int:
        return 1


@irdl_attr_definition
class XYMeasurementAttr(MeasurementAttr):
    """
    An XY plane measurement attribute with specified angle.
    """

    name = "measurement.xy"

    angle: AngleAttr

    def __init__(self, angle: float | AngleAttr):
        if not isinstance(angle, AngleAttr):
            angle = AngleAttr(angle)

        super().__init__(angle)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[AngleAttr]:
        return (AngleAttr.new(AngleAttr.parse_parameters(parser)),)

    def print_parameters(self, printer: Printer) -> None:
        return self.angle.print_parameters(printer)

    @property
    def num_qubits(self) -> int:
        return 1


@irdl_op_definition
class XYDynMeasurementOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    """
    Generate a measurement type for a measurement on the XY plane with input angle.
    """

    name = "measurement.dyn_xy"

    angle = operand_def(AngleType)

    out = result_def(InstrumentType(1, i1))

    assembly_format = "`<` $angle `>` attr-dict"

    traits = traits_def(Pure())

    def __init__(self, angle: SSAValue | Operation):
        super().__init__(operands=(angle,), result_types=(InstrumentType(1, i1),))

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import measurement

        return (measurement.XYDynMeasurementConst(),)


Measurement = Dialect(
    "measurement",
    [
        XYDynMeasurementOp,
    ],
    [
        CompBasisMeasurementAttr,
        XBasisMeasurementAttr,
        XYMeasurementAttr,
    ],
)
