from __future__ import annotations
from typing import ClassVar

from xdsl.ir import Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AnyInt,
    GenericAttrConstraint,
    IRDLOperation,
    IntConstraint,
    IntVarConstraint,
    ParamAttrConstraint,
    ParameterDef,
    irdl_attr_definition,
    eq,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from inconspiquous.measurement import MeasurementAttr
from xdsl.dialects.builtin import IndexType, IntAttrConstraint, IntegerAttr
from xdsl.traits import ConstantLike, HasCanonicalizationPatternsTrait, Pure
from inconspiquous.constraints import SizedAttributeConstraint
from inconspiquous.dialects.angle import AngleAttr, AngleType


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


@irdl_attr_definition
class MeasurementType(ParametrizedAttribute, TypeAttribute):
    """
    A type for dynamic measurements.
    """

    name = "measurement.type"

    num_qubits: ParameterDef[IntegerAttr[IndexType]]

    def __init__(self, num_qubits: int | IntegerAttr[IndexType]):
        if isinstance(num_qubits, int):
            num_qubits = IntegerAttr.from_index_int_value(num_qubits)
        super().__init__((num_qubits,))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntegerAttr[IndexType]]:
        with parser.in_angle_brackets():
            i = parser.parse_integer(allow_boolean=False, allow_negative=False)
            return (IntegerAttr.from_index_int_value(i),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(str(self.num_qubits.value.data))

    @classmethod
    def constr(
        cls, int_constraint: IntConstraint | None = None
    ) -> GenericAttrConstraint[MeasurementType]:
        if int_constraint is None:
            return super().constr()
        return ParamAttrConstraint(
            MeasurementType,
            (
                IntegerAttr.constr(
                    value=IntAttrConstraint(int_constraint), type=eq(IndexType())
                ),
            ),
        )


@irdl_op_definition
class ConstantMeasurmentOp(IRDLOperation):
    """
    Constant-like operation for producing measurement types from measurement attributes.
    """

    _I: ClassVar = IntVarConstraint("I", AnyInt())

    name = "measurement.constant"

    measurement = prop_def(SizedAttributeConstraint(MeasurementAttr, _I))

    out = result_def(MeasurementType.constr(_I))

    assembly_format = "$measurement attr-dict"

    traits = traits_def(
        ConstantLike(),
        Pure(),
    )

    def __init__(self, measurement: MeasurementAttr):
        super().__init__(
            properties={
                "measurement": measurement,
            },
            result_types=(MeasurementType(measurement.num_qubits),),
        )


class XYDynMeasurementOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import measurement

        return (measurement.XYDynMeasurementConst(),)


@irdl_op_definition
class XYDynMeasurementOp(IRDLOperation):
    """
    Generate a measurement type for a measurement on the XY plane with input angle.
    """

    name = "measurement.dyn_xy"

    angle = operand_def(AngleType)

    out = result_def(MeasurementType(1))

    assembly_format = "`<` $angle `>` attr-dict"

    traits = traits_def(Pure(), XYDynMeasurementOpHasCanonicalizationPatterns())

    def __init__(self, angle: SSAValue | Operation):
        super().__init__(operands=(angle,), result_types=(MeasurementType(1),))


Measurement = Dialect(
    "measurement",
    [
        ConstantMeasurmentOp,
        XYDynMeasurementOp,
    ],
    [
        CompBasisMeasurementAttr,
        XYMeasurementAttr,
        MeasurementType,
    ],
)
