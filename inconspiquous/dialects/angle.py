from __future__ import annotations

import math
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.dialects.builtin import Float64Type, FloatAttr, i1
from xdsl.parser import AttrParser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import ConstantLike, HasCanonicalizationPatternsTrait, Pure


@irdl_attr_definition
class AngleAttr(ParametrizedAttribute):
    """
    Attribute that wraps around a float attr, implicitly keeping it in the range
    [0, 2) and implicitly multiplying by pi
    """

    name = "angle.attr"
    data: FloatAttr[Float64Type]

    def __init__(self, f: float):
        f_attr: FloatAttr[Float64Type] = FloatAttr(f % 2, 64)
        super().__init__(f_attr)

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
class AngleType(ParametrizedAttribute, TypeAttribute):
    """
    A type for runtime angle values.
    """

    name = "angle.type"


@irdl_op_definition
class ConstantAngleOp(IRDLOperation):
    """
    Constant-like operation for producing angles
    """

    name = "angle.constant"

    angle = prop_def(AngleAttr)

    out = result_def(AngleType)

    assembly_format = "`` $angle attr-dict"

    traits = traits_def(
        ConstantLike(),
        Pure(),
    )

    def __init__(self, angle: AngleAttr):
        super().__init__(
            properties={
                "angle": angle,
            },
            result_types=(AngleType(),),
        )


class CondNegateAngleOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import angle

        return (
            angle.CondNegateAngleOpZeroPiPattern(),
            angle.CondNegateAngleOpFoldPattern(),
            angle.CondNegateAngleOpAssocPattern(),
        )


@irdl_op_definition
class CondNegateAngleOp(IRDLOperation):
    """
    Negates an angle if input condition is true.
    """

    name = "angle.cond_negate"

    cond = operand_def(i1)

    angle = operand_def(AngleType)

    out = result_def(AngleType)

    traits = traits_def(CondNegateAngleOpHasCanonicalizationPatterns(), Pure())

    assembly_format = "$cond `,` $angle attr-dict"

    def __init__(self, cond: SSAValue | Operation, angle: SSAValue | Operation):
        super().__init__(operands=(cond, angle), result_types=(AngleType(),))


Angle = Dialect("angle", [ConstantAngleOp, CondNegateAngleOp], [AngleAttr, AngleType])
