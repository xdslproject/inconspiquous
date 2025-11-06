from __future__ import annotations

import math
from xdsl.interfaces import ConstantLikeInterface, HasCanonicalizationPatternsInterface
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
from xdsl.dialects.builtin import FloatData, i1
from xdsl.parser import AttrParser, Float64Type
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import Pure


@irdl_attr_definition
class AngleAttr(ParametrizedAttribute):
    """
    Attribute that wraps around a float attr, implicitly keeping it in the range
    [0, 2) and implicitly multiplying by pi
    """

    name = "angle.attr"
    data: FloatData

    def __init__(self, f: float):
        f_attr = FloatData(f % 2)
        super().__init__(f_attr)

    def as_float_raw(self) -> float:
        return self.data.data

    def as_float(self) -> float:
        return self.as_float_raw() * math.pi

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[FloatData]:
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
            return (FloatData(f % 2),)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            f = self.as_float_raw()
            if f == 0.0:
                printer.print_string("0")
            elif f == 1.0:
                printer.print_string("pi")
            else:
                printer.print_string(f"{f}pi")

    def __add__(self, other: AngleAttr) -> AngleAttr:
        return AngleAttr(self.data.data + other.data.data)

    def __sub__(self, other: AngleAttr) -> AngleAttr:
        return AngleAttr(self.data.data - other.data.data)

    def __neg__(self) -> AngleAttr:
        return AngleAttr(-self.data.data)

    def __mul__(self, other: float):
        return AngleAttr(self.data.data * other)


@irdl_attr_definition
class AngleType(ParametrizedAttribute, TypeAttribute):
    """
    A type for runtime angle values.
    """

    name = "angle.type"


@irdl_op_definition
class ConstantAngleOp(IRDLOperation, ConstantLikeInterface):
    """
    Constant-like operation for producing angles
    """

    name = "angle.constant"

    angle = prop_def(AngleAttr)

    out = result_def(AngleType)

    assembly_format = "`` $angle attr-dict"

    traits = traits_def(
        Pure(),
    )

    def __init__(self, angle: AngleAttr):
        super().__init__(
            properties={
                "angle": angle,
            },
            result_types=(AngleType(),),
        )

    def get_constant_value(self) -> AngleAttr:
        return self.angle


@irdl_op_definition
class NegateAngleOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    """
    Negate an angle.
    """

    name = "angle.negate"

    angle = operand_def(AngleType)

    out = result_def(AngleType)

    traits = traits_def(Pure())

    assembly_format = "$angle attr-dict"

    def __init__(self, angle: SSAValue | Operation):
        super().__init__(operands=(angle,), result_types=(AngleType(),))

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import angle

        return (angle.NegateAngleOpFoldPattern(), angle.NegateMergePattern())


@irdl_op_definition
class CondNegateAngleOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    """
    Negates an angle if input condition is true.
    """

    name = "angle.cond_negate"

    cond = operand_def(i1)

    angle = operand_def(AngleType)

    out = result_def(AngleType)

    traits = traits_def(Pure())

    assembly_format = "$cond `,` $angle attr-dict"

    def __init__(self, cond: SSAValue | Operation, angle: SSAValue | Operation):
        super().__init__(operands=(cond, angle), result_types=(AngleType(),))

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import angle

        return (
            angle.CondNegateAngleOpZeroPiPattern(),
            angle.CondNegateAngleOpFoldPattern(),
            angle.CondNegateMergePattern(),
        )


@irdl_op_definition
class ScaleAngleOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    """
    Scale an angle by a float or integer
    """

    name = "angle.scale"

    angle = operand_def(AngleType)

    scale = operand_def(Float64Type)

    out = result_def(AngleType)

    traits = traits_def(Pure())

    assembly_format = "$angle `,` $scale attr-dict"

    def __init__(self, angle: SSAValue | Operation, scale: SSAValue | Operation):
        super().__init__(operands=(angle, scale), result_types=(AngleType(),))

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import angle

        return (angle.ScaleAngleFoldPattern(),)


@irdl_op_definition
class AddAngleOp(IRDLOperation, HasCanonicalizationPatternsInterface):
    """
    Adds two angles.
    """

    name = "angle.add"

    lhs = operand_def(AngleType)

    rhs = operand_def(AngleType)

    out = result_def(AngleType)

    assembly_format = "$lhs `,` $rhs attr-dict"

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        super().__init__(operands=(lhs, rhs), result_types=(AngleType(),))

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization import angle

        return (angle.AddAngleFoldPattern(),)


Angle = Dialect(
    "angle",
    [
        ConstantAngleOp,
        NegateAngleOp,
        CondNegateAngleOp,
        ScaleAngleOp,
        AddAngleOp,
    ],
    [
        AngleAttr,
        AngleType,
    ],
)
