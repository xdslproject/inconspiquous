from __future__ import annotations

from typing import TypeVar, Self

from xdsl.ir import Dialect, TypeAttribute, Operation, SSAValue, ParametrizedAttribute
from xdsl.dialects import builtin
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    IRDLOperation,
    traits_def,
    attr_def,
)
from xdsl.traits import Pure, ConstantLike


@irdl_attr_definition
class DurationType(ParametrizedAttribute, TypeAttribute):
    "Integer-like type with unknown, platform-specific width."

    # inspired by builtin's `index`
    name = "pulse.duration"

    # todo: make this parametric by underlying value type?


@irdl_attr_definition
class FrameType(ParametrizedAttribute, TypeAttribute):
    """An OpenPulse-like frame.

    A frame is also a software abstraction that acts as both a
       - clock within the quantum program with its time being incremented on each usage and
       - a stateful carrier signal defined by a frequency and phase.
    """

    name = "pulse.frame"
    frame_name: builtin.StringAttr

    @classmethod
    def from_name(cls, frame_name: str) -> Self:
        return cls(builtin.StringAttr(frame_name))


FrameT = TypeVar("FrameT", bound=FrameType)


@irdl_op_definition
class AllocFrame(IRDLOperation):
    name = "pulse.alloc_frame"
    result = result_def(FrameType)
    traits = traits_def()

    def __init__(self, frame: FrameType) -> None:
        super().__init__(result_types=[frame])


@irdl_op_definition
class ConstDuration(IRDLOperation):
    name = "pulse.const_duration"
    value = attr_def(builtin.IntAttr)
    result = result_def(DurationType())
    traits = traits_def(Pure(), ConstantLike())

    def __init__(self, value: int) -> None:
        super().__init__(
            attributes={"value": builtin.IntAttr(value)}, result_types=[DurationType()]
        )

    def const_value(self) -> int:
        return self.value.data


# todo: cast duration to/from more arith types?
@irdl_op_definition
class DurationFromInt(IRDLOperation):
    name = "pulse.duration_from_int"
    arg = operand_def(builtin.i32)
    result = result_def(DurationType())
    traits = traits_def(Pure())

    def __init__(
        self,
        operand: Operation | SSAValue,
    ):
        # todo: ensure operand is well typed?
        super().__init__(operands=[operand], result_types=[DurationType()])


@irdl_op_definition
class DurationToInt(IRDLOperation):
    name = "pulse.duration_to_int"
    arg = operand_def(DurationType())
    result = result_def(builtin.i32)
    traits = traits_def(Pure())

    def __init__(
        self,
        operand: Operation | SSAValue,
    ):
        super().__init__(operands=[operand], result_types=[DurationType()])


@irdl_op_definition
class Delay(IRDLOperation):
    name = "pulse.delay"
    duration = operand_def(DurationType())
    in_frame = operand_def(FrameType)
    out_frame = result_def(FrameType)

    traits = traits_def()  # todo: trait for timing / side effects?

    def __init__(self, duration: Operation | SSAValue, frame: Operation | SSAValue):
        result_type = SSAValue.get(frame).type
        super().__init__(operands=[duration, frame], result_types=[result_type])


Pulse = Dialect(
    "pulse",
    [AllocFrame, ConstDuration, DurationFromInt, DurationToInt, Delay],
    [
        FrameType,
        DurationType,
    ],
)
