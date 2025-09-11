from xdsl.passes import ModulePass
from xdsl.ir import Operation, OpResult
from xdsl.context import Context
from xdsl.dialects import builtin
from inconspiquous.dialects import pulse

START_TIME = "start_time"
END_TIME = "end_time"


def get_time(op: Operation, key: str) -> int | None:
    if (attr := op.attributes.get(key)) is None:
        return None
    assert isinstance(attr, builtin.IntAttr)
    return attr.data


def get_const_duration(op: Operation) -> int | None:
    if isinstance(op, pulse.ConstDuration):
        return op.const_value()
    return None


def compute_frame_times(op: Operation) -> tuple[int, int] | None:
    match op:
        case pulse.AllocFrame():
            return (0, 0)
        case pulse.Delay():
            duration, frame = op.operands
            # todo: deal with block args
            if not isinstance(duration, OpResult) or not isinstance(frame, OpResult):
                return None
            if (delay_start := get_time(frame.op, END_TIME)) is None:
                return None
            if (dt := get_const_duration(duration.op)) is None:
                return None
            return (delay_start, delay_start + dt)
        case _:
            pass


class ComputeFrameTimesPass(ModulePass):
    name = "compute-frame-times"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for node in op.walk(region_first=True):
            if node.dialect_name() != "pulse":
                continue
            if START_TIME in op.attributes and END_TIME in op.attributes:
                continue
            if (times := compute_frame_times(node)) is None:
                continue
            start, stop = times
            node.attributes[START_TIME] = builtin.IntAttr(start)
            node.attributes[END_TIME] = builtin.IntAttr(stop)
