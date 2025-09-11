from xdsl.ir import Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.dialects import builtin, arith
from xdsl.context import Context
from inconspiquous.dialects import pulse


class FoldArithConstDurations(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /) -> None:
        match op:
            case pulse.DurationFromInt():
                (arg,) = op.operands
                if not isinstance(arg, OpResult):
                    return
                if not isinstance(arg.op, arith.ConstantOp):
                    return
                value = arg.op.value
                if not isinstance(value, builtin.IntegerAttr):
                    return
                const_duration = pulse.ConstDuration(value.value.data)
                rewriter.replace_matched_op([const_duration], const_duration.results)
            case pulse.DurationToInt():
                breakpoint()
            case _:
                return


class FoldArithConstDurationsPass(ModulePass):
    name = "fold-arith-const-durations"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pattern = FoldArithConstDurations()
        PatternRewriteWalker(pattern).rewrite_module(op)
