from xdsl.dialects import builtin
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects import qssa
from inconspiquous.dialects.gate import (
    CXGate,
    CZGate,
    HadamardGate,
    JGate,
    PhaseGate,
    RZGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)


class ToCZJPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qssa.GateOp, rewriter: PatternRewriter):
        match op.gate:
            case CXGate():
                j1 = qssa.GateOp(JGate(0), op.ins[1])
                cz = qssa.GateOp(CZGate(), op.ins[0], j1)
                j2 = qssa.GateOp(JGate(0), cz.outs[1])
                rewriter.replace_matched_op((j1, cz, j2), (cz.outs[0], j2.outs[0]))
            case ZGate():
                j1 = qssa.GateOp(JGate(0), op.ins[0])
                j2 = qssa.GateOp(JGate(1), j1)
                rewriter.replace_matched_op((j1, j2))
            case XGate():
                j1 = qssa.GateOp(JGate(1), op.ins[0])
                j2 = qssa.GateOp(JGate(0), j1)
                rewriter.replace_matched_op((j1, j2))
            case YGate():
                j1 = qssa.GateOp(JGate(1), op.ins[0])
                j2 = qssa.GateOp(JGate(1), j1)
                rewriter.replace_matched_op((j1, j2))
            case HadamardGate():
                rewriter.replace_matched_op(qssa.GateOp(JGate(0), op.ins[0]))
            case PhaseGate():
                j1 = qssa.GateOp(JGate(0), op.ins[0])
                j2 = qssa.GateOp(JGate(0.5), j1)
                rewriter.replace_matched_op((j1, j2))
            case TGate():
                j1 = qssa.GateOp(JGate(0), op.ins[0])
                j2 = qssa.GateOp(JGate(0.25), j1)
                rewriter.replace_matched_op((j1, j2))
            case RZGate():
                j1 = qssa.GateOp(JGate(0), op.ins[0])
                j2 = qssa.GateOp(JGate(op.gate.angle), j1)
                rewriter.replace_matched_op((j1, j2))
            case _:
                return


class ToCZJPass(ModulePass):
    name = "convert-to-cz-j"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ToCZJPattern()).rewrite_module(op)
