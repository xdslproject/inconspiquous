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
from inconspiquous.dialects.gate import XGate, YGate, ZGate


class PauliFusionPattern(RewritePattern):
    """
    Fuse adjacent Pauli gates.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qssa.GateOp, rewriter: PatternRewriter, /):
        if not isinstance(op.gate, XGate | YGate | ZGate):
            return

        prev = op.ins[0].owner
        if not isinstance(prev, qssa.GateOp):
            return

        if not isinstance(prev.gate, XGate | YGate | ZGate):
            return

        if op.gate == prev.gate:
            rewriter.replace_matched_op((), prev.ins)
            rewriter.erase_op(prev)
            return

        match (prev.gate, op.gate):
            case (XGate(), YGate()):
                new_gate = ZGate()
            case (XGate(), ZGate()):
                new_gate = YGate()
            case (YGate(), XGate()):
                new_gate = ZGate()
            case (YGate(), ZGate()):
                new_gate = XGate()
            case (ZGate(), XGate()):
                new_gate = YGate()
            case (ZGate(), YGate()):
                new_gate = XGate()
            case _:
                return

        rewriter.replace_matched_op(qssa.GateOp(new_gate, *prev.ins))
        rewriter.erase_op(prev)


class PauliFusionPass(ModulePass):
    name = "pauli-fusion"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(PauliFusionPattern()).rewrite_module(op)
