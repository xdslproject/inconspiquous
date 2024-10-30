from xdsl.dialects import builtin
from xdsl.dialects.arith import Addi, Muli
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from inconspiquous.dialects.gate import XSGateOp
from inconspiquous.dialects.qssa import DynGateOp


class MergeXSGatesPattern(RewritePattern):
    """
    Merge two consecutive XS gates
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        gate2 = op.gate.owner
        if not isinstance(gate2, XSGateOp):
            return

        predecessor = op.ins[0].owner

        if not isinstance(predecessor, DynGateOp):
            return

        gate1 = op.gate.owner
        if not isinstance(gate1, XSGateOp):
            return

        new_x = Muli(gate1.x, gate2.x)
        new_phase_mul = Muli(gate1.phase, gate2.x)
        new_phase = Addi(new_phase_mul, gate2.phase)
        new_gate = XSGateOp(new_x, new_phase)

        rewriter.insert_op(
            (new_x, new_phase_mul, new_phase, new_gate), InsertPoint.before(op)
        )
        rewriter.replace_matched_op(DynGateOp(new_gate))
        rewriter.erase_op(predecessor)


class MergeXSGates(ModulePass):
    """
    Merge consecutive XS gates and push arith.select inwards
    """

    name = "merge-xs-gates"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(MergeXSGatesPattern()).rewrite_op(op)
