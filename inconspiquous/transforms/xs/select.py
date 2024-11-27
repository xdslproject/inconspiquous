from xdsl.dialects import builtin
from xdsl.dialects.arith import Select
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


class XSSelectPattern(RewritePattern):
    """
    Rewrite an `arith.select` on two XS gates into a single gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Select, rewriter: PatternRewriter):
        lhs = op.lhs.owner
        rhs = op.rhs.owner
        if not isinstance(lhs, XSGateOp) or not isinstance(rhs, XSGateOp):
            return

        sel_x = Select(op.cond, lhs.x, rhs.x)
        sel_phase = Select(op.cond, lhs.phase, rhs.phase)

        rewriter.insert_op((sel_x, sel_phase), InsertPoint.before(op))

        rewriter.replace_matched_op(
            XSGateOp(
                sel_x,
                sel_phase,
            )
        )


class XSSelect(ModulePass):
    """
    Push `arith.select`s inwards past XS-gate operations
    """

    name = "xs-select"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(XSSelectPattern()).rewrite_module(op)
