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

from inconspiquous.dialects.gate import XSGateOp


class XSSelectPattern(RewritePattern):
    """
    Rewrite an `arith.select` on two XS gates into a single gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Select, rewriter: PatternRewriter):
        if not isinstance(op.lhs, XSGateOp) or not isinstance(op.rhs, XSGateOp):
            return

        rewriter.replace_matched_op(
            XSGateOp(
                Select(op.cond, op.lhs.x, op.rhs.x),
                Select(op.cond, op.lhs.phase, op.rhs.phase),
            )
        )


class XSSelect(ModulePass):
    """
    Push `arith.select`s inwards past XS-gate operations
    """

    name = "xs-select"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(XSSelectPattern()).rewrite_op(op)
