from xdsl.dialects import builtin
from xdsl.dialects.arith import SelectOp
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import XZSOp


class XZSSelectPattern(RewritePattern):
    """
    Rewrite an `arith.select` on two XZS gadgets into a single gadget
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SelectOp, rewriter: PatternRewriter):
        lhs = op.lhs.owner
        rhs = op.rhs.owner
        if not isinstance(lhs, XZSOp) or not isinstance(rhs, XZSOp):
            return

        sel_x = SelectOp(op.cond, lhs.x, rhs.x)
        sel_z = SelectOp(op.cond, lhs.z, rhs.z)
        sel_phase = SelectOp(op.cond, lhs.phase, rhs.phase)

        rewriter.replace_matched_op(
            (
                sel_x,
                sel_z,
                sel_phase,
                XZSOp(
                    sel_x,
                    sel_z,
                    sel_phase,
                ),
            )
        )


class XZSSelect(ModulePass):
    """
    Push `arith.select`s inwards past XZS-gadget operations
    """

    name = "xzs-select"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(XZSSelectPattern()).rewrite_module(op)
