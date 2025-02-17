from xdsl.dialects import builtin
from xdsl.dialects.arith import SelectOp, ConstantOp
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import XZOp, XZSOp


class XZSelectPattern(RewritePattern):
    """
    Rewrite an `arith.select` on two XZ gadgets into a single gadget
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SelectOp, rewriter: PatternRewriter):
        lhs = op.lhs.owner
        rhs = op.rhs.owner
        if not isinstance(lhs, XZOp) or not isinstance(rhs, XZOp):
            return

        sel_x = SelectOp(op.cond, lhs.x, rhs.x)
        sel_z = SelectOp(op.cond, lhs.z, rhs.z)

        rewriter.replace_matched_op(
            (
                sel_x,
                sel_z,
                XZOp(
                    sel_x,
                    sel_z,
                ),
            )
        )


class XZSSelectPattern(RewritePattern):
    """
    Rewrite an `arith.select` on two XZS gadgets into a single gadget
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SelectOp, rewriter: PatternRewriter):
        lhs = op.lhs.owner
        rhs = op.rhs.owner
        if not isinstance(lhs, XZSOp | XZOp) or not isinstance(rhs, XZSOp | XZOp):
            return

        sel_x = SelectOp(op.cond, lhs.x, rhs.x)
        sel_z = SelectOp(op.cond, lhs.z, rhs.z)
        c0 = None
        if isinstance(lhs, XZSOp):
            lhs_phase = lhs.phase
        else:
            c0 = ConstantOp.from_int_and_width(0, 1)
            lhs_phase = c0

        if isinstance(rhs, XZSOp):
            rhs_phase = rhs.phase
        else:
            if c0 is None:
                c0 = ConstantOp.from_int_and_width(0, 1)
            rhs_phase = c0
        sel_phase = SelectOp(op.cond, lhs_phase, rhs_phase)

        rewriter.replace_matched_op(
            (
                sel_x,
                sel_z,
                *((c0,) if c0 is not None else ()),
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

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([XZSelectPattern(), XZSSelectPattern()])
        ).rewrite_module(op)
