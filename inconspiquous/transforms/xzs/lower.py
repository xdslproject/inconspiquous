from xdsl.dialects import arith, builtin
from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import (
    ComposeGateOp,
    IdentityGate,
    PhaseGate,
    XGate,
    XZOp,
    XZSOp,
    ConstantGateOp,
    YGate,
    ZGate,
)


class LowerXZToSelectPattern(RewritePattern):
    """Replace an XZ gadget by a selection between X/Z gates"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: XZOp, rewriter: PatternRewriter):
        identity = ConstantGateOp(IdentityGate())
        z = ConstantGateOp(ZGate())
        y = ConstantGateOp(YGate())
        x = ConstantGateOp(XGate())

        z_no_x_sel_op = arith.SelectOp(op.z, z, identity)
        z_x_sel_op = arith.SelectOp(op.z, y, x)
        x_sel_op = arith.SelectOp(op.x, z_x_sel_op, z_no_x_sel_op)

        rewriter.replace_matched_op(
            (
                identity,
                z,
                y,
                x,
                z_no_x_sel_op,
                z_x_sel_op,
                x_sel_op,
            )
        )


class LowerXZSToSelectPattern(RewritePattern):
    """Replace an XZS gadget by a selection between X/S/Z gates"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: XZSOp, rewriter: PatternRewriter):
        identity = ConstantGateOp(IdentityGate())
        s = ConstantGateOp(PhaseGate())
        z = ConstantGateOp(ZGate())
        y = ConstantGateOp(YGate())
        x = ConstantGateOp(XGate())

        z_no_x_sel_op = arith.SelectOp(op.z, z, identity)
        z_x_sel_op = arith.SelectOp(op.z, y, x)
        x_sel_op = arith.SelectOp(op.x, z_x_sel_op, z_no_x_sel_op)
        phase_sel_op = arith.SelectOp(op.phase, s, identity)
        comp = ComposeGateOp(x_sel_op, phase_sel_op)

        rewriter.replace_matched_op(
            (
                identity,
                s,
                z,
                y,
                x,
                z_no_x_sel_op,
                z_x_sel_op,
                x_sel_op,
                phase_sel_op,
                comp,
            )
        )


class LowerXZSToSelect(ModulePass):
    """Replace XZS gadgets by selections between X/S/Z gates"""

    name = "lower-xzs-to-select"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [LowerXZToSelectPattern(), LowerXZSToSelectPattern()]
            )
        ).rewrite_module(op)
