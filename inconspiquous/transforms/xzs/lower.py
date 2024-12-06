from xdsl.dialects import arith, builtin
from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
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
    XZSOp,
    ConstantGateOp,
    ZGate,
)


class LowerXZSToSelectPattern(RewritePattern):
    """Replace an XZS gadget by a selection between X/S/Z gates"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: XZSOp, rewriter: PatternRewriter):
        identity = ConstantGateOp(IdentityGate())
        s = ConstantGateOp(PhaseGate())
        z = ConstantGateOp(ZGate())
        x = ConstantGateOp(XGate())

        x_sel_op = arith.SelectOp(op.x, x, identity)
        z_sel_op = arith.SelectOp(op.z, z, identity)
        phase_sel_op = arith.SelectOp(op.phase, s, identity)

        comp_1 = ComposeGateOp(x_sel_op, z_sel_op)
        comp_2 = ComposeGateOp(comp_1, phase_sel_op)

        rewriter.replace_matched_op(
            (
                identity,
                s,
                z,
                x,
                x_sel_op,
                z_sel_op,
                phase_sel_op,
                comp_1,
                comp_2,
            )
        )


class LowerXZSToSelect(ModulePass):
    """Replace XZS gadgets by selections between X/S/Z gates"""

    name = "lower-xzs-to-select"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerXZSToSelectPattern()).rewrite_module(op)
