from xdsl.dialects import arith
from xdsl.dialects import builtin
from xdsl.parser import DenseIntOrFPElementsAttr, MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.varith import VarithSwitchOp
from xdsl.dialects.builtin import IntegerType

from inconspiquous.dialects.gate import (
    ComposeGateOp,
    IdentityGate,
    PhaseGate,
    XGate,
    XSGateOp,
    ConstantGateOp,
    YGate,
    ZGate,
)


class LowerXSToSelectPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: XSGateOp, rewriter: PatternRewriter):
        case_values = DenseIntOrFPElementsAttr.vector_from_list(
            [1, 2, 3], IntegerType(2)
        )

        identity = ConstantGateOp(IdentityGate())
        s = ConstantGateOp(PhaseGate())
        z = ConstantGateOp(ZGate())
        zs = ComposeGateOp(z, s)
        x = ConstantGateOp(XGate())
        xs = ComposeGateOp(x, s)
        y = ConstantGateOp(YGate())
        sx = ComposeGateOp(s, x)

        one_case = VarithSwitchOp(op.phase, case_values, identity, s, z, zs)
        three_case = VarithSwitchOp(op.phase, case_values, x, xs, y, sx)

        one = arith.Constant.from_int_and_width(1, 2)
        cmpi = arith.Cmpi(op.x, one, "eq")

        rewriter.replace_matched_op(
            (
                identity,
                s,
                z,
                zs,
                x,
                xs,
                y,
                sx,
                one_case,
                three_case,
                one,
                cmpi,
                arith.Select(cmpi, one_case, three_case),
            )
        )


class LowerXSToSelect(ModulePass):
    name = "lower-xs-to-select"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerXSToSelectPattern()).rewrite_module(op)
