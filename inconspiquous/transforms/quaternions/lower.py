from xdsl.dialects import builtin
from xdsl.dialects.arith import Constant
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)

from inconspiquous.dialects.gate import (
    ConstantGateOp,
    IdentityGate,
    PhaseGate,
    QuaternionGateOp,
    XGate,
    YGate,
    ZGate,
)
from inconspiquous.dialects.qssa import DynGateOp, GateOp


class MakeGateDynamic(RewritePattern):
    """
    Make an Identity/X/Y/Z/Phase gate into a dynamic gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if op.gate not in (IdentityGate(), XGate(), YGate(), ZGate(), PhaseGate()):
            return
        constant = ConstantGateOp(op.gate)
        dyn_gate = DynGateOp(constant, *op.ins)
        rewriter.replace_matched_op((constant, dyn_gate), dyn_gate.results)


class LowerGateToQuaternion(RewritePattern):
    """
    Lower a constant Identity/X/Y/Z/Phase gate to a quaternion.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantGateOp, rewriter: PatternRewriter):
        if op.gate not in (IdentityGate(), XGate(), YGate(), ZGate(), PhaseGate()):
            return
        zero = Constant.from_int_and_width(0, 64)
        zero.result.name_hint = "c0"
        one = Constant.from_int_and_width(1, 64)
        one.result.name_hint = "c1"

        match op.gate:
            case IdentityGate():
                new = QuaternionGateOp(one, zero, zero, zero)
                rewriter.replace_matched_op((zero, one, new), new.results)
            case XGate():
                new = QuaternionGateOp(zero, one, zero, zero)
                rewriter.replace_matched_op((zero, one, new), new.results)
            case YGate():
                new = QuaternionGateOp(zero, zero, one, zero)
                rewriter.replace_matched_op((zero, one, new), new.results)
            case ZGate():
                new = QuaternionGateOp(zero, zero, zero, one)
                rewriter.replace_matched_op((zero, one, new), new.results)
            case _:
                new = QuaternionGateOp(one, zero, zero, one)
                rewriter.replace_matched_op((zero, one, new), new.results)


class LowerToQuaternion(ModulePass):
    """
    Converts all Identity/X/Y/Z/Phase gates to dynamic quaternion gates
    """

    name = "lower-to-quaternion"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            MakeGateDynamic(),
            apply_recursively=False,
        ).rewrite_op(op)
        CommonSubexpressionElimination().apply(ctx, op)
        PatternRewriteWalker(
            LowerGateToQuaternion(),
            apply_recursively=False,
        ).rewrite_op(op)
        CommonSubexpressionElimination().apply(ctx, op)
