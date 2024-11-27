from xdsl.dialects import builtin
from xdsl.dialects.arith import Constant
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from inconspiquous.dialects.gate import (
    ConstantGateOp,
    IdentityGate,
    PhaseGate,
    XGate,
    XSGateOp,
    YGate,
    ZGate,
)
from inconspiquous.dialects.qssa import DynGateOp, GateOp


class ToDynGate(RewritePattern):
    """
    Rewrite a static Identity/X/Y/Z/Phase gate to a dynamic gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, IdentityGate | XGate | YGate | ZGate | PhaseGate):
            return

        constant = ConstantGateOp(op.gate)
        constant.out.name_hint = "g"
        rewriter.insert_op(constant, InsertPoint.before(op))

        rewriter.replace_matched_op(DynGateOp(constant, *op.ins))


class ToXSGate(RewritePattern):
    """
    Rewrite a constant Identity/X/Y/Z/Phase gate to an xs gate
    """

    @staticmethod
    def get_const(i: int, rewriter: PatternRewriter) -> Constant:
        n = Constant.from_int_and_width(i, 2)
        n.result.name_hint = f"c{i}"
        rewriter.insert_op(n, InsertPoint.before(rewriter.current_operation))
        return n

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantGateOp, rewriter: PatternRewriter):
        match op.gate:
            case IdentityGate():
                rewriter.replace_matched_op(
                    XSGateOp(self.get_const(1, rewriter), self.get_const(0, rewriter))
                )
            case XGate():
                rewriter.replace_matched_op(
                    XSGateOp(self.get_const(3, rewriter), self.get_const(0, rewriter))
                )
            case YGate():
                rewriter.replace_matched_op(
                    XSGateOp(self.get_const(3, rewriter), self.get_const(2, rewriter))
                )
            case ZGate():
                rewriter.replace_matched_op(
                    XSGateOp(self.get_const(1, rewriter), self.get_const(2, rewriter))
                )
            case PhaseGate():
                rewriter.replace_matched_op(
                    XSGateOp(self.get_const(1, rewriter), self.get_const(1, rewriter))
                )
            case _:
                return


class ConvertToXS(ModulePass):
    """
    Convert all Identity/X/Y/Z/Phase gates to xs gates
    """

    name = "convert-to-xs"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ToDynGate(), ToXSGate()])
        ).rewrite_module(op)
