from xdsl.dialects import builtin
from xdsl.dialects.arith import ConstantOp
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
    XZSOp,
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


class ToXZSGate(RewritePattern):
    """
    Rewrite a constant Identity/X/Y/Z/Phase gate to an xzs gadget
    """

    @staticmethod
    def get_const(b: bool, rewriter: PatternRewriter) -> ConstantOp:
        n = ConstantOp(builtin.BoolAttr.from_bool(b))
        n.result.name_hint = f"c{b}"
        rewriter.insert_op(n, InsertPoint.before(rewriter.current_operation))
        return n

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantGateOp, rewriter: PatternRewriter):
        match op.gate:
            case IdentityGate():
                false = self.get_const(False, rewriter)
                rewriter.replace_matched_op(XZSOp(false, false, false))
            case XGate():
                false = self.get_const(False, rewriter)
                true = self.get_const(True, rewriter)
                rewriter.replace_matched_op(XZSOp(true, false, false))
            case YGate():
                false = self.get_const(False, rewriter)
                true = self.get_const(True, rewriter)
                rewriter.replace_matched_op(XZSOp(true, true, false))
            case ZGate():
                false = self.get_const(False, rewriter)
                true = self.get_const(True, rewriter)
                rewriter.replace_matched_op(XZSOp(false, true, false))
            case PhaseGate():
                false = self.get_const(False, rewriter)
                true = self.get_const(True, rewriter)
                rewriter.replace_matched_op(XZSOp(false, false, true))
            case _:
                return


class ConvertToXZS(ModulePass):
    """
    Convert all Identity/X/Y/Z/Phase gates to xzs gadgets
    """

    name = "convert-to-xzs"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ToDynGate(), ToXZSGate()])
        ).rewrite_module(op)
