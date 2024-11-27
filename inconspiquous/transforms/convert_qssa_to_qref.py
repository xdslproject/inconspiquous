from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects import qssa, qref


class ConvertQssaGateToQrefGate(RewritePattern):
    """
    Replaces a qssa gate operation by its qref counterpart.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qssa.GateOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(qref.GateOp(op.gate, *op.ins), op.operands)


class ConvertQssaDynGateToQrefDynGate(RewritePattern):
    """
    Replaces a qssa dyn_gate operation by its qref counterpart.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qssa.DynGateOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(qref.DynGateOp(op.gate, *op.ins), op.ins)


class ConvertQssaMeasureToQrefMeasure(RewritePattern):
    """
    Replaces a qssa measurement by its qref counterpart.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qssa.MeasureOp, rewriter: PatternRewriter):
        new_measure = qref.MeasureOp(op.in_qubit)
        rewriter.replace_matched_op(new_measure, (new_measure.out, op.in_qubit))


class ConvertQssaToQref(ModulePass):
    """
    Converts uses of the qssa dialect to the qref dialect in a module.
    Inverse to the "convert-qref-to-qssa" pass.
    """

    name = "convert-qssa-to-qref"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertQssaGateToQrefGate(),
                    ConvertQssaDynGateToQrefDynGate(),
                    ConvertQssaMeasureToQrefMeasure(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
