from xdsl.dialects import builtin
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects import qref, qssa


class ConvertQrefGateToQssaGate(RewritePattern):
    """
    Replaces a qssa gate operation by its qref counterpart.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qref.GateOp | qref.DynGateOp, rewriter: PatternRewriter
    ):
        # Don't rewrite if uses live in different blocks
        if op.parent_block() is None:
            return
        for operand in op.ins:
            for use in operand.uses:
                if use.operation.parent_block() != op.parent_block():
                    return

        if isinstance(op, qref.GateOp):
            new_op = qssa.GateOp(op.gate, *op.ins)
        else:
            new_op = qssa.DynGateOp(op.gate, *op.ins)

        rewriter.replace_matched_op(new_op, ())

        for operand, result in zip(op.ins, new_op.results):
            operand.replace_by_if(result, lambda use: use.operation is not new_op)


class ConvertQrefMeasureToQssaMeasure(RewritePattern):
    """
    Replaces a qssa measurement by its qref counterpart.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.MeasureOp, rewriter: PatternRewriter):
        # Don't rewrite if uses live in different blocks
        if op.parent_block() is None:
            return
        for use in op.in_qubit.uses:
            if use.operation.parent_block() != op.parent_block():
                return

        # Don't rewrite if there are further uses of the measured qubit
        if len(op.in_qubit.uses) != 1:
            return

        new_op = qssa.MeasureOp(op.in_qubit)

        rewriter.replace_matched_op(new_op, (new_op.out,))


class ConvertQrefToQssa(ModulePass):
    """
    Converts uses of the qssa dialect to the qref dialect in a module.
    Inverse to the "convert-qref-to-qssa" pass.
    """

    name = "convert-qref-to-qssa"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertQrefGateToQssaGate(),
                    ConvertQrefMeasureToQssaMeasure(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
