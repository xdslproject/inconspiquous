from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects import qref, qssa


class ConvertQssaToQrefPattern(RewritePattern):
    """
    Replace a qssa operation by its qref counterpart.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qssa.QssaApplyInterface, rewriter: PatternRewriter, /
    ):
        new_op = qref.QrefApplyInterface.create_op(op.get_instrument(), *op.in_qubits)
        rewriter.replace_matched_op(new_op, new_op.get_outs() + op.in_qubits)


class ConvertQssaToQref(ModulePass):
    """
    Converts uses of the qssa dialect to the qref dialect in a module.
    Inverse to the "convert-qref-to-qssa" pass.
    """

    name = "convert-qssa-to-qref"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertQssaToQrefPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
