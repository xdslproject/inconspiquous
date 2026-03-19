from xdsl.dialects import builtin
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects import qref, qssa


class ConvertQrefToQssaPattern(RewritePattern):
    """
    Replace a qref operation by its qssa counterpart.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: qref.QrefApplyInterface, rewriter: PatternRewriter, /
    ):
        # Don't rewrite if uses live in different blocks
        if op.parent_block() is None:
            return
        for operand in op.in_qubits:
            for use in operand.uses:
                if use.operation.parent_block() != op.parent_block():
                    return

        new_op = qssa.QssaApplyInterface.create_op(op.get_instrument(), *op.in_qubits)

        rewriter.replace_matched_op(new_op, new_op.get_outs())

        for operand, result in zip(op.in_qubits, new_op.out_qubits, strict=True):
            operand.replace_uses_with_if(
                result, lambda use: use.operation is not new_op
            )


class ConvertQrefToQssa(ModulePass):
    """
    Converts uses of the qssa dialect to the qref dialect in a module.
    Inverse to the "convert-qref-to-qssa" pass.
    """

    name = "convert-qref-to-qssa"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertQrefToQssaPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
