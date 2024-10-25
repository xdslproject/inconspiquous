from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import ConstantGateOp
from inconspiquous.dialects.qref import GateOp
from inconspiquous.dialects.qref import DynGateOp


class DynGateConst(RewritePattern):
    """
    Simplifies a dynamic gate with constant input to a regular gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        owner = op.gate.owner
        if isinstance(owner, ConstantGateOp):
            rewriter.replace_matched_op(GateOp(owner.gate, *op.ins))
