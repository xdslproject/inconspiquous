from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import ConstantGateOp, IdentityGate
from inconspiquous.dialects.staged import DynGateOp, GateOp


class DynGateConst(RewritePattern):
    """
    Simplifies a dynamic gate with constant input to a regular gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        owner = op.gate.owner
        if isinstance(owner, ConstantGateOp):
            rewriter.replace_matched_op(GateOp(owner.gate, *op.in_qubits))


class GateIdentity(RewritePattern):
    """
    Remove an identity gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if isinstance(op.gate, IdentityGate):
            rewriter.replace_matched_op(())
