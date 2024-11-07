from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import ComposeGateOp, ConstantGateOp, IdentityGate
from inconspiquous.dialects.qssa import GateOp
from inconspiquous.dialects.qssa import DynGateOp


class DynGateConst(RewritePattern):
    """
    Simplifies a dynamic gate with constant input to a regular gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        if isinstance(owner := op.gate.owner, ConstantGateOp):
            rewriter.replace_matched_op(GateOp(owner.gate, *op.ins))


class DynGateCompose(RewritePattern):
    """
    Simplifies a dynamic gate with composed input to two dynamic gates
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        if isinstance(gate := op.gate.owner, ComposeGateOp):
            dyn_gate_lhs = DynGateOp(gate.lhs, *op.ins)
            dyn_gate_rhs = DynGateOp(gate.rhs, dyn_gate_lhs)
            rewriter.replace_matched_op((dyn_gate_lhs, dyn_gate_rhs))


class GateIdentity(RewritePattern):
    """
    Remove an identity gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if isinstance(op.gate, IdentityGate):
            rewriter.replace_matched_op((), op.ins)
