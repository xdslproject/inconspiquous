from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import (
    ComposeGateOp,
    IdentityGate,
)
from inconspiquous.dialects.instrument import ConstantInstrumentOp
from inconspiquous.dialects.qssa import (
    DynGateOp,
    QssaApplyInterface,
    QssaDynamicApplyInterface,
)


class DynApplyConst(RewritePattern):
    """
    Simplifies a dynamic application with constant input to a regular application
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: QssaDynamicApplyInterface, rewriter: PatternRewriter, /
    ):
        if isinstance(owner := op.get_instrument().owner, ConstantInstrumentOp):
            rewriter.replace_matched_op(
                QssaApplyInterface.create_op(owner.instrument, *op.in_qubits)
            )


class DynGateCompose(RewritePattern):
    """
    Simplifies a dynamic gate with composed input to two dynamic gates
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: QssaDynamicApplyInterface, rewriter: PatternRewriter
    ):
        if isinstance(owner := op.get_instrument().owner, ComposeGateOp):
            dyn_gate_lhs = DynGateOp(owner.lhs, *op.in_qubits)
            dyn_gate_rhs = DynGateOp(owner.rhs, *dyn_gate_lhs.out_qubits)
            rewriter.replace_matched_op((dyn_gate_lhs, dyn_gate_rhs))


class GateIdentity(RewritePattern):
    """
    Remove an identity gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: QssaApplyInterface, rewriter: PatternRewriter):
        if isinstance(op.get_instrument(), IdentityGate):
            rewriter.replace_matched_op((), op.in_qubits)
