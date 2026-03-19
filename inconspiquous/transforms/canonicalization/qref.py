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
from inconspiquous.dialects.qref import (
    DynGateOp,
    GateOp,
    QrefApplyInterface,
    QrefDynamicApplyInterface,
)


class DynApplyConst(RewritePattern):
    """
    Simplifies a dynamic application with constant input to a regular appliacation
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: QrefDynamicApplyInterface, rewriter: PatternRewriter
    ):
        if isinstance(owner := op.get_instrument().owner, ConstantInstrumentOp):
            rewriter.replace_matched_op(
                QrefApplyInterface.create_op(owner.instrument, *op.in_qubits)
            )


class DynGateCompose(RewritePattern):
    """
    Simplifies a dynamic gate with composed input to two dynamic gates
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: QrefDynamicApplyInterface, rewriter: PatternRewriter
    ):
        if isinstance(owner := op.get_instrument().owner, ComposeGateOp):
            dyn_gate_lhs = DynGateOp(owner.lhs, *op.in_qubits)
            dyn_gate_rhs = DynGateOp(owner.rhs, *op.in_qubits)
            rewriter.replace_matched_op((dyn_gate_lhs, dyn_gate_rhs))


class GateIdentity(RewritePattern):
    """
    Remove an identity gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if isinstance(op.gate, IdentityGate):
            rewriter.replace_matched_op(())
