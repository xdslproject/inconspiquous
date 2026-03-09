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
from inconspiquous.dialects.measurement import ConstantMeasurementOp
from inconspiquous.dialects.qssa import ApplyOp, DynApplyOp, DynMeasureOp, MeasureOp


class DynGateConst(RewritePattern):
    """
    Simplifies a dynamic gate with constant input to a regular gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynApplyOp, rewriter: PatternRewriter):
        if isinstance(owner := op.gate.owner, ConstantInstrumentOp):
            rewriter.replace_matched_op(ApplyOp(owner.instrument, *op.ins))


class DynGateCompose(RewritePattern):
    """
    Simplifies a dynamic gate with composed input to two dynamic gates
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynApplyOp, rewriter: PatternRewriter):
        if isinstance(gate := op.gate.owner, ComposeGateOp):
            dyn_gate_lhs = DynApplyOp(gate.lhs, *op.ins)
            dyn_gate_rhs = DynApplyOp(gate.rhs, *dyn_gate_lhs.outs)
            rewriter.replace_matched_op((dyn_gate_lhs, dyn_gate_rhs))


class GateIdentity(RewritePattern):
    """
    Remove an identity gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter):
        if isinstance(op.gate, IdentityGate):
            rewriter.replace_matched_op((), op.ins)


class DynMeasureConst(RewritePattern):
    """
    Simplifies a dynamic measurement with constant measurement to a regular measurement
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynMeasureOp, rewriter: PatternRewriter):
        if isinstance(owner := op.measurement.owner, ConstantMeasurementOp):
            rewriter.replace_matched_op(
                MeasureOp(*op.in_qubits, measurement=owner.measurement)
            )
