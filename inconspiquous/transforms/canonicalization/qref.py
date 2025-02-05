from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.gate import ConstantGateOp
from inconspiquous.dialects.measurement import ConstantMeasurementOp
from inconspiquous.dialects.qref import DynMeasureOp, GateOp, DynGateOp, MeasureOp


class DynGateConst(RewritePattern):
    """
    Simplifies a dynamic gate with constant input to a regular gate
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        owner = op.gate.owner
        if isinstance(owner, ConstantGateOp):
            rewriter.replace_matched_op(GateOp(owner.gate, *op.ins))


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
