from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
from inconspiquous.dialects import gate
from inconspiquous.dialects.angle import ConstantAngleOp


class FoldCondOpPattern(RewritePattern):
    """
    Convert a conditional gate with constant condition.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gate.CondOp, rewriter: PatternRewriter, /):
        c = const_evaluate_operand(op.cond)
        if c is None:
            return
        if c:
            rewriter.replace_matched_op(gate.ConstantGateOp(op.gate))
        else:
            rewriter.replace_matched_op(
                gate.ConstantGateOp(gate.IdentityGate(op.gate.num_qubits))
            )


class XZSToXZPattern(RewritePattern):
    """
    Convert an XZS gadget with 0 phase to an XZ gadget.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gate.XZSOp, rewriter: PatternRewriter):
        if const_evaluate_operand(op.phase) == 0:
            rewriter.replace_matched_op(gate.XZOp(op.x, op.z))


class DynJGateToJPattern(RewritePattern):
    """
    Convert a dynamic J gate with a constant angle to a J gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gate.DynJGate, rewriter: PatternRewriter, /):
        if isinstance(op.angle.owner, ConstantAngleOp):
            rewriter.replace_matched_op(
                gate.ConstantGateOp(gate.JGate(op.angle.owner.angle))
            )
