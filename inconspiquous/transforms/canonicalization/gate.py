from xdsl.ir import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand

from inconspiquous.dialects import gate, instrument
from inconspiquous.dialects.angle import ConstantAngleOp


class XZSToXZPattern(RewritePattern):
    """
    Convert an XZS gadget with 0 phase to an XZ gadget.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gate.XZSOp, rewriter: PatternRewriter):
        if const_evaluate_operand(op.phase) == 0:
            rewriter.replace_matched_op(gate.XZOp(op.x, op.z))


@dataclass(frozen=True)
class DynRotationGateToRotationPattern(RewritePattern):
    """
    Convert a dynamic rotation gate with a constant angle to the given rotation gate.
    """

    rot_gate: type[gate.RotationGate]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gate.DynRotationGate, rewriter: PatternRewriter):
        if isinstance(op.angle.owner, ConstantAngleOp):
            rewriter.replace_matched_op(
                instrument.ConstantInstrumentOp(self.rot_gate(op.angle.owner.angle))
            )
