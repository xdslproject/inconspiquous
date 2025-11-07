from xdsl.ir import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
from inconspiquous.dialects import gate
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
                gate.ConstantGateOp(self.rot_gate(op.angle.owner.angle))
            )


class ControlOpFoldPattern(RewritePattern):
    """
    Converts the control of some constant gates to their constant controlled variants.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gate.ControlOp, rewriter: PatternRewriter):
        if not isinstance(op.gate.owner, gate.ConstantGateOp):
            return
        match op.gate.owner.gate:
            case gate.XGate():
                rewriter.replace_matched_op(gate.ConstantGateOp(gate.CXGate()))
            case gate.ZGate():
                rewriter.replace_matched_op(gate.ConstantGateOp(gate.CZGate()))
            case gate.CXGate():
                rewriter.replace_matched_op(gate.ConstantGateOp(gate.ToffoliGate()))
            case _:
                return
