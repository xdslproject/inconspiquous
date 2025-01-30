from xdsl.dialects.arith import XOrIOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand

from inconspiquous.dialects.angle import CondNegateAngleOp, ConstantAngleOp


class CondNegateAngleOpZeroPiPattern(RewritePattern):
    """
    Negating a zero angle on pi angle has no effect.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CondNegateAngleOp, rewriter: PatternRewriter):
        if (
            isinstance(op.angle.owner, ConstantAngleOp)
            and op.angle.owner.angle == -op.angle.owner.angle
        ):
            rewriter.replace_matched_op((), (op.angle,))


class CondNegateAngleOpFoldPattern(RewritePattern):
    """
    Fold an angle.cond_negate when both arguments are constant.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CondNegateAngleOp, rewriter: PatternRewriter):
        if (cond := const_evaluate_operand(op.cond)) is None:
            return
        if not cond:
            rewriter.replace_matched_op((), (op.angle,))

        if isinstance(op.angle.owner, ConstantAngleOp):
            rewriter.replace_matched_op(ConstantAngleOp(-op.angle.owner.angle))


class CondNegateAngleOpAssocPattern(RewritePattern):
    """
    Reassociate two conditional negations to a conditional negation on
    the xor of the conditions.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CondNegateAngleOp, rewriter: PatternRewriter):
        if not isinstance(op.angle.owner, CondNegateAngleOp):
            return
        xor = XOrIOp(op.cond, op.angle.owner.cond)
        rewriter.replace_matched_op((xor, CondNegateAngleOp(xor, op.angle.owner.angle)))
