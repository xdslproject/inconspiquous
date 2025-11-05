from xdsl.dialects.arith import ConstantOp, XOrIOp
from xdsl.parser import BoolAttr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand

from inconspiquous.dialects.angle import (
    CondNegateAngleOp,
    ConstantAngleOp,
    NegateAngleOp,
)


class NegateAngleOpFoldPattern(RewritePattern):
    """
    Fold the negation of a constant angle.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: NegateAngleOp, rewriter: PatternRewriter):
        if isinstance(op.angle.owner, ConstantAngleOp):
            rewriter.replace_matched_op(ConstantAngleOp(-op.angle.owner.angle))


class NegateMergePattern(RewritePattern):
    """
    Replace a negation on a (conditional) negation with the identity
    (or conditional negation with the complementary condition).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: NegateAngleOp, rewriter: PatternRewriter, /):
        arg = op.angle.owner

        if isinstance(arg, NegateAngleOp):
            rewriter.replace_matched_op((), (arg.angle,))
        elif isinstance(arg, CondNegateAngleOp):
            cTrue = ConstantOp(BoolAttr.from_bool(True))
            cTrue.result.name_hint = "cTrue"
            xor = XOrIOp(arg.cond, cTrue)
            rewriter.replace_matched_op((cTrue, xor, CondNegateAngleOp(xor, arg.angle)))


class CondNegateAngleOpZeroPiPattern(RewritePattern):
    """
    Negating a zero angle or pi angle has no effect.
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
    Fold an angle.cond_negate when the condition is constant.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CondNegateAngleOp, rewriter: PatternRewriter):
        if (cond := const_evaluate_operand(op.cond)) is None:
            return
        if not cond:
            rewriter.replace_matched_op((), (op.angle,))
        else:
            rewriter.replace_matched_op(NegateAngleOp(op.angle))


class CondNegateMergePattern(RewritePattern):
    """
    Replace a conditional negation on a (conditional) negation by a single
    conditional negation with updated condition.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CondNegateAngleOp, rewriter: PatternRewriter):
        arg = op.angle.owner
        if isinstance(arg, CondNegateAngleOp):
            xor = XOrIOp(op.cond, arg.cond)
        elif isinstance(arg, NegateAngleOp):
            cTrue = ConstantOp(BoolAttr.from_bool(True))
            cTrue.result.name_hint = "cTrue"
            rewriter.insert_op_before_matched_op(cTrue)
            xor = XOrIOp(op.cond, cTrue)
        else:
            return

        rewriter.replace_matched_op((xor, CondNegateAngleOp(xor, arg.angle)))
