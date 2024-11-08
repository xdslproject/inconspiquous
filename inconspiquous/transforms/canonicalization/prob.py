from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import BoolAttr
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.prob import BernoulliOp, FinSuppOp


class BernoulliConst(RewritePattern):
    """
    prob.bernoulli 1.0 == arith.constant true
    prob.bernoulli 0.0 == arith.constant false
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BernoulliOp, rewriter: PatternRewriter):
        prob = op.prob.value.data
        if prob == 1.0:
            rewriter.replace_matched_op(Constant(BoolAttr.from_bool(True)))

        if prob == 0.0:
            rewriter.replace_matched_op(Constant(BoolAttr.from_bool(False)))


class FinSuppTrivial(RewritePattern):
    """
    prob.fin_supp [ %x ] == %x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FinSuppOp, rewriter: PatternRewriter):
        if not op.probabilities.data.data:
            rewriter.replace_matched_op((), (op.default_value,))


class FinSuppRemoveCase(RewritePattern):
    """
    A case can be removed if its probability is 0 or it's equal to the default case.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FinSuppOp, rewriter: PatternRewriter):
        probs = op.probabilities.as_tuple()
        if not any(
            p == 0.0 or c == op.default_value
            for p, c in zip(probs, op.ins, strict=True)
        ):
            return
        new_probabilities = tuple(
            p
            for p, c in zip(probs, op.ins, strict=True)
            if p != 0.0 and c != op.default_value
        )
        new_ins = tuple(
            c
            for p, c in zip(probs, op.ins, strict=True)
            if p != 0.0 and c != op.default_value
        )
        rewriter.replace_matched_op(
            FinSuppOp(new_probabilities, op.default_value, *new_ins)
        )


class FinSuppDuplicate(RewritePattern):
    """
    If two cases are the same then we can merge them.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FinSuppOp, rewriter: PatternRewriter):
        print(op.ins)
        if len(set(op.ins)) == len(op.ins):
            return
        seen: dict[SSAValue, int] = dict()
        new_probs: list[float] = []
        new_ins: list[SSAValue] = []

        for p, c in zip(op.probabilities.as_tuple(), op.ins, strict=True):
            if c not in seen:
                seen[c] = len(new_probs)
                new_probs.append(p)
                new_ins.append(c)
            else:
                new_probs[seen[c]] += p

        rewriter.replace_matched_op(FinSuppOp(new_probs, op.default_value, *new_ins))
