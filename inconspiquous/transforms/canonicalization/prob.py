from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import BoolAttr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.prob import BernoulliOp


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
