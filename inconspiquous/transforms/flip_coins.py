from xdsl.ir import dataclass, field
from xdsl.parser import Context, IntegerAttr, IntegerType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects import arith, builtin
from inconspiquous.dialects.prob import BernoulliOp, UniformOp

import random


class FlipCoinBernoulliPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BernoulliOp, rewriter: PatternRewriter, /):
        r = random.random()
        if r < op.prob.value.data:
            rewriter.replace_matched_op(arith.ConstantOp.from_int_and_width(1, 1))
        else:
            rewriter.replace_matched_op(arith.ConstantOp.from_int_and_width(0, 1))


class FlipCoinUniformPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: UniformOp, rewriter: PatternRewriter, /):
        assert isinstance(op.out.type, IntegerType)
        (a, b) = op.out.type.value_range()
        r = random.randrange(a, b)
        rewriter.replace_matched_op(arith.ConstantOp(IntegerAttr(r, op.out.type)))


@dataclass(frozen=True)
class FlipCoins(ModulePass):
    """
    Replace all probabalistic values by randomly generated constant values.

    Parameters:
      "seed": seed for random number generation.
    """

    name = "flip-coins"

    seed: int | None = field(default=None)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        random.seed(self.seed)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [FlipCoinBernoulliPattern(), FlipCoinUniformPattern()]
            )
        ).rewrite_module(op)
