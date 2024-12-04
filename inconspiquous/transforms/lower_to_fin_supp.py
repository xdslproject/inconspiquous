from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.builtin import BoolAttr
from xdsl.ir import Operation, dataclass
from xdsl.parser import IntegerType
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects import arith
from xdsl.passes import ModulePass
from inconspiquous.dialects.prob import BernoulliOp, FinSuppOp, UniformOp


class LowerBernoulli(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BernoulliOp, rewriter: PatternRewriter):
        zero = arith.ConstantOp(BoolAttr.from_bool(False))
        one = arith.ConstantOp(BoolAttr.from_bool(True))
        rewriter.replace_matched_op(
            (zero, one, FinSuppOp((op.prob.value.data,), zero, one))
        )


@dataclass(frozen=True)
class LowerUniform(RewritePattern):
    max_size: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: UniformOp, rewriter: PatternRewriter):
        ty = op.out.type
        if not isinstance(ty, IntegerType):
            return

        if ty.bitwidth > self.max_size:
            return

        zero = arith.ConstantOp.from_int_and_width(0, ty.bitwidth)
        ops: list[Operation] = []
        for i in range(1, 2**ty.bitwidth):
            ops.append(arith.ConstantOp.from_int_and_width(i, ty.bitwidth))

        fin_supp = FinSuppOp(
            tuple(1.0 / (2**ty.bitwidth) for _ in range(1, 2**ty.bitwidth)), zero, *ops
        )

        ops.append(zero)
        ops.append(fin_supp)

        rewriter.replace_matched_op(ops)


@dataclass(frozen=True)
class LowerToFinSupp(ModulePass):
    max_size: int

    name = "lower-to-fin-supp"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerBernoulli(), LowerUniform(self.max_size)])
        ).rewrite_module(op)
