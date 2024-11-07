from xdsl.dialects.builtin import IntegerAttrTypeConstr, i1
from xdsl.ir import Dialect, VerifyException
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import Float64Type, FloatAttr, IndexType, IntegerType
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait


class BernoulliOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.prob import BernoulliConst

        return (BernoulliConst(),)


@irdl_op_definition
class BernoulliOp(IRDLOperation):
    name = "prob.bernoulli"

    prob = prop_def(FloatAttr[Float64Type])

    out = result_def(i1)

    assembly_format = "$prob attr-dict"

    traits = traits_def(BernoulliOpHasCanonicalizationPatterns())

    def __init__(self, prob: float | FloatAttr[Float64Type]):
        if isinstance(prob, float):
            prob = FloatAttr(prob, 64)

        # Why is this needed?
        assert not isinstance(prob, int)

        super().__init__(
            properties={
                "prob": prob,
            },
            result_types=(i1,),
        )

    def verify_(self) -> None:
        prob = self.prob.value.data
        if prob < 0 or prob > 1:
            raise VerifyException(
                f"Property 'prob' = {prob} should be in the range [0, 1]"
            )


@irdl_op_definition
class UniformOp(IRDLOperation):
    name = "prob.uniform"

    out = result_def(IntegerAttrTypeConstr)

    assembly_format = "attr-dict `:` type($out)"

    def __init__(self, out_type: IntegerType | IndexType):
        super().__init__(
            result_types=(out_type,),
        )


Prob = Dialect(
    "prob",
    [
        BernoulliOp,
        UniformOp,
    ],
    [],
)
