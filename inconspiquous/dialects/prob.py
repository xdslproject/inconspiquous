from typing import ClassVar, Sequence
from typing_extensions import Self

from xdsl.dialects.builtin import IntegerAttrTypeConstr, i1
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
    VerifyException,
)
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import (
    DenseArrayBase,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerType,
    UnresolvedOperand,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
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


class FinSuppOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from inconspiquous.transforms.canonicalization.prob import (
            FinSuppTrivial,
            FinSuppRemoveCase,
            FinSuppDuplicate,
        )

        return (FinSuppTrivial(), FinSuppRemoveCase(), FinSuppDuplicate())


@irdl_op_definition
class FinSuppOp(IRDLOperation):
    name = "prob.fin_supp"

    _T: ClassVar = VarConstraint("T", AnyAttr())

    ins = var_operand_def(_T)

    default_value = operand_def(_T)

    out = result_def(_T)

    probabilities = prop_def(DenseArrayBase)

    traits = traits_def(FinSuppOpHasCanonicalizationPatterns())

    def __init__(
        self,
        probabilities: Sequence[float] | DenseArrayBase,
        default_value: SSAValue,
        *ins: SSAValue | Operation,
        attr_dict: dict[str, Attribute] | None = None,
    ):
        result_type = SSAValue.get(default_value).type
        if not isinstance(probabilities, DenseArrayBase):
            probabilities = DenseArrayBase.create_dense_float(
                Float64Type(), probabilities
            )
        super().__init__(
            operands=(ins, default_value),
            result_types=(result_type,),
            properties={"probabilities": probabilities},
            attributes=attr_dict,
        )

    @staticmethod
    def parse_case(parser: Parser) -> tuple[UnresolvedOperand, float]:
        prob = parser.parse_number()
        assert isinstance(prob, float)
        parser.parse_keyword("or")
        operand = parser.parse_unresolved_operand()
        return (operand, prob)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_punctuation("[")
        probabilities: list[float] = []
        cases: list[UnresolvedOperand] = []
        while (n := parser.parse_optional_number()) is not None:
            assert isinstance(n, float)
            probabilities.append(n)
            parser.parse_keyword("of")
            cases.append(parser.parse_unresolved_operand())
            parser.parse_punctuation(",")
        if cases:
            parser.parse_keyword("else")
        default_unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation("]")
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        ins = tuple(parser.resolve_operand(x, result_type) for x in cases)
        default_value = parser.resolve_operand(default_unresolved, result_type)
        attr_dict = parser.parse_optional_attr_dict()
        return cls(probabilities, default_value, *ins, attr_dict=attr_dict)

    @staticmethod
    def print_case(c: tuple[SSAValue, int | float], printer: Printer):
        operand, prob = c
        printer.print_string(repr(prob) + " of ")
        printer.print_operand(operand)

    def print(self, printer: Printer):
        printer.print_string(" [ ")
        printer.print_list(
            zip(self.ins, self.probabilities.as_tuple()),
            lambda c: self.print_case(c, printer),
        )
        if self.ins:
            printer.print_string(", else ")
        printer.print_operand(self.default_value)
        printer.print_string(" ] : ")
        printer.print_attribute(self.out.type)


Prob = Dialect(
    "prob",
    [
        BernoulliOp,
        UniformOp,
        FinSuppOp,
    ],
    [],
)
