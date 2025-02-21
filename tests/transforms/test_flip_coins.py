from xdsl.dialects import arith
from xdsl.parser import Parser
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import IntegerAttr

from inconspiquous.dialects.prob import BernoulliOp, Prob
from inconspiquous.transforms.flip_coins import FlipCoinsPass


def test_flip_coins_random():
    mod = ModuleOp([BernoulliOp(0.3) for _ in range(0, 1000)])
    # Check that there is some element of randomness
    p = FlipCoinsPass(1)
    seen_true = False
    seen_false = False
    p.apply(Context(), mod)
    for op in mod.body.ops:
        assert isinstance(op, arith.ConstantOp)
        assert isinstance(attr := op.value, IntegerAttr)
        if attr.value.data:
            seen_true = True
        else:
            seen_false = True

    assert seen_true, "Flip coins only produced false values"
    assert seen_false, "Flip coins only produced true values"


def test_flip_coins_independent():
    ctx = Context()
    ctx.register_dialect("prob", lambda: Prob)

    def gen_uniform() -> ModuleOp:
        parser = Parser(ctx, "%0 = prob.uniform i32")
        return parser.parse_module()

    p1 = FlipCoinsPass(20)

    prog1 = gen_uniform()
    p1.apply(ctx, prog1)
    prog2 = gen_uniform()
    p1.apply(ctx, prog2)

    assert prog1.body.is_structurally_equivalent(prog2.body)
