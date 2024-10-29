from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.varith import VarithSwitchOp
from xdsl.parser import DenseIntOrFPElementsAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerType

from inconspiquous.dialects.gate import (
    ComposeGateOp,
    ConstantGateOp,
    HadamardGate,
    IdentityGate,
    PhaseGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from inconspiquous.dialects.qssa import DynGateOp, GateOp
from inconspiquous.dialects.prob import UniformOp


class PadTGate(RewritePattern):
    """
    Places randomized dynamic pauli gates before and after a T gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, TGate):
            return
        rand = UniformOp(IntegerType(2))
        id_gate = ConstantGateOp(IdentityGate())
        x_gate = ConstantGateOp(XGate())
        y_gate = ConstantGateOp(YGate())
        z_gate = ConstantGateOp(ZGate())
        phase_gate = ConstantGateOp(PhaseGate())
        cases = DenseIntOrFPElementsAttr.vector_from_list([0, 1, 2], IntegerType(2))
        pre_choice = VarithSwitchOp(rand, cases, id_gate, x_gate, y_gate, z_gate)
        pre_gate = DynGateOp(pre_choice, *op.ins)

        new_t = GateOp(TGate(), pre_gate)

        x_case = ComposeGateOp(phase_gate, y_gate)
        y_case = ComposeGateOp(y_gate, phase_gate)
        post_choice = VarithSwitchOp(rand, cases, id_gate, x_case, y_case, z_gate)
        post_gate = DynGateOp(post_choice, new_t)

        rewriter.replace_matched_op(
            (
                rand,
                id_gate,
                x_gate,
                y_gate,
                z_gate,
                phase_gate,
                pre_choice,
                pre_gate,
                new_t,
                x_case,
                y_case,
                post_choice,
                post_gate,
            ),
            post_gate.outs,
        )


class PadHadamardGate(RewritePattern):
    """
    Places randomized dynamic pauli gates before and after a hadamard gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, HadamardGate):
            return
        rand = UniformOp(IntegerType(2))
        id_gate = ConstantGateOp(IdentityGate())
        x_gate = ConstantGateOp(XGate())
        y_gate = ConstantGateOp(YGate())
        z_gate = ConstantGateOp(ZGate())
        phase_gate = ConstantGateOp(PhaseGate())
        cases = DenseIntOrFPElementsAttr.vector_from_list([0, 1, 2], IntegerType(2))
        pre_choice = VarithSwitchOp(rand, cases, id_gate, x_gate, y_gate, z_gate)
        pre_gate = DynGateOp(pre_choice, *op.ins)

        new_hadamard = GateOp(HadamardGate(), pre_gate)

        post_choice = VarithSwitchOp(rand, cases, id_gate, z_gate, y_gate, x_gate)
        post_gate = DynGateOp(post_choice, new_hadamard)

        rewriter.replace_matched_op(
            (
                rand,
                id_gate,
                x_gate,
                y_gate,
                z_gate,
                phase_gate,
                pre_choice,
                pre_gate,
                new_hadamard,
                post_choice,
                post_gate,
            ),
            post_gate.outs,
        )


# class PadCNotGate(RewritePattern):
#     """
#     Places randomized dynamic pauli gates before and after a cnot gate.
#     """

#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
#         if not isinstance(op.gate, CNotGate):
#             return
#         rand = UniformOp(IntegerType(2))
#         id_gate = ConstantGateOp(IdentityGate())
#         x_gate = ConstantGateOp(XGate())
#         y_gate = ConstantGateOp(YGate())
#         z_gate = ConstantGateOp(ZGate())
#         phase_gate = ConstantGateOp(PhaseGate())
#         cases = DenseIntOrFPElementsAttr.vector_from_list([0,1,2], IntegerType(2))
#         pre_choice = VarithSwitchOp(rand, cases, id_gate, x_gate, y_gate, z_gate)
#         pre_gate = DynGateOp(pre_choice, *op.ins)

#         new_t = GateOp(TGate(), pre_gate)

#         x_case = ComposeGateOp(phase_gate, y_gate)
#         y_case = ComposeGateOp(y_gate, phase_gate)
#         post_choice = VarithSwitchOp(rand, cases, id_gate, x_case, y_case, z_gate)
#         post_gate = DynGateOp(post_choice, new_t)

#         rewriter.replace_matched_op((rand, id_gate, x_gate, y_gate, z_gate, phase_gate, pre_choice, pre_gate, new_t, x_case, y_case, post_choice, post_gate), post_gate.outs)


class RandomizedComp(ModulePass):
    """
    Pads all "difficult" gates in a circuit.
    """

    name = "randomized-comp"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([PadTGate(), PadHadamardGate()]),
            apply_recursively=False,  # Do not reapply
        ).rewrite_op(op)
