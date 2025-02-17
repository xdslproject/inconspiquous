from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.arith import AddiOp, SelectOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import i1
from xdsl.rewriter import InsertPoint

from inconspiquous.dialects.gate import (
    CNotGate,
    ConstantGateOp,
    HadamardGate,
    IdentityGate,
    PhaseDaggerGate,
    PhaseGate,
    TDaggerGate,
    TGate,
    XGate,
    ZGate,
)
from inconspiquous.dialects.measurement import CompBasisMeasurementAttr
from inconspiquous.dialects.qssa import DynGateOp, GateOp, MeasureOp
from inconspiquous.dialects.prob import UniformOp


class PadTGate(RewritePattern):
    """
    Places randomized dynamic pauli gates before and after a T gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, TGate):
            return
        x_rand = UniformOp(i1)
        z_rand = UniformOp(i1)
        id_gate = ConstantGateOp(IdentityGate())
        x_gate = ConstantGateOp(XGate())
        z_gate = ConstantGateOp(ZGate())
        phase_gate = ConstantGateOp(PhaseGate())
        pre_x_sel = SelectOp(x_rand, x_gate, id_gate)
        pre_x = DynGateOp(pre_x_sel, *op.ins)
        pre_z_sel = SelectOp(z_rand, z_gate, id_gate)
        pre_z = DynGateOp(pre_z_sel, pre_x)

        new_t = GateOp(op.gate, pre_z)

        post_z_sel = SelectOp(z_rand, z_gate, id_gate)
        post_z = DynGateOp(post_z_sel, new_t)

        post_x_1 = DynGateOp(pre_x_sel, post_z)
        post_x_sel_2 = SelectOp(x_rand, phase_gate, id_gate)
        post_x_2 = DynGateOp(post_x_sel_2, post_x_1)

        rewriter.insert_op(
            (
                x_rand,
                z_rand,
                id_gate,
                x_gate,
                z_gate,
                phase_gate,
                pre_x_sel,
                pre_x,
                pre_z_sel,
                pre_z,
                new_t,
                post_z_sel,
                post_z,
                post_x_1,
                post_x_sel_2,
            ),
            InsertPoint.before(op),
        )

        rewriter.replace_matched_op(post_x_2)


class PadTDaggerGate(RewritePattern):
    """
    Places randomized dynamic pauli gates before and after a TDagger gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, TDaggerGate):
            return
        x_rand = UniformOp(i1)
        z_rand = UniformOp(i1)
        id_gate = ConstantGateOp(IdentityGate())
        x_gate = ConstantGateOp(XGate())
        z_gate = ConstantGateOp(ZGate())
        phase_dagger_gate = ConstantGateOp(PhaseDaggerGate())
        pre_x_sel = SelectOp(x_rand, x_gate, id_gate)
        pre_x = DynGateOp(pre_x_sel, *op.ins)
        pre_z_sel = SelectOp(z_rand, z_gate, id_gate)
        pre_z = DynGateOp(pre_z_sel, pre_x)

        new_t = GateOp(op.gate, pre_z)

        post_z_sel = SelectOp(z_rand, z_gate, id_gate)
        post_z = DynGateOp(post_z_sel, new_t)

        post_x_1 = DynGateOp(pre_x_sel, post_z)
        post_x_sel_2 = SelectOp(x_rand, phase_dagger_gate, id_gate)
        post_x_2 = DynGateOp(post_x_sel_2, post_x_1)

        rewriter.insert_op(
            (
                x_rand,
                z_rand,
                id_gate,
                x_gate,
                z_gate,
                phase_dagger_gate,
                pre_x_sel,
                pre_x,
                pre_z_sel,
                pre_z,
                new_t,
                post_z_sel,
                post_z,
                post_x_1,
                post_x_sel_2,
            ),
            InsertPoint.before(op),
        )

        rewriter.replace_matched_op(post_x_2)


class PadHadamardGate(RewritePattern):
    """
    Places randomized dynamic pauli gates before and after a hadamard gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, HadamardGate):
            return
        x_rand = UniformOp(i1)
        z_rand = UniformOp(i1)
        id_gate = ConstantGateOp(IdentityGate())
        x_gate = ConstantGateOp(XGate())
        z_gate = ConstantGateOp(ZGate())
        pre_x_sel = SelectOp(x_rand, x_gate, id_gate)
        pre_x = DynGateOp(pre_x_sel, *op.ins)
        pre_z_sel = SelectOp(z_rand, z_gate, id_gate)
        pre_z = DynGateOp(pre_z_sel, pre_x)

        new_hadamard = GateOp(HadamardGate(), pre_z)

        post_z_sel = SelectOp(z_rand, x_gate, id_gate)
        post_z = DynGateOp(post_z_sel, new_hadamard)
        post_x_sel = SelectOp(x_rand, z_gate, id_gate)
        post_x = DynGateOp(post_x_sel, post_z)

        rewriter.insert_op(
            (
                x_rand,
                z_rand,
                id_gate,
                x_gate,
                z_gate,
                pre_x_sel,
                pre_x,
                pre_z_sel,
                pre_z,
                new_hadamard,
                post_z_sel,
                post_z,
                post_x_sel,
            ),
            InsertPoint.before(op),
        )

        rewriter.replace_matched_op(post_x)


class PadCNotGate(RewritePattern):
    """
    Places randomized dynamic pauli gates before and after a cnot gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, CNotGate):
            return
        x_rand_q1 = UniformOp(i1)
        x_rand_q2 = UniformOp(i1)
        z_rand_q1 = UniformOp(i1)
        z_rand_q2 = UniformOp(i1)

        id_gate = ConstantGateOp(IdentityGate())
        x_gate = ConstantGateOp(XGate())
        z_gate = ConstantGateOp(ZGate())

        x_sel_q1 = SelectOp(x_rand_q1, x_gate, id_gate)
        x_sel_q2 = SelectOp(x_rand_q2, x_gate, id_gate)

        z_sel_q1 = SelectOp(z_rand_q1, z_gate, id_gate)
        z_sel_q2 = SelectOp(z_rand_q2, z_gate, id_gate)

        pre_x_q1 = DynGateOp(x_sel_q1, op.ins[0])
        pre_z_q1 = DynGateOp(z_sel_q1, pre_x_q1)
        pre_x_q2 = DynGateOp(x_sel_q2, op.ins[1])
        pre_z_q2 = DynGateOp(z_sel_q2, pre_x_q2)

        new_cnot = GateOp(CNotGate(), pre_z_q1, pre_z_q2)

        post_z_q1_1 = DynGateOp(z_sel_q1, new_cnot.outs[0])
        post_z_q1_2 = DynGateOp(z_sel_q2, post_z_q1_1)
        post_x_q1 = DynGateOp(x_sel_q1, post_z_q1_2)

        post_z_q2 = DynGateOp(z_sel_q2, new_cnot.outs[1])
        post_x_q2_1 = DynGateOp(x_sel_q1, post_z_q2)
        post_x_q2_2 = DynGateOp(x_sel_q2, post_x_q2_1)

        rewriter.insert_op(
            (
                x_rand_q1,
                x_rand_q2,
                z_rand_q1,
                z_rand_q2,
                id_gate,
                x_gate,
                z_gate,
                x_sel_q1,
                x_sel_q2,
                z_sel_q1,
                z_sel_q2,
                pre_x_q1,
                pre_z_q1,
                pre_x_q2,
                pre_z_q2,
                new_cnot,
                post_z_q1_1,
                post_z_q1_2,
                post_z_q2,
                post_x_q2_1,
            ),
            InsertPoint.before(op),
        )

        rewriter.replace_matched_op(
            (post_x_q1, post_x_q2_2), (post_x_q1.outs[0], post_x_q2_2.outs[0])
        )


class PadMeasure(RewritePattern):
    """
    Places randomized dynamic pauli gates before a measurement.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MeasureOp, rewriter: PatternRewriter):
        if op.measurement != CompBasisMeasurementAttr():
            # Only try to pad computation basis measurements
            return
        x_rand = UniformOp(i1)
        z_rand = UniformOp(i1)

        id_gate = ConstantGateOp(IdentityGate())
        x_gate = ConstantGateOp(XGate())
        z_gate = ConstantGateOp(ZGate())

        pre_x_sel = SelectOp(x_rand, x_gate, id_gate)
        pre_x = DynGateOp(pre_x_sel, *op.in_qubits)
        pre_z_sel = SelectOp(z_rand, z_gate, id_gate)
        pre_z = DynGateOp(pre_z_sel, pre_x)

        new_measure = MeasureOp(pre_z)

        corrected_measure = AddiOp(x_rand, new_measure.out[0])

        rewriter.insert_op(
            (
                x_rand,
                z_rand,
                id_gate,
                x_gate,
                z_gate,
                pre_x_sel,
                pre_x,
                pre_z_sel,
                pre_z,
            ),
            InsertPoint.before(op),
        )

        rewriter.replace_matched_op(
            (new_measure, corrected_measure),
        )


class RandomizedComp(ModulePass):
    """
    Pads all "difficult" gates in a circuit.
    """

    name = "randomized-comp"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PadTGate(),
                    PadTDaggerGate(),
                    PadHadamardGate(),
                    PadCNotGate(),
                    PadMeasure(),
                ]
            ),
            apply_recursively=False,  # Do not reapply
        ).rewrite_module(op)
