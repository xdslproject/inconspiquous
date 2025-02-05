from xdsl.dialects import arith, builtin
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from inconspiquous.dialects import qssa
from inconspiquous.dialects.angle import AngleAttr, CondNegateAngleOp, ConstantAngleOp
from inconspiquous.dialects.gate import (
    CNotGate,
    CZGate,
    HadamardGate,
    XZOp,
)
from inconspiquous.dialects.measurement import (
    CompBasisMeasurementAttr,
    XYDynMeasurementOp,
    XYMeasurementAttr,
)
from inconspiquous.transforms.xzs.merge import MergeXZGatesPattern
from inconspiquous.utils.linear_walker import LinearWalker


class XZCommutePattern(RewritePattern):
    """Commute an XZ gadget past a Hadamard/CNot/CZ gate, or MeasureOp."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op1: qssa.DynGateOp, rewriter: PatternRewriter):
        gate = op1.gate.owner
        if not isinstance(gate, XZOp):
            return
        if len(op1.outs[0].uses) != 1:
            return

        (use,) = op1.outs[0].uses

        op2 = use.operation

        # Check for XY measurement
        angle = None
        if isinstance(op2, qssa.DynMeasureOp) and isinstance(
            op2.measurement.owner, XYDynMeasurementOp
        ):
            angle = op2.measurement.owner.angle

        if isinstance(op2, qssa.MeasureOp) and isinstance(
            op2.measurement, XYMeasurementAttr
        ):
            angle = op2.measurement.angle

        if angle is not None:
            if isinstance(angle, AngleAttr):
                angle_op = (ConstantAngleOp(angle),)
                angle = angle_op[0].out
            else:
                angle_op = ()

            negate = CondNegateAngleOp(gate.x, angle)
            new_measurement = XYDynMeasurementOp(negate)
            new_op2 = qssa.DynMeasureOp(new_measurement, op1.ins[0])
            new_op1 = arith.AddiOp(new_op2.outs[0], gate.z)

            rewriter.replace_op(op2, (*angle_op, negate, new_measurement, new_op2, new_op1))
            rewriter.erase_op(op1)

        if isinstance(op2, qssa.MeasureOp):
            if not isinstance(op2.measurement, CompBasisMeasurementAttr):
                return
            new_op2 = qssa.MeasureOp(op1.ins[0])
            new_op1 = arith.AddiOp(new_op2.out[0], gate.x)

            rewriter.replace_op(op2, (new_op2, new_op1))
            rewriter.erase_op(op1)

        if not isinstance(op2, qssa.GateOp):
            return

        if isinstance(op2.gate, HadamardGate):
            new_op2 = qssa.GateOp(HadamardGate(), *op1.ins)
            new_gate = XZOp(gate.z, gate.x)
            new_op1 = qssa.DynGateOp(new_gate, *new_op2.outs)

            rewriter.replace_op(op2, (new_op2, new_gate, new_op1))
            rewriter.erase_op(op1)

        if isinstance(op2.gate, CNotGate):
            c0 = arith.ConstantOp.from_int_and_width(0, 1)
            if use.index == 0:
                new_op2 = qssa.GateOp(CNotGate(), *(op1.ins[0], op2.ins[1]))
                new_op1_left = qssa.DynGateOp(gate, new_op2.outs[0])
                new_gate_right = XZOp(gate.x, c0)
                new_op1_right = qssa.DynGateOp(new_gate_right, new_op2.outs[1])
                rewriter.replace_op(
                    op2,
                    (c0, new_op2, new_op1_left, new_gate_right, new_op1_right),
                    (new_op1_left.outs[0], new_op1_right.outs[0]),
                )
                rewriter.erase_op(op1)
            elif use.index == 1:
                new_op2 = qssa.GateOp(CNotGate(), *(op2.ins[0], op1.ins[0]))
                new_gate_left = XZOp(c0, gate.z)
                new_op1_left = qssa.DynGateOp(new_gate_left, new_op2.outs[0])
                new_op1_right = qssa.DynGateOp(gate, new_op2.outs[1])
                rewriter.replace_op(
                    op2,
                    (c0, new_op2, new_gate_left, new_op1_left, new_op1_right),
                    (new_op1_left.outs[0], new_op1_right.outs[0]),
                )
                rewriter.erase_op(op1)

        if isinstance(op2.gate, CZGate):
            c0 = arith.ConstantOp.from_int_and_width(0, 1)
            if use.index == 0:
                new_op2 = qssa.GateOp(CZGate(), *(op1.ins[0], op2.ins[1]))
                new_op1_left = qssa.DynGateOp(gate, new_op2.outs[0])
                new_gate_right = XZOp(c0, gate.x)
                new_op1_right = qssa.DynGateOp(new_gate_right, new_op2.outs[1])
                rewriter.replace_op(
                    op2,
                    (c0, new_op2, new_op1_left, new_gate_right, new_op1_right),
                    (new_op1_left.outs[0], new_op1_right.outs[0]),
                )
                rewriter.erase_op(op1)
            elif use.index == 1:
                new_op2 = qssa.GateOp(CZGate(), *(op2.ins[0], op1.ins[0]))
                new_gate_left = XZOp(c0, gate.x)
                new_op1_left = qssa.DynGateOp(new_gate_left, new_op2.outs[0])
                new_op1_right = qssa.DynGateOp(gate, new_op2.outs[1])
                rewriter.replace_op(
                    op2,
                    (c0, new_op2, new_gate_left, new_op1_left, new_op1_right),
                    (new_op1_left.outs[0], new_op1_right.outs[0]),
                )
                rewriter.erase_op(op1)


class XZCommute(ModulePass):
    name = "xz-commute"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        LinearWalker(
            GreedyRewritePatternApplier([MergeXZGatesPattern(), XZCommutePattern()])
        ).rewrite_module(op)
