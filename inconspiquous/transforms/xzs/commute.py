from xdsl.ir import Operation, SSAValue
from xdsl.dialects import arith, builtin
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from inconspiquous.dialects import qssa
from inconspiquous.dialects.angle import AngleAttr, CondNegateAngleOp, ConstantAngleOp
from inconspiquous.dialects.gate import XZOp
from inconspiquous.dialects.measurement import (
    CompBasisMeasurementAttr,
    XYDynMeasurementOp,
    XYMeasurementAttr,
)
from inconspiquous.gates import CliffordGateAttr
from inconspiquous.transforms.xzs.fusion import FuseXZGatesPattern
from inconspiquous.utils.linear_walker import LinearWalker


class XZCommutePattern(RewritePattern):
    """Commute an XZ gadget past a Hadamard/CX/CZ gate, or MeasureOp."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op1: qssa.DynGateOp, rewriter: PatternRewriter):
        gate = op1.gate.owner
        if not isinstance(gate, XZOp):
            return
        if (use := op1.outs[0].get_unique_use()) is None:
            return

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
            new_op2 = qssa.DynMeasureOp(op1.ins[0], measurement=new_measurement)
            new_op1 = arith.XOrIOp(new_op2.outs[0], gate.z)

            rewriter.replace_op(
                op2, (*angle_op, negate, new_measurement, new_op2, new_op1)
            )
            rewriter.erase_op(op1)
            return

        if isinstance(op2, qssa.MeasureOp):
            if not isinstance(op2.measurement, CompBasisMeasurementAttr):
                return
            new_op2 = qssa.MeasureOp(op1.ins[0])
            new_op1 = arith.XOrIOp(new_op2.outs[0], gate.x)

            rewriter.replace_op(op2, (new_op2, new_op1))
            rewriter.erase_op(op1)
            return

        if not isinstance(op2, qssa.GateOp):
            return

        if isinstance(op2.gate, CliffordGateAttr):
            input_idx = use.index

            x_prop = op2.gate.pauli_prop(input_idx, "X")
            z_prop = op2.gate.pauli_prop(input_idx, "Z")

            new_operands = list(op2.ins)
            new_operands[input_idx] = op1.ins[0]

            new_op2 = qssa.GateOp(op2.gate, *new_operands)
            ops_to_insert: list[Operation] = []

            false_const = arith.ConstantOp.from_int_and_width(0, 1)
            false_const_needed = False
            new_outputs: list[SSAValue] = []

            for out_idx, (from_x, from_z) in enumerate(zip(x_prop, z_prop)):
                apply_x: SSAValue
                apply_z: SSAValue
                needs_gate = False

                if from_x.x and from_z.x:
                    xor_x = arith.XOrIOp(gate.x, gate.z)
                    apply_x = xor_x.result
                    ops_to_insert.append(xor_x)
                    needs_gate = True
                elif from_x.x:
                    apply_x = gate.x
                    needs_gate = True
                elif from_z.x:
                    apply_x = gate.z
                    needs_gate = True
                else:
                    apply_x = false_const.result
                    false_const_needed = True

                # Z component
                if from_x.z and from_z.z:
                    xor_z = arith.XOrIOp(gate.x, gate.z)
                    apply_z = xor_z.result
                    ops_to_insert.append(xor_z)
                    needs_gate = True
                elif from_x.z:
                    apply_z = gate.x
                    needs_gate = True
                elif from_z.z:
                    apply_z = gate.z
                    needs_gate = True
                else:
                    apply_z = false_const.result
                    false_const_needed = True

                # Create XZ gate for this output
                if needs_gate:
                    if apply_x == gate.x and apply_z == gate.z:
                        xz_gate = gate
                    else:
                        xz_gate = XZOp(apply_x, apply_z)
                        ops_to_insert.append(xz_gate)

                    dyn_gate = qssa.DynGateOp(xz_gate, new_op2.outs[out_idx])
                    ops_to_insert.append(dyn_gate)
                    new_outputs.append(dyn_gate.outs[0])
                else:
                    new_outputs.append(new_op2.outs[out_idx])

            final_ops = (
                (false_const, new_op2, *ops_to_insert)
                if false_const_needed
                else (new_op2, *ops_to_insert)
            )

            rewriter.replace_op(op2, final_ops, new_outputs)
            rewriter.erase_op(op1)
            return


class XZCommute(ModulePass):
    name = "xz-commute"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        LinearWalker(
            GreedyRewritePatternApplier(
                [FuseXZGatesPattern(), XZCommutePattern()], dce_enabled=False
            )
        ).rewrite_module(op)
