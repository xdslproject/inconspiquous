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
from inconspiquous.dialects.gate import (
    CXGate,
    CZGate,
    HadamardGate,
    XZOp,
    CliffordGateAttr,
)
from inconspiquous.dialects.measurement import (
    CompBasisMeasurementAttr,
    XYDynMeasurementOp,
    XYMeasurementAttr,
)
from inconspiquous.transforms.xzs.fusion import FuseXZGatesPattern
from inconspiquous.utils.linear_walker import LinearWalker


class XZCommutePattern(RewritePattern):
    """Commute an XZ gadget past a Hadamard/CX/CZ gate, or MeasureOp."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op1: qssa.DynGateOp, rewriter: PatternRewriter):
        initial_xz_op = op1.gate.owner
        if not isinstance(initial_xz_op, XZOp):
            return
        if len(op1.outs[0].uses) != 1:
            return

        (use,) = op1.outs[0].uses
        op2 = use.operation

        # Handle XY measurement (existing logic)
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
                angle_val = angle_op[0].out
            else:
                angle_op = ()
                angle_val = angle # type: ignore (angle is SSAValue if not AngleAttr)

            negate = CondNegateAngleOp(initial_xz_op.x, angle_val)
            new_measurement = XYDynMeasurementOp(negate.out)
            new_op2_measure = qssa.DynMeasureOp(op1.ins[0], measurement=new_measurement)
            new_op1_xor = arith.XOrIOp(new_op2_measure.outs[0], initial_xz_op.z)

            rewriter.replace_op(
                op2,
                (
                    *angle_op,
                    negate,
                    new_measurement,
                    new_op2_measure,
                    new_op1_xor,
                ),
            )
            rewriter.erase_op(op1)
            return

        # Handle computational basis measurement (existing logic)
        if isinstance(op2, qssa.MeasureOp):
            if not isinstance(op2.measurement, CompBasisMeasurementAttr):
                return # Not a computational basis measurement we can handle here
            new_op2_measure = qssa.MeasureOp(op1.ins[0])
            new_op1_xor = arith.XOrIOp(new_op2_measure.outs[0], initial_xz_op.x)

            rewriter.replace_op(op2, (new_op2_measure, new_op1_xor))
            rewriter.erase_op(op1)
            return

        if not isinstance(op2, qssa.GateOp):
            return
        
        clifford_gate_attr = op2.gate
        if not isinstance(clifford_gate_attr, CliffordGateAttr):
            return

        # At this point, op2 is a qssa.GateOp with a CliffordGateAttr

        c0 = arith.ConstantOp.from_int_and_width(0, rewriter.get_type(initial_xz_op.x)) # type: ignore
        
        # Create the new Clifford gate op, with op1's input qubit
        new_clifford_inputs = list(op2.ins)
        input_idx_on_clifford = use.index
        new_clifford_inputs[input_idx_on_clifford] = op1.ins[0]
        
        commuted_clifford_op = qssa.GateOp(clifford_gate_attr, *new_clifford_inputs)

        ops_to_insert = [c0, commuted_clifford_op]
        final_ssa_results = list(commuted_clifford_op.outs)

        for i, clifford_out_ssa in enumerate(commuted_clifford_op.outs):
            # Determine the X and Z components for the XZ gate after this output
            # Initialize with no Pauli (represented by c0)
            final_x_component_ssa = c0.result
            final_z_component_ssa = c0.result
            
            # Contribution from the original X part of initial_xz_op
            if initial_xz_op.x != c0.result:
                prop_x_rules = clifford_gate_attr.pauli_prop(input_idx_on_clifford, "X")
                if i < len(prop_x_rules):
                    propagates_to_x, propagates_to_z = prop_x_rules[i]
                    if propagates_to_x:
                        # If final_x_component_ssa is c0, it becomes initial_xz_op.x
                        # Otherwise, XOR with initial_xz_op.x
                        if final_x_component_ssa == c0.result:
                            final_x_component_ssa = initial_xz_op.x
                        else:
                            xor_op = arith.XOrIOp(final_x_component_ssa, initial_xz_op.x)
                            ops_to_insert.append(xor_op)
                            final_x_component_ssa = xor_op.result
                    if propagates_to_z:
                        if final_z_component_ssa == c0.result:
                            final_z_component_ssa = initial_xz_op.x
                        else:
                            xor_op = arith.XOrIOp(final_z_component_ssa, initial_xz_op.x)
                            ops_to_insert.append(xor_op)
                            final_z_component_ssa = xor_op.result
            
            # Contribution from the original Z part of initial_xz_op
            if initial_xz_op.z != c0.result:
                prop_z_rules = clifford_gate_attr.pauli_prop(input_idx_on_clifford, "Z")
                if i < len(prop_z_rules):
                    propagates_to_x, propagates_to_z = prop_z_rules[i]
                    if propagates_to_x:
                        if final_x_component_ssa == c0.result:
                            final_x_component_ssa = initial_xz_op.z
                        else:
                            xor_op = arith.XOrIOp(final_x_component_ssa, initial_xz_op.z)
                            ops_to_insert.append(xor_op)
                            final_x_component_ssa = xor_op.result
                    if propagates_to_z:
                        if final_z_component_ssa == c0.result:
                            final_z_component_ssa = initial_xz_op.z
                        else:
                            xor_op = arith.XOrIOp(final_z_component_ssa, initial_xz_op.z)
                            ops_to_insert.append(xor_op)
                            final_z_component_ssa = xor_op.result

            # If there's any Pauli component for this output, add the XZOp
            if final_x_component_ssa != c0.result or final_z_component_ssa != c0.result:
                propagated_xz_gate_op = XZOp(final_x_component_ssa, final_z_component_ssa)
                dyn_propagated_gate_op = qssa.DynGateOp(propagated_xz_gate_op, clifford_out_ssa)
                ops_to_insert.extend([propagated_xz_gate_op, dyn_propagated_gate_op])
                final_ssa_results[i] = dyn_propagated_gate_op.outs[0]
        
        rewriter.replace_op(op2, ops_to_insert, final_ssa_results)
        rewriter.erase_op(op1)


class XZCommute(ModulePass):
    name = "xz-commute"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        LinearWalker(
            GreedyRewritePatternApplier([FuseXZGatesPattern(), XZCommutePattern()])
        ).rewrite_module(op)
