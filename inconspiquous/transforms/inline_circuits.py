"""
Transformation to inline subcircuits when a qssa.dyn_gate is called with a circuit value.
"""

from xdsl.dialects import builtin
from xdsl.ir import Operation, SSAValue
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects.qssa import DynGateOp, CircuitOp, ReturnOp


class InlineCircuitPattern(RewritePattern):
    """
    Inline a circuit when used with qssa.dyn_gate.

    Transforms:
    ```
    %circuit = qssa.circuit() {
      ^bb0(%arg0: !qu.bit, %arg1: !qu.bit):
        %0 = qssa.gate<X> %arg0
        %1 = qssa.gate<Y> %arg1
        qssa.return %0, %1
    }
    %result:2 = qssa.dyn_gate %circuit(%in0, %in1)
    ```

    Into:
    ```
    %0 = qssa.gate<X> %in0
    %1 = qssa.gate<Y> %in1
    ```
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter, /) -> None:
        # Check if the gate operand is a circuit
        gate_op = op.gate.owner
        if not isinstance(gate_op, CircuitOp):
            return

        # Get the circuit body
        circuit_body = gate_op.body
        if len(circuit_body.blocks) != 1:
            return

        entry_block = circuit_body.blocks[0]

        # Validate that the circuit is properly terminated
        ops_list = list(entry_block.ops)
        if not ops_list or not isinstance(ops_list[-1], ReturnOp):
            return

        # Get the return operation
        return_op = ops_list[-1]

        # Create a mapping from circuit block arguments to dyn_gate inputs
        value_mapping: dict[SSAValue, SSAValue] = {}
        for circuit_arg, dyn_gate_input in zip(entry_block.args, op.ins):
            value_mapping[circuit_arg] = dyn_gate_input

        # Clone all operations except the return and remap their operands
        replacement_ops: list[Operation] = []
        for circuit_op in ops_list[:-1]:  # Exclude return
            cloned_op = circuit_op.clone()

            # Remap operands in the cloned operation
            remapped_operands: list[SSAValue] = []
            for operand in cloned_op.operands:
                mapped_operand: SSAValue = value_mapping.get(operand, operand)
                remapped_operands.append(mapped_operand)
            cloned_op.operands = remapped_operands

            # Update value mapping for results
            for old_result, new_result in zip(circuit_op.results, cloned_op.results):
                value_mapping[old_result] = new_result

            replacement_ops.append(cloned_op)

        # Map return values to replacement values
        return_values: list[SSAValue] = []
        for return_operand in return_op.args:
            mapped_return: SSAValue = value_mapping.get(return_operand, return_operand)
            return_values.append(mapped_return)

        # Replace the dyn_gate with the cloned operations and return values
        rewriter.replace_matched_op(tuple(replacement_ops), return_values)


class InlineCircuitsPass(ModulePass):
    """
    Pass that inlines all circuit operations used with dyn_gate.
    """

    name = "inline-circuits"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InlineCircuitPattern()).rewrite_module(op)
