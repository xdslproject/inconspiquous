"""
Transformation to inline subcircuits when a qssa.dyn_gate is called with a circuit value.
"""

from xdsl.dialects import builtin
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
        """
        Inline a circuit when used with qssa.dyn_gate.

        This transforms:
          %result = qssa.dyn_gate<%circuit>(%inputs...)

        Into the inlined circuit body with proper SSA value mapping.
        """
        # Only process dyn_gate operations that use a circuit
        circuit_op = op.gate.owner
        if not isinstance(circuit_op, CircuitOp):
            return

        # Skip empty circuits
        if not circuit_op.body.blocks:
            return

        circuit_block = circuit_op.body.blocks[0]

        # Find the circuit's return operation (there should be exactly one)
        circuit_return = None
        for circuit_op_in_block in circuit_block.ops:
            if isinstance(circuit_op_in_block, ReturnOp):
                circuit_return = circuit_op_in_block
                break

        if circuit_return is None:
            return

        # Inline the entire circuit block before the dyn_gate operation
        # This automatically handles SSA value mapping from circuit args to dyn_gate inputs
        rewriter.inline_block(circuit_block, rewriter.insertion_point, op.ins)

        # After inlining, the return operation is now in the parent block right before dyn_gate
        # Find it by searching backward from the dyn_gate position
        inlined_return = None
        parent_block = op.parent
        if parent_block is not None:
            ops_list = list(parent_block.ops)
            try:
                dyn_gate_index = ops_list.index(op)
                # Look backward from dyn_gate to find the most recent return
                for i in range(dyn_gate_index - 1, -1, -1):
                    if isinstance(ops_list[i], ReturnOp):
                        inlined_return = ops_list[i]
                        break
            except ValueError:
                pass

        if inlined_return is not None:
            # Replace the dyn_gate with the values being returned by the circuit
            return_values = (
                list(inlined_return.operands)
                if hasattr(inlined_return, "operands")
                else []
            )
            rewriter.replace_op(op, (), return_values)
            # Remove the inlined return operation (it's not needed in the parent function)
            rewriter.erase_op(inlined_return)
        else:
            # Fallback: replace with empty results if no return found
            rewriter.replace_op(op, (), [])

        # Clean up the original circuit operation
        rewriter.erase_op(circuit_op)


class InlineCircuitsPass(ModulePass):
    """
    Pass that inlines all circuit operations used with dyn_gate.
    """

    name = "inline-circuits"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InlineCircuitPattern()).rewrite_module(op)
