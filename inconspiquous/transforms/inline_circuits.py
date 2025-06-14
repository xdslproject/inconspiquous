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
    Inline a circuit when used with `qssa.dyn_gate`.

    Transforms:
    ```
    %circuit = qssa.circuit() ({
    ^bb0(%arg0: !qu.bit, %arg1: !qu.bit):
      %0 = qssa.gate<#gate.x> %arg0
      %1 = qssa.gate<#gate.y> %arg1
      qssa.return %0, %1
    }) : () -> !gate.type<2>
    %result0, %result1 = qssa.dyn_gate<%circuit> %in0, %in1
    ```

    Into:
    ```
    %circuit = qssa.circuit() ({
    ^bb0(%arg0: !qu.bit, %arg1: !qu.bit):
      %0 = qssa.gate<#gate.x> %arg0
      %1 = qssa.gate<#gate.y> %arg1
      qssa.return %0, %1
    }) : () -> !gate.type<2>
    %0 = qssa.gate<#gate.x> %in0
    %1 = qssa.gate<#gate.y> %in1
    ```
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter, /) -> None:
        """
        Inline a circuit when used with `qssa.dyn_gate`.

        This transforms:
          %result = qssa.dyn_gate<%circuit>(%inputs...)

        Into the inlined circuit body with proper SSA value mapping.
        """
        # Only process dyn_gate operations that use a circuit
        circuit_op = op.gate.owner
        if not isinstance(circuit_op, CircuitOp):
            return

        circuit_block = circuit_op.body.clone().blocks[0]

        # Get the circuit's return operation (terminators are always last)
        circuit_return = circuit_block.last_op
        if not isinstance(circuit_return, ReturnOp):
            return

        # Inline the entire circuit block before the dyn_gate operation
        # This automatically handles SSA value mapping from circuit args to dyn_gate inputs
        rewriter.inline_block(circuit_block, rewriter.insertion_point, op.ins)

        # Replace the dyn_gate with the values being returned by the circuit
        return_values = list(circuit_return.operands)
        rewriter.replace_op(op, (), return_values)

        # Remove the inlined return operation (it's not needed in the parent function)
        rewriter.erase_op(circuit_return)


class InlineCircuitsPass(ModulePass):
    """
    Pass that inlines all circuit operations used with dyn_gate.
    """

    name = "inline-circuits"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InlineCircuitPattern()).rewrite_module(op)