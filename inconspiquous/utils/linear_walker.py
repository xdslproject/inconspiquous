from xdsl.ir import Block, Operation, Region
from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriterListener,
    RewritePattern,
)


class LinearWalker:
    """
    For each block in the IR, this runner walks through the operations, continously
    applying the supplied pattern.

    It functions by maintaining a single pointer to an operation and performs the
    following:
    - Run the supplied pattern on the current operation
    - If the pattern does no action, then move the pointer forward one
    - If the pattern deletes the current operation, then backtrack by one operation
    We repeat this until we reach the end of the block.
    """

    pattern: RewritePattern
    """
    The pattern to be run.
    """

    _current_operation: Operation | None
    """
    Pointer to the current operation
    """

    _listener: PatternRewriterListener

    def __init__(self, pattern: RewritePattern) -> None:
        self.pattern = pattern
        self._current_operation = None
        self._listener = PatternRewriterListener(
            operation_removal_handler=[
                self._handle_operation_removal,
            ]
        )

    def _handle_operation_removal(self, op: Operation):
        if op == self._current_operation:
            assert self._current_operation is not None
            if (prev := self._current_operation.prev_op) is None:
                block = self._current_operation.parent_block()
                assert block is not None
                self._current_operation = block.first_op
            else:
                self._current_operation = prev

    def rewrite_module(self, module: ModuleOp) -> bool:
        return self.rewrite_region(module.body)

    def rewrite_region(self, region: Region) -> bool:
        modified = False
        for block in region.blocks:
            modified |= self.rewrite_block(block)

        return modified

    def rewrite_block(self, block: Block) -> bool:
        modified = False
        for op in block.ops:
            for region in op.regions:
                modified |= self.rewrite_region(region)

        self._current_operation = block.first_op

        while self._current_operation is not None:
            rewriter = PatternRewriter(self._current_operation)
            rewriter.extend_from_listener(self._listener)
            self.pattern.match_and_rewrite(self._current_operation, rewriter)

            if rewriter.has_done_action:
                modified = True
            else:
                self._current_operation = self._current_operation.next_op

        return modified
