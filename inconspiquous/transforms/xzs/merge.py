from xdsl.dialects import builtin
from xdsl.dialects import arith
from xdsl.dialects.arith import AddiOp, AndIOp
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from inconspiquous.dialects.gate import XZOp, XZSOp
from inconspiquous.dialects.qssa import DynGateOp


class MergeXZGatesPattern(RewritePattern):
    """
    Merge two consecutive XZ gadgets
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        gate2 = op.gate.owner
        if not isinstance(gate2, XZOp):
            return

        if len(op.ins[0].uses) != 1:
            return

        predecessor = op.ins[0].owner

        if not isinstance(predecessor, DynGateOp):
            return

        gate1 = predecessor.gate.owner
        if not isinstance(gate1, XZOp):
            return

        new_x = AddiOp(gate1.x, gate2.x)
        new_z = AddiOp(gate1.z, gate2.z)
        new_gate = XZOp(new_x, new_z)

        new_gate.out.name_hint = gate1.out.name_hint

        rewriter.insert_op(
            (
                new_x,
                new_z,
                new_gate,
            ),
            InsertPoint.before(op),
        )

        rewriter.replace_matched_op(DynGateOp(new_gate, *predecessor.ins))
        rewriter.erase_op(predecessor)


class MergeXZSGatesPattern(RewritePattern):
    """
    Merge two consecutive XZS gadgets
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        gate2 = op.gate.owner
        if not isinstance(gate2, XZSOp | XZOp):
            return

        if len(op.ins[0].uses) != 1:
            return

        predecessor = op.ins[0].owner

        if not isinstance(predecessor, DynGateOp):
            return

        gate1 = predecessor.gate.owner
        if not isinstance(gate1, XZSOp | XZOp):
            return

        c0 = None
        if isinstance(gate1, XZSOp):
            gate1_phase = gate1.phase
        else:
            c0 = arith.ConstantOp.from_int_and_width(0, 1)
            gate1_phase = c0

        if isinstance(gate2, XZSOp):
            gate2_phase = gate2.phase
        else:
            if c0 is None:
                c0 = arith.ConstantOp.from_int_and_width(0, 1)
            gate2_phase = c0

        new_x = AddiOp(gate1.x, gate2.x)
        new_z_and = AndIOp(gate2.x, gate1_phase)
        new_z_add = AddiOp(gate1.z, gate2.z)
        new_z = AddiOp(new_z_and, new_z_add)
        new_phase = AddiOp(gate1_phase, gate2_phase)
        new_gate = XZSOp(new_x, new_z, new_phase)

        new_gate.out.name_hint = gate1.out.name_hint

        rewriter.insert_op(
            (
                *((c0,) if c0 is not None else ()),
                new_x,
                new_z_and,
                new_z_add,
                new_z,
                new_phase,
                new_gate,
            ),
            InsertPoint.before(op),
        )

        rewriter.replace_matched_op(DynGateOp(new_gate, *predecessor.ins))
        rewriter.erase_op(predecessor)


class XZSMerge(ModulePass):
    """
    Merge consecutive XZS gadgets.
    """

    name = "xzs-merge"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([MergeXZGatesPattern(), MergeXZSGatesPattern()])
        ).rewrite_module(op)
