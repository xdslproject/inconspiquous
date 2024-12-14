from xdsl.dialects import builtin
from xdsl.dialects.arith import AddiOp, AndIOp
from xdsl.parser import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from inconspiquous.dialects.gate import XZSOp
from inconspiquous.dialects.qssa import DynGateOp


class MergeXZSGatesPattern(RewritePattern):
    """
    Merge two consecutive XZS gadgets
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        gate2 = op.gate.owner
        if not isinstance(gate2, XZSOp):
            return

        predecessor = op.ins[0].owner

        if not isinstance(predecessor, DynGateOp):
            return

        gate1 = predecessor.gate.owner
        if not isinstance(gate1, XZSOp):
            return

        new_x = AddiOp(gate1.x, gate2.x)
        new_z_and = AndIOp(gate2.x, gate1.phase)
        new_z_add = AddiOp(gate1.z, gate2.z)
        new_z = AddiOp(new_z_and, new_z_add)
        new_phase = AddiOp(gate1.phase, gate2.phase)
        new_gate = XZSOp(new_x, new_z, new_phase)

        new_gate.out.name_hint = gate1.out.name_hint

        rewriter.insert_op(
            (
                new_x,
                new_z_and,
                new_z_add,
                new_z,
                new_phase,
                new_gate,
            ),
            InsertPoint.after(gate2),
        )

        rewriter.replace_matched_op(DynGateOp(new_gate, *predecessor.ins))
        rewriter.erase_op(predecessor)


class MergeXZSGates(ModulePass):
    """
    Merge consecutive XZS gadgets and push arith.select inwards
    """

    name = "merge-xzs"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(MergeXZSGatesPattern()).rewrite_module(op)
