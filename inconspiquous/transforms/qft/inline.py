from xdsl.builder import ImplicitBuilder
from xdsl.dialects import builtin
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from inconspiquous.dialects.qft import QFTAttr, HadamardGate, nthrootGate
from inconspiquous.dialects.qssa import GateOp
from inconspiquous.utils.qssa_builder import QSSABuilder, QubitRef


class StandardQFTInliner(RewritePattern):
    """
    Replaces the Perfect code gate with its definition
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, QFTAttr):
            return

        qubits = tuple(QubitRef(qubit) for qubit in op.ins)
        num_qubits = len(qubits)

        builder = QSSABuilder(InsertPoint.before(op))

        with ImplicitBuilder(builder):
            for i in range(num_qubits):
                builder.gate(HadamardGate(), qubits[i])
                for j in range(i + 1, num_qubits):
                    builder.gate(nthrootGate(j - i), qubits[i], qubits[j])

        rewriter.replace_matched_op((), tuple([q.qubit for q in qubits]))


class StandardQFTInlinerPass(ModulePass):
    name = "qft-std-inline"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            StandardQFTInliner(), apply_recursively=False
        ).rewrite_module(op)
