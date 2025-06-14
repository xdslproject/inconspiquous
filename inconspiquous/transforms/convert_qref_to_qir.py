from typing import List
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects import qref, qir, qu
from inconspiquous.dialects.gate import (
    HadamardGate,
    XGate,
    YGate,
    ZGate,
    PhaseGate,
    TGate,
    CXGate,
    CZGate,
)
from inconspiquous.dialects.measurement import CompBasisMeasurementAttr


class ConvertQuAllocToQirAlloc(RewritePattern):
    """Convert qu.alloc to qir.qubit_allocate"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qu.AllocOp, rewriter: PatternRewriter):
        new_ops: List[qir.QubitAllocateOp] = []
        for _ in op.outs:
            alloc_op = qir.QubitAllocateOp()
            new_ops.append(alloc_op)

        rewriter.replace_matched_op(new_ops, [op.result for op in new_ops])


class ConvertQrefGateToQir(RewritePattern):
    """Convert qref.gate operations to QIR gate operations"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.GateOp, rewriter: PatternRewriter):
        match op.gate:
            case HadamardGate():
                rewriter.replace_matched_op(qir.HGateOp(op.ins[0]))

            case XGate():
                rewriter.replace_matched_op(qir.XGateOp(op.ins[0]))

            case YGate():
                rewriter.replace_matched_op(qir.YGateOp(op.ins[0]))

            case ZGate():
                rewriter.replace_matched_op(qir.ZGateOp(op.ins[0]))

            case PhaseGate():
                rewriter.replace_matched_op(qir.SGateOp(op.ins[0]))

            case TGate():
                rewriter.replace_matched_op(qir.TGateOp(op.ins[0]))

            case CXGate():
                rewriter.replace_matched_op(qir.CXGateOp(op.ins[0], op.ins[1]))

            case CZGate():
                rewriter.replace_matched_op(qir.CZGateOp(op.ins[0], op.ins[1]))

            case _:
                # Skip unsupported gates
                pass


class ConvertQrefMeasureToQir(RewritePattern):
    """Convert qref.measure to QIR measure + read_result"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.MeasureOp, rewriter: PatternRewriter):
        if not isinstance(op.measurement, CompBasisMeasurementAttr):
            return

        measure_ops: List[qir.MeasureOp] = []
        read_ops: List[qir.ReadResultOp] = []

        for qubit in op.in_qubits:
            measure_op = qir.MeasureOp(qubit)
            read_op = qir.ReadResultOp(measure_op.result)
            measure_ops.append(measure_op)
            read_ops.append(read_op)

        all_ops = measure_ops + read_ops
        results = [read_op.output for read_op in read_ops]

        rewriter.replace_matched_op(all_ops, results)


class ConvertQrefToQir(ModulePass):
    """Convert qref dialect operations to QIR dialect operations"""

    name = "convert-qref-to-qir"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertQuAllocToQirAlloc(),
                    ConvertQrefGateToQir(),
                    ConvertQrefMeasureToQir(),
                ]
            )
        ).rewrite_module(op)
