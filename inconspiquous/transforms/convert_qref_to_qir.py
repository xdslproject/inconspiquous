# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
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
from xdsl.ir import Operation, SSAValue

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


class QIRConversionError(Exception):
    """Base class for QIR conversion errors"""

    pass


class UnsupportedGateError(QIRConversionError):
    """Gate not supported in QIR"""

    pass


class ConvertQuAllocToQirAlloc(RewritePattern):
    """Convert qu.alloc to qir.qubit_allocate"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qu.AllocOp, rewriter: PatternRewriter) -> None:
        try:
            new_ops: List[Operation] = []
            for _ in op.outs:
                alloc_op = qir.QubitAllocateOp()
                new_ops.append(alloc_op)

            rewriter.replace_matched_op(new_ops, [op.result for op in new_ops])
        except Exception as e:
            raise QIRConversionError(f"Failed to convert qubit allocation: {e}")


class ConvertQrefGateToQir(RewritePattern):
    """Convert qref.gate operations to QIR gate operations"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.GateOp, rewriter: PatternRewriter):
        try:
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
                    # Skip unsupported gates with warning
                    raise UnsupportedGateError(f"Gate {op.gate} not supported in QIR")
        except Exception as e:
            raise QIRConversionError(f"Failed to convert gate operation: {e}")


class ConvertQrefMeasureToQir(RewritePattern):
    """Convert qref.measure to QIR measure + read_result"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.MeasureOp, rewriter: PatternRewriter) -> None:
        try:
            if not isinstance(op.measurement, CompBasisMeasurementAttr):
                raise QIRConversionError(
                    "Only computational basis measurements supported in QIR"
                )

            measure_ops: List[Operation] = []
            read_ops: List[Operation] = []

            for qubit in op.in_qubits:
                measure_op = qir.MeasureOp(qubit)
                read_op = qir.ReadResultOp(measure_op.result)
                measure_ops.append(measure_op)
                read_ops.append(read_op)

            all_ops = measure_ops + read_ops
            results: List[SSAValue] = [read_op.output for read_op in read_ops]

            rewriter.replace_matched_op(all_ops, results)
        except Exception as e:
            raise QIRConversionError(f"Failed to convert measurement operation: {e}")


class ConvertQrefToQir(ModulePass):
    """Convert qref dialect operations to QIR dialect operations"""

    name = "convert-qref-to-qir"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            PatternRewriteWalker(
                GreedyRewritePatternApplier(
                    [
                        ConvertQuAllocToQirAlloc(),
                        ConvertQrefGateToQir(),
                        ConvertQrefMeasureToQir(),
                    ]
                )
            ).rewrite_module(op)
        except Exception as e:
            raise QIRConversionError(f"Failed to convert module to QIR: {e}")
