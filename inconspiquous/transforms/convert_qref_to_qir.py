from xdsl.dialects import arith
from xdsl.dialects import builtin
from xdsl.dialects.builtin import Float64Type, FloatAttr
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from inconspiquous.dialects import qref, qir
from inconspiquous.dialects.gate import (
    CXGate,
    CZGate,
    HadamardGate,
    PhaseDaggerGate,
    PhaseGate,
    RZGate,
    TDaggerGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from inconspiquous.dialects.measurement import (
    CompBasisMeasurementAttr,
    XBasisMeasurementAttr,
)
from inconspiquous.dialects import qu


class QRefTypeToQIRPattern(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: qu.BitType) -> qir.QubitType:
        return qir.QubitType()


class QRefGateToQIRPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.GateOp, rewriter: PatternRewriter):
        match op.gate:
            case CXGate():
                rewriter.replace_matched_op(qir.CNotOp(op.ins[0], op.ins[1]))
            case CZGate():
                rewriter.replace_matched_op(qir.CZOp(op.ins[0], op.ins[1]))
            case HadamardGate():
                rewriter.replace_matched_op(qir.HOp(op.ins[0]))
            case PhaseGate():
                rewriter.replace_matched_op(qir.SOp(op.ins[0]))
            case PhaseDaggerGate():
                rewriter.replace_matched_op(qir.SAdjOp(op.ins[0]))
            case TGate():
                rewriter.replace_matched_op(qir.TOp(op.ins[0]))
            case TDaggerGate():
                rewriter.replace_matched_op(qir.TAdjOp(op.ins[0]))
            case XGate():
                rewriter.replace_matched_op(qir.XOp(op.ins[0]))
            case YGate():
                rewriter.replace_matched_op(qir.YOp(op.ins[0]))
            case ZGate():
                rewriter.replace_matched_op(qir.ZOp(op.ins[0]))
            case RZGate():
                rewriter.replace_matched_op(
                    (
                        const := arith.ConstantOp(
                            FloatAttr(op.gate.angle.as_float(), type=Float64Type())
                        ),
                        qir.RZOp(const, op.ins[0]),
                    )
                )
            case _:
                return


class QRefAllocToQIRPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qu.AllocOp, rewriter: PatternRewriter, /):
        if not op.alloc == qu.AllocZeroAttr():
            return
        rewriter.replace_matched_op(qir.QubitAllocateOp())


class QRefMeasureToQIRPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.MeasureOp, rewriter: PatternRewriter, /):
        match op.measurement:
            case CompBasisMeasurementAttr():
                correction = ()
            case XBasisMeasurementAttr():
                correction = (qir.HOp(op.in_qubits[0]),)
            case _:
                return

        rewriter.replace_matched_op(
            correction
            + (
                m := qir.MeasureOp(op.in_qubits[0]),
                qir.ReleaseOp(op.in_qubits[0]),
                one := qir.ResultGetOneOp(),
                qir.ResultEqualOp(m, one),
            )
        )


class ConvertQRefToQIRPass(ModulePass):
    name = "convert-qref-to-qir"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    QRefTypeToQIRPattern(recursive=True),
                    QRefAllocToQIRPattern(),
                    QRefGateToQIRPattern(),
                    QRefMeasureToQIRPattern(),
                ]
            )
        ).rewrite_module(op)
