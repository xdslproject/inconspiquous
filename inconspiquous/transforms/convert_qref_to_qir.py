from xdsl.dialects import arith
from xdsl.parser import Context, ModuleOp
from xdsl.dialects.builtin import Float64Type, FloatAttr
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
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from inconspiquous.dialects import qref, qir, angle
from inconspiquous.dialects.gate import (
    CXGate,
    CZGate,
    ControlOp,
    DynRXGate,
    DynRYGate,
    DynRZGate,
    DynRZZGate,
    HadamardGate,
    PhaseDaggerGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    RZZGate,
    TDaggerGate,
    TGate,
    ToffoliGate,
    XGate,
    YGate,
    ZGate,
)
from inconspiquous.dialects.measurement import (
    CompBasisMeasurementAttr,
    XBasisMeasurementAttr,
    XYDynMeasurementOp,
    XYMeasurementAttr,
)
from inconspiquous.dialects import qu


class QRefTypeToQIRPattern(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: qu.BitType) -> qir.QubitType:
        return qir.QubitType()


class AngleTypeToFloatPattern(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: angle.AngleType) -> Float64Type:
        return Float64Type()


class LowerConstantAnglePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: angle.ConstantAngleOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            arith.ConstantOp(FloatAttr(op.angle.as_float(), Float64Type()))
        )


class LowerNegateAnglePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: angle.NegateAngleOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(arith.NegfOp(op.angle))


class LowerCondNegateAnglePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: angle.CondNegateAngleOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(
            (n := arith.NegfOp(op.angle), arith.SelectOp(op.cond, n, op.angle))
        )


class LowerScaleAnglePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: angle.ScaleAngleOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(arith.MulfOp(op.angle, op.scale))


class LowerAddAnglePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: angle.AddAngleOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(arith.AddfOp(op.lhs, op.rhs))


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
            case ToffoliGate():
                rewriter.replace_matched_op(qir.CCXOp(op.ins[0], op.ins[1], op.ins[2]))
            case RXGate():
                rewriter.replace_matched_op(
                    (
                        const := arith.ConstantOp(
                            FloatAttr(op.gate.angle.as_float(), type=Float64Type())
                        ),
                        qir.RXOp(const, op.ins[0]),
                    )
                )
            case RYGate():
                rewriter.replace_matched_op(
                    (
                        const := arith.ConstantOp(
                            FloatAttr(op.gate.angle.as_float(), type=Float64Type())
                        ),
                        qir.RYOp(const, op.ins[0]),
                    )
                )
            case RZGate():
                rewriter.replace_matched_op(
                    (
                        const := arith.ConstantOp(
                            FloatAttr(op.gate.angle.as_float(), type=Float64Type())
                        ),
                        qir.RZOp(const, op.ins[0]),
                    )
                )
            case RZZGate():
                rewriter.replace_matched_op(
                    (
                        const := arith.ConstantOp(
                            FloatAttr(op.gate.angle.as_float(), type=Float64Type())
                        ),
                        qir.RZZOp(const, op.ins[0], op.ins[1]),
                    )
                )
            case _:
                return


class QRefDynGateToQIRPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.DynGateOp, rewriter: PatternRewriter):
        gate_op = op.gate.owner
        control = False
        if isinstance(gate_op, ControlOp):
            control = True
            gate_op = gate_op.gate.owner

        match gate_op:
            case DynRXGate():
                rewriter.replace_matched_op(
                    (qir.CRXOp if control else qir.RXOp)(gate_op.angle, *op.ins)
                )
            case DynRYGate():
                rewriter.replace_matched_op(
                    (qir.CRYOp if control else qir.RYOp)(gate_op.angle, *op.ins)
                )
            case DynRZGate():
                rewriter.replace_matched_op(
                    (qir.CRZOp if control else qir.RZOp)(gate_op.angle, *op.ins)
                )
            case DynRZZGate():
                if control:
                    return
                rewriter.replace_matched_op((qir.RZZOp)(gate_op.angle, *op.ins))
            case _:
                return


class QRefAllocToQIRPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qu.AllocOp, rewriter: PatternRewriter, /):
        match op.alloc:
            case qu.AllocZeroAttr():
                rewriter.replace_matched_op(qir.QubitAllocateOp())
            case qu.AllocPlusAttr():
                rewriter.replace_matched_op(
                    (a := qir.QubitAllocateOp(), qir.HOp(a)), (a.out,)
                )
            case _:
                return


class QRefMeasureToQIRPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.MeasureOp, rewriter: PatternRewriter, /):
        match op.measurement:
            case CompBasisMeasurementAttr():
                correction = ()
            case XBasisMeasurementAttr():
                correction = (qir.HOp(op.in_qubits[0]),)
            case XYMeasurementAttr():
                correction = (
                    c := arith.ConstantOp(
                        FloatAttr(-op.measurement.angle.as_float_raw(), Float64Type())
                    ),
                    qir.RZOp(c, op.in_qubits[0]),
                    qir.HOp(op.in_qubits[0]),
                )
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


class QRefDynMeasureToQIRPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qref.DynMeasureOp, rewriter: PatternRewriter):
        if not isinstance(op.measurement.owner, XYDynMeasurementOp):
            return

        rewriter.replace_matched_op(
            (
                a := arith.NegfOp(op.measurement.owner.angle),
                qir.RZOp(a, op.in_qubits[0]),
                qir.HOp(op.in_qubits[0]),
                m := qir.MeasureOp(op.in_qubits[0]),
                qir.ReleaseOp(op.in_qubits[0]),
                one := qir.ResultGetOneOp(),
                qir.ResultEqualOp(m, one),
            )
        )


class ConvertQRefToQIRPass(ModulePass):
    name = "convert-qref-to-qir"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    QRefTypeToQIRPattern(recursive=True),
                    AngleTypeToFloatPattern(recursive=True),
                    LowerConstantAnglePattern(),
                    LowerNegateAnglePattern(),
                    LowerCondNegateAnglePattern(),
                    LowerScaleAnglePattern(),
                    LowerAddAnglePattern(),
                    QRefAllocToQIRPattern(),
                    QRefGateToQIRPattern(),
                    QRefDynGateToQIRPattern(),
                    QRefMeasureToQIRPattern(),
                    QRefDynMeasureToQIRPattern(),
                ]
            )
        ).rewrite_module(op)
        DeadCodeElimination().apply(ctx, op)
