from xdsl.dialects import arith, builtin
from xdsl.parser import BoolAttr, Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from inconspiquous.dialects import angle, measurement
from inconspiquous.dialects import qssa, qu
from inconspiquous.dialects.gate import (
    CZGate,
    ConstantGateOp,
    DynJGate,
    IdentityGate,
    JGate,
    XGate,
)

"""
CME is a normal form for MBQC patterns, see https://en.wikipedia.org/wiki/One-way_quantum_computer#CME_pattern which uses only entanglement operations, measurement, and classical controlled correction. This pass rewrites operations to be of this form.

As CZ gates are legal in CME, this pass only rewrites J gates into CME form.
"""


class ToCMEPattern(RewritePattern):
    """
    Converts a J Gate to the CME Pattern.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qssa.GateOp, rewriter: PatternRewriter):
        if not isinstance(op.gate, JGate):
            return

        q1 = op.ins[0]
        q2 = qu.AllocOp(qu.AllocPlusAttr())
        cz = qssa.GateOp(CZGate(), q1, q2)

        m = qssa.MeasureOp(
            cz.outs[0], measurement=measurement.XYMeasurementAttr(-op.gate.angle)
        )

        x = ConstantGateOp(XGate())
        i = ConstantGateOp(IdentityGate())

        x_sel = arith.SelectOp(m.outs[0], x, i)

        x_gate = qssa.DynGateOp(x_sel, cz.outs[1])

        rewriter.replace_matched_op((q2, cz, m, x, i, x_sel, x_gate))


class DynToCMEPattern(RewritePattern):
    """
    Converts a dynamic J gate to the CME Pattern.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: qssa.DynGateOp, rewriter: PatternRewriter, /):
        if not isinstance(op.gate.owner, DynJGate):
            return

        q1 = op.ins[0]
        q2 = qu.AllocOp(qu.AllocPlusAttr())
        cz = qssa.GateOp(CZGate(), q1, q2)
        ctrue = arith.ConstantOp(BoolAttr.from_bool(True))
        a = angle.CondNegateAngleOp(ctrue, op.gate.owner.angle)
        dyn_measure = measurement.XYDynMeasurementOp(a)

        m = qssa.DynMeasureOp(cz.outs[0], measurement=dyn_measure)

        x = ConstantGateOp(XGate())
        i = ConstantGateOp(IdentityGate())

        x_sel = arith.SelectOp(m.outs[0], x, i)

        x_gate = qssa.DynGateOp(x_sel, cz.outs[1])

        rewriter.replace_matched_op(
            (q2, cz, ctrue, a, dyn_measure, m, x, i, x_sel, x_gate)
        )


class ToCMEPass(ModulePass):
    name = "convert-to-cme"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ToCMEPattern(), DynToCMEPattern()])
        ).rewrite_module(op)
