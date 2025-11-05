from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import builtin
from xdsl.dialects.arith import AndIOp, ConstantOp, OrIOp, SelectOp, XOrIOp
from xdsl.parser import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from inconspiquous.dialects.gate import (
    CXGate,
    CZGate,
    ConstantGateOp,
    HadamardGate,
    IdentityGate,
    XGate,
    ZGate,
)
from inconspiquous.dialects.qec import PerfectCode5QubitCorrectionAttr
from inconspiquous.dialects.qssa import GateOp
from inconspiquous.utils.qssa_builder import QSSABuilder, QubitRef


class PerfectCode5QubitInliner(RewritePattern):
    """
    Replaces the Perfect code gate with its definition
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GateOp, rewriter: PatternRewriter):
        if op.gate != PerfectCode5QubitCorrectionAttr():
            return

        (q1, q2, q3, q4, q5) = tuple(QubitRef(qubit) for qubit in op.ins)

        builder = Builder(InsertPoint.before(op))

        with ImplicitBuilder(builder):
            a1 = QSSABuilder.alloc(name_hint="a1")
            QSSABuilder.gate(HadamardGate(), a1)
            QSSABuilder.gate(CXGate(), a1, q1)
            QSSABuilder.gate(CZGate(), a1, q2)
            QSSABuilder.gate(CZGate(), a1, q3)
            QSSABuilder.gate(CXGate(), a1, q4)
            QSSABuilder.gate(HadamardGate(), a1)
            s1 = QSSABuilder.measure(a1, name_hint="s1")

            a2 = QSSABuilder.alloc(name_hint="a2")
            QSSABuilder.gate(HadamardGate(), a2)
            QSSABuilder.gate(CXGate(), a2, q2)
            QSSABuilder.gate(CZGate(), a2, q3)
            QSSABuilder.gate(CZGate(), a2, q4)
            QSSABuilder.gate(CXGate(), a2, q5)
            QSSABuilder.gate(HadamardGate(), a2)
            s2 = QSSABuilder.measure(a2, name_hint="s2")

            a3 = QSSABuilder.alloc(name_hint="a3")
            QSSABuilder.gate(HadamardGate(), a3)
            QSSABuilder.gate(CXGate(), a3, q1)
            QSSABuilder.gate(CXGate(), a3, q3)
            QSSABuilder.gate(CZGate(), a3, q4)
            QSSABuilder.gate(CZGate(), a3, q5)
            QSSABuilder.gate(HadamardGate(), a3)
            s3 = QSSABuilder.measure(a3, name_hint="s3")

            a4 = QSSABuilder.alloc(name_hint="a4")
            QSSABuilder.gate(HadamardGate(), a4)
            QSSABuilder.gate(CZGate(), a4, q1)
            QSSABuilder.gate(CXGate(), a4, q2)
            QSSABuilder.gate(CXGate(), a4, q4)
            QSSABuilder.gate(CZGate(), a4, q5)
            QSSABuilder.gate(HadamardGate(), a4)
            s4 = QSSABuilder.measure(a4, name_hint="s4")

            x = ConstantGateOp(XGate()).out
            x.name_hint = "x"
            z = ConstantGateOp(ZGate()).out
            z.name_hint = "z"

            i = ConstantGateOp(IdentityGate(1)).out
            i.name_hint = "id"

            true = ConstantOp.from_int_and_width(1, 1)

            v0 = XOrIOp(s1, s3)
            v1 = OrIOp(v0, s2)
            cor = XOrIOp(v1, true)

            corx = AndIOp(cor, s4)
            corx_sel = SelectOp(corx, x, i)
            QSSABuilder.gate(corx_sel, q1)

            corz = AndIOp(cor, s1)
            corz_sel = SelectOp(corz, z, i)
            QSSABuilder.gate(corz_sel, q1)

            v0 = XOrIOp(s2, s4)
            v1 = OrIOp(v0, s3)
            cor = XOrIOp(v1, true)

            corx = AndIOp(cor, s1)
            corx_sel = SelectOp(corx, x, i)
            QSSABuilder.gate(corx_sel, q2)

            corz = AndIOp(cor, s2)
            corz_sel = SelectOp(corz, z, i)
            QSSABuilder.gate(corz_sel, q2)

            v0 = XOrIOp(s1, s2)
            v1 = OrIOp(v0, s4)
            cor = XOrIOp(v1, true)

            corx = AndIOp(cor, s1)
            corx_sel = SelectOp(corx, x, i)
            QSSABuilder.gate(corx_sel, q3)

            corz = AndIOp(cor, s3)
            corz_sel = SelectOp(corz, z, i)
            QSSABuilder.gate(corz_sel, q3)

            v0 = XOrIOp(s1, s4)
            v1 = XOrIOp(s2, s3)
            v2 = OrIOp(v0, v1)
            cor = XOrIOp(v2, true)

            corx = AndIOp(cor, s2)
            corx_sel = SelectOp(corx, x, i)
            QSSABuilder.gate(corx_sel, q4)

            corz = AndIOp(cor, s1)
            corz_sel = SelectOp(corz, z, i)
            QSSABuilder.gate(corz_sel, q4)

            v0 = XOrIOp(s3, s4)
            v1 = OrIOp(v0, s1)
            cor = XOrIOp(v1, true)

            corx = AndIOp(cor, s3)
            corx_sel = SelectOp(corx, x, i)
            QSSABuilder.gate(corx_sel, q5)

            corz = AndIOp(cor, s2)
            corz_sel = SelectOp(corz, z, i)
            QSSABuilder.gate(corz_sel, q5)

        rewriter.replace_matched_op(
            (), (q1.qubit, q2.qubit, q3.qubit, q4.qubit, q5.qubit)
        )


class QECInlinerPass(ModulePass):
    name = "qec-inline"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            PerfectCode5QubitInliner(), apply_recursively=False
        ).rewrite_module(op)
