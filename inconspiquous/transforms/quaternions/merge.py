from xdsl.dialects.arith import Addf, Addi, Mulf, Muli, Subf, Subi
from xdsl.irdl import base
from xdsl.parser import IntegerType
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.isattr import isattr

from inconspiquous.dialects.gate import QuaternionGateOp
from inconspiquous.dialects.qssa import DynGateOp


class MergeSequencedQuaternion(RewritePattern):
    """
    Merges two quaternion gates which appear in sequence by multiplying them.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynGateOp, rewriter: PatternRewriter):
        gate2 = op.gate.owner
        if not isinstance(gate2, QuaternionGateOp):
            return

        preceding = op.ins[0].owner

        if not isinstance(preceding, DynGateOp):
            return

        gate1 = preceding.gate.owner
        if not isinstance(gate1, QuaternionGateOp):
            return

        if isattr(gate1.real.type, base(IntegerType)):
            Mul = Muli
            Add = Addi
            Sub = Subi
        else:
            Mul = Mulf
            Add = Addf
            Sub = Subf

        realreal = Mul(gate1.real, gate2.real)
        ii = Mul(gate1.i, gate2.i)
        jj = Mul(gate2.j, gate2.j)
        kk = Mul(gate2.k, gate2.k)
        real1 = Sub(realreal, ii)
        real2 = Sub(real1, jj)
        real = Sub(real2, kk)

        reali = Mul(gate1.real, gate2.i)
        ireal = Mul(gate1.i, gate2.real)
        jk = Mul(gate1.j, gate2.k)
        kj = Mul(gate1.k, gate2.j)
        i1 = Add(reali, ireal)
        i2 = Add(i1, jk)
        i = Sub(i2, kj)

        realj = Mul(gate1.real, gate2.j)
        jreal = Mul(gate1.j, gate2.real)
        ik = Mul(gate1.i, gate2.k)
        ki = Mul(gate1.k, gate2.i)
        j1 = Add(realj, jreal)
        j2 = Add(j1, ki)
        j = Sub(j2, ik)

        realk = Mul(gate1.real, gate2.k)
        kreal = Mul(gate1.k, gate2.real)
        ij = Mul(gate1.i, gate2.j)
        ji = Mul(gate1.j, gate2.i)
        k1 = Add(realk, kreal)
        k2 = Add(k1, ij)
        k = Sub(k2, ji)

        new_quaternion = QuaternionGateOp(real, i, j, k)

        rewriter.insert_op(
            (
                realreal,
                ii,
                jj,
                kk,
                real1,
                real2,
                real,
                reali,
                ireal,
                jk,
                kj,
                i1,
                i2,
                i,
                realj,
                jreal,
                ik,
                ki,
                j1,
                j2,
                j,
                realk,
                kreal,
                ij,
                ji,
                k1,
                k2,
                k,
                new_quaternion,
            ),
            InsertPoint.after(gate2),
        )

        rewriter.replace_matched_op(DynGateOp(new_quaternion, *preceding.ins))

        rewriter.erase_op(preceding)
