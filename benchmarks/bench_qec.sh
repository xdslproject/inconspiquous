#! /usr/bin/env bash

SETUP="
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region
from xdsl.context import Context
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination

from inconspiquous.dialects.gate import CXGate, HadamardGate
from inconspiquous.dialects.qec import PerfectCode5QubitCorrectionAttr
from inconspiquous.transforms.flip_coins import FlipCoinsPass
from inconspiquous.transforms.pauli_fusion import PauliFusionPass
from inconspiquous.transforms.qec.inline import QECInlinerPass
from inconspiquous.transforms.xzs.pipeline import XZSSimpl
from inconspiquous.transforms.xzs.convert_to_xzs import ConvertToXZS
from inconspiquous.transforms.xzs.select import XZSSelect
from inconspiquous.transforms.xzs.commute import XZCommute
from inconspiquous.utils.qssa_builder import QSSABuilder


def build_qec(n: int) -> ModuleOp:
    block = Block()
    builder = QSSABuilder(InsertPoint.at_start(block))
    q1 = builder.alloc()
    q2 = builder.alloc()
    q3 = builder.alloc()
    q4 = builder.alloc()
    q5 = builder.alloc()
    for _ in range(n):
        builder.gate(PerfectCode5QubitCorrectionAttr(), q1, q2, q3, q4, q5)

    return ModuleOp(Region(block))

mod = build_qec($PASSES)
ctx = Context()
QECInlinerPass().apply(ctx, mod)
ConvertToXZS().apply(ctx, mod)
"

TEST="
XZSSelect().apply(ctx, mod)
XZCommute().apply(ctx, mod)
CommonSubexpressionElimination().apply(ctx, mod)
CanonicalizePass().apply(ctx, mod)
"

echo "For $PASSES passes"
python -m timeit -u msec -n 10 -s "$SETUP" "$TEST"
