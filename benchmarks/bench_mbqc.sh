#! /usr/bin/env bash

SETUP="
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region
from xdsl.context import Context
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass

from inconspiquous.dialects.gate import CXGate, HadamardGate
from inconspiquous.transforms.convert_to_mbqc import ToMBQC
from inconspiquous.transforms.convert_to_cz_j import ToCZJPass
from inconspiquous.utils.qssa_builder import QSSABuilder


def build_ghz(n: int) -> ModuleOp:
    block = Block()
    builder = QSSABuilder(InsertPoint.at_start(block))
    first = builder.alloc()
    builder.gate(HadamardGate(), first)
    for _ in range(n):
        qubit = builder.alloc()
        builder.gate(CXGate(), first, qubit)

    return ModuleOp(Region(block))

mod = build_ghz($SIZE)
ctx = Context()
"

TEST="
ToCZJPass().apply(ctx, mod)
ToMBQC().apply(ctx, mod)
"

echo "Size $SIZE"
python -m timeit -u msec -n 10 -s "$SETUP" "$TEST"
