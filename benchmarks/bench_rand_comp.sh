#! /usr/bin/env bash

SETUP="
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region
from xdsl.context import Context
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass

from inconspiquous.dialects.gate import CXGate, HadamardGate
from inconspiquous.transforms.flip_coins import FlipCoinsPass
from inconspiquous.transforms.pauli_fusion import PauliFusionPass
from inconspiquous.transforms.randomized_comp import RandomizedComp
from inconspiquous.transforms.xzs.pipeline import XZSSimpl
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

mod = build_ghz(10)
ctx = Context()
RandomizedComp().apply(ctx, mod)
"

TEST_NAIVE="
for i in range($ITER_COUNT):
    m = mod.clone()
    FlipCoinsPass(seed=10+i).apply(ctx, m)
    CanonicalizePass().apply(ctx, m)
    PauliFusionPass().apply(ctx, m)
"

TEST_DYN="
XZSSimpl().apply(ctx, mod)
for _ in range($ITER_COUNT):
    m_clone = mod.clone()
    FlipCoinsPass(seed=10).apply(ctx, m_clone)
    CanonicalizePass().apply(ctx, m_clone)
"

echo "For $ITER_COUNT iterations"
echo "Naive"
python -m timeit -u msec -n 10 -s "$SETUP" "$TEST_NAIVE"
echo "Dyn gates"
python -m timeit -u msec -n 10 -s "$SETUP" "$TEST_DYN"
