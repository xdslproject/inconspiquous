from xdsl.ir import Block
from xdsl.rewriter import InsertPoint
from inconspiquous.dialects import qssa, qu, gate
from inconspiquous.dialects.gate import HadamardGate
from inconspiquous.utils.qssa_builder import QSSABuilder

import pytest


def test_qssa_builder_alloc():
    block = Block()
    builder = QSSABuilder(InsertPoint.at_start(block))

    no_hint = builder.alloc()
    assert isinstance(no_hint.get().owner, qu.AllocOp)
    assert no_hint.get().name_hint is None

    hint = builder.alloc(name_hint="test")
    assert isinstance(hint.get().owner, qu.AllocOp)
    assert hint.get().name_hint == "test"


def test_qssa_gate():
    block = Block()
    builder = QSSABuilder(InsertPoint.at_start(block))

    no_hint = builder.alloc()

    builder.gate(HadamardGate(), no_hint)
    assert isinstance(no_hint.get().owner, qssa.GateOp)
    assert no_hint.get().name_hint is None

    hint = builder.alloc(name_hint="test")
    builder.gate(HadamardGate(), hint)
    assert isinstance(hint.get().owner, qssa.GateOp)
    assert hint.get().name_hint == "test"


def test_qssa_dyn_gate():
    block = Block()
    builder = QSSABuilder(InsertPoint.at_start(block))

    no_hint = builder.alloc()
    gate_val = gate.ConstantGateOp(HadamardGate())

    builder.gate(gate_val, no_hint)
    assert isinstance(no_hint.get().owner, qssa.DynGateOp)
    assert no_hint.get().name_hint is None

    hint = builder.alloc(name_hint="test")
    builder.gate(gate_val, hint)
    assert isinstance(hint.get().owner, qssa.DynGateOp)
    assert hint.get().name_hint == "test"


def test_qssa_measure():
    block = Block()
    builder = QSSABuilder(InsertPoint.at_start(block))

    q = builder.alloc()
    c = builder.measure(q)
    assert isinstance(c.owner, qssa.MeasureOp)
    assert c.name_hint is None

    q1 = builder.alloc()
    c1 = builder.measure(q1, name_hint="test")
    assert isinstance(c1.owner, qssa.MeasureOp)
    assert c1.name_hint == "test"


def test_double_measure():
    block = Block()
    builder = QSSABuilder(InsertPoint.at_start(block))

    q = builder.alloc()
    builder.measure(q)

    with pytest.raises(ValueError):
        builder.measure(q)
