import pytest

from inconspiquous.dialects import instrument, qssa, qu
from inconspiquous.dialects.gate import HadamardGate
from inconspiquous.utils.qssa_builder import QSSABuilder


def test_qssa_builder_alloc():
    no_hint = QSSABuilder.alloc()
    assert isinstance(no_hint.get().owner, qu.AllocOp)
    assert no_hint.get().name_hint is None

    hint = QSSABuilder.alloc(name_hint="test")
    assert isinstance(hint.get().owner, qu.AllocOp)
    assert hint.get().name_hint == "test"


def test_qssa_gate():
    no_hint = QSSABuilder.alloc()

    QSSABuilder.gate(HadamardGate(), no_hint)
    assert isinstance(no_hint.get().owner, qssa.GateOp)
    assert no_hint.get().name_hint is None

    hint = QSSABuilder.alloc(name_hint="test")
    QSSABuilder.gate(HadamardGate(), hint)
    assert isinstance(hint.get().owner, qssa.GateOp)
    assert hint.get().name_hint == "test"


def test_qssa_dyn_gate():
    no_hint = QSSABuilder.alloc()
    gate_val = instrument.ConstantInstrumentOp(HadamardGate())

    QSSABuilder.gate(gate_val, no_hint)
    assert isinstance(no_hint.get().owner, qssa.DynGateOp)
    assert no_hint.get().name_hint is None

    hint = QSSABuilder.alloc(name_hint="test")
    QSSABuilder.gate(gate_val, hint)
    assert isinstance(hint.get().owner, qssa.DynGateOp)
    assert hint.get().name_hint == "test"


def test_qssa_measure():
    q = QSSABuilder.alloc()
    c = QSSABuilder.measure(q)
    assert isinstance(c.owner, qssa.MeasureOp)
    assert c.name_hint is None

    q1 = QSSABuilder.alloc()
    c1 = QSSABuilder.measure(q1, name_hint="test")
    assert isinstance(c1.owner, qssa.MeasureOp)
    assert c1.name_hint == "test"


def test_double_release():
    q = QSSABuilder.alloc()
    QSSABuilder.release(q)

    with pytest.raises(ValueError, match="Consumed qubit was used"):
        QSSABuilder.release(q)
