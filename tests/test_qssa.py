from inconspiquous.dialects.gate import HadamardGate
from inconspiquous.dialects.instrument import ConstantInstrumentOp
from inconspiquous.dialects.measurement import CompBasisMeasurementAttr
from inconspiquous.dialects.qec import StabilizerAttr
from inconspiquous.dialects.qssa import (
    ApplyOp,
    DynApplyOp,
    DynGateOp,
    DynMeasureOp,
    GateOp,
    MeasureOp,
    QssaApplyInterface,
)
from inconspiquous.dialects.qu import AllocOp


def test_qssa_create():
    q = AllocOp()
    gate = QssaApplyInterface.create_op(HadamardGate(), q)
    assert isinstance(gate, GateOp)

    g = ConstantInstrumentOp(HadamardGate())
    dyn_gate = QssaApplyInterface.create_op(g.out, q)
    assert isinstance(dyn_gate, DynGateOp)

    measure = QssaApplyInterface.create_op(CompBasisMeasurementAttr(), q)
    assert isinstance(measure, MeasureOp)

    m = ConstantInstrumentOp(CompBasisMeasurementAttr())
    dyn_measure = QssaApplyInterface.create_op(m.out, q)
    assert isinstance(dyn_measure, DynMeasureOp)

    q2 = AllocOp()
    ap = QssaApplyInterface.create_op(StabilizerAttr("X", "X"), q, q2)
    assert isinstance(ap, ApplyOp)

    i = ConstantInstrumentOp(StabilizerAttr("X", "X"))
    dyn_ap = QssaApplyInterface.create_op(i.out, q, q2)
    assert isinstance(dyn_ap, DynApplyOp)
