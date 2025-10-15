from xdsl.ir import Operation, SSAValue
from dataclasses import dataclass

from inconspiquous.dialects.qssa import DynGateOp, GateOp, MeasureOp
from inconspiquous.dialects.qu import AllocOp
from inconspiquous.gates.core import GateAttr


@dataclass
class QubitRef:
    qubit: SSAValue | None

    def get(self) -> SSAValue:
        if self.qubit is None:
            raise ValueError("Consumed qubit was used.")
        return self.qubit


@dataclass
class QSSABuilder:
    """
    Helper for building qssa circuits.
    """

    @staticmethod
    def gate(gate: GateAttr | SSAValue | Operation, *qubit_refs: QubitRef):
        if isinstance(gate, GateAttr):
            new_op = GateOp(gate, *(ref.get() for ref in qubit_refs))
        else:
            new_op = DynGateOp(gate, *(ref.get() for ref in qubit_refs))
        for ref, qubit in zip(qubit_refs, new_op.outs):
            qubit.name_hint = ref.get().name_hint
            ref.qubit = qubit

    @staticmethod
    def alloc(*, name_hint: str | None = None) -> QubitRef:
        new_op = AllocOp()
        qubit = new_op.outs[0]
        qubit.name_hint = name_hint
        return QubitRef(qubit)

    @staticmethod
    def measure(ref: QubitRef, *, name_hint: str | None = None) -> SSAValue:
        new_op = MeasureOp(ref.get())
        ref.qubit = None
        out = new_op.outs[0]
        out.name_hint = name_hint
        return out
