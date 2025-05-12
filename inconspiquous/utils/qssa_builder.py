from xdsl.builder import Builder, ImplicitBuilder
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
class QSSABuilder(Builder):
    """
    Helper for building qssa circuits.
    """

    def gate(self, gate: GateAttr | SSAValue | Operation, *qubit_refs: QubitRef):
        if isinstance(gate, GateAttr):
            new_op = GateOp(gate, *(ref.get() for ref in qubit_refs))
        else:
            new_op = DynGateOp(gate, *(ref.get() for ref in qubit_refs))
        if ImplicitBuilder.get() is None:
            self.insert(new_op)
        for ref, qubit in zip(qubit_refs, new_op.outs):
            qubit.name_hint = ref.get().name_hint
            ref.qubit = qubit

    def alloc(self, *, name_hint: str | None = None) -> QubitRef:
        new_op = AllocOp()
        if ImplicitBuilder.get() is None:
            self.insert(new_op)
        qubit = new_op.outs[0]
        qubit.name_hint = name_hint
        return QubitRef(qubit)

    def measure(self, ref: QubitRef, *, name_hint: str | None = None) -> SSAValue:
        new_op = MeasureOp(ref.get())
        if ImplicitBuilder.get() is None:
            self.insert(new_op)
        ref.qubit = None
        out = new_op.outs[0]
        out.name_hint = name_hint
        return out
