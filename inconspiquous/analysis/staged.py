from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Generic

from typing_extensions import TypeVar
from xdsl.dialects import cf
from xdsl.ir import Block, Operation, Region, SSAValue, SSAValues, dataclass, field
from xdsl.utils.disjoint_set import DisjointSet

from inconspiquous.dialects import qref, qu

WorklistItemInvT = TypeVar("WorklistItemInvT", bound=Block | Operation)


# Should use BranchOpInterface here
def _get_branches(op: Operation | None) -> tuple[tuple[Block, SSAValues], ...]:
    match op:
        case cf.BranchOp():
            return ((op.successor, op.arguments),)
        case cf.ConditionalBranchOp():
            return (
                (op.then_block, op.then_arguments),
                (op.else_block, op.else_arguments),
            )
        case _:
            return ()


@dataclass(eq=False)
class Worklist(Generic[WorklistItemInvT]):
    _stack: list[WorklistItemInvT | None] = field(
        default_factory=list[WorklistItemInvT | None]
    )
    """
    The list of operations to iterate over, used as a last-in-first-out stack.
    Operations are added and removed at the end of the list.
    Operation that are `None` are meant to be discarded, and are used to
    keep removal of operations O(1).
    """

    _map: dict[WorklistItemInvT, int] = field(
        default_factory=dict[WorklistItemInvT, int]
    )
    """
    The map of operations to their index in the stack.
    It is used to check if an operation is already in the stack, and to
    remove it in O(1).
    """

    def is_empty(self) -> bool:
        """Check if the worklist is empty."""
        while self._stack and self._stack[-1] is None:
            self._stack.pop()
        return not bool(self._stack)

    def push(self, op: WorklistItemInvT):
        """
        Push an operation to the end of the worklist, if it is not already in it.
        """
        if op not in self._map:
            self._map[op] = len(self._stack)
            self._stack.append(op)

    def pop(self) -> WorklistItemInvT | None:
        """Pop the operation at the end of the worklist."""
        # All `None` operations at the end of the stack are discarded,
        # as they were removed previously.
        # We either return `None` if the stack is empty, or the last operation
        # that is not `None`.
        while self._stack:
            op = self._stack.pop()
            if op is not None:
                del self._map[op]
                return op
        return None

    def remove(self, op: WorklistItemInvT):
        """Remove an operation from the worklist."""
        if op in self._map:
            index = self._map[op]
            self._stack[index] = None
            del self._map[op]


class LiveVariableAnalysis:
    _live_in: dict[Block, set[SSAValue]]

    def __init__(self, region: Region):
        self._live_in = {}
        block_defs: dict[Block, set[SSAValue]] = {}
        for block in region.blocks:
            live = set[SSAValue]()
            defs = set[SSAValue]()
            for op in block.ops:
                for operand in op.operands:
                    live.add(operand)
                for result in op.results:
                    defs.add(result)
            for arg in block.args:
                defs.add(arg)
            live.difference_update(defs)
            self._live_in[block] = live
            block_defs[block] = defs

        worklist = Worklist[Block]()
        for block in region.blocks:
            worklist.push(block)

        while (block := worklist.pop()) is not None:
            assert block.last_op is not None
            succ_live = set[SSAValue]()
            for succ in block.last_op.successors:
                succ_live.update(self._live_in[succ])
            succ_live.difference_update(block_defs[block])
            if not succ_live.issubset(self._live_in[block]):
                self._live_in[block].update(succ_live)
                for predecessor in block.uses:
                    parent = predecessor.operation.parent_block()
                    assert parent is not None
                    worklist.push(parent)

    def live_in(self, block: Block) -> AbstractSet[SSAValue]:
        return self._live_in[block]


class CircuitAnalysis:
    _circuits: dict[Block, DisjointSet[SSAValue]]
    _circuit_maps: dict[tuple[Block, Block], dict[SSAValue, SSAValue]]

    def __init__(self, region: Region, *, liveness: LiveVariableAnalysis | None = None):
        if liveness is None:
            liveness = LiveVariableAnalysis(region)
        self._circuits = {}
        self._circuit_maps = {}
        for block in region.blocks:
            ds = DisjointSet[SSAValue]()
            # Add block arguments to the set but don't unify them yet
            for arg in block.args:
                if arg.type == qu.BitType():
                    ds.add(arg)
            for i in liveness.live_in(block):
                if i.type == qu.BitType():
                    ds.add(i)

            for op in block.ops:
                match op:
                    case qu.AllocOp(outs=outs):
                        # Add and unify alloc results
                        if outs:
                            ds.add(outs[0])
                            for o in outs[1:]:
                                ds.add(o)
                                ds.union(outs[0], o)
                    case (
                        qref.GateOp(in_qubits=in_qubits)
                        | qref.DynGateOp(in_qubits=in_qubits)
                    ):
                        # Unify inputs
                        for i in in_qubits[1:]:
                            ds.union(in_qubits[0], i)
                    case _:
                        pass
            self._circuits[block] = ds
            term = block.last_op
            assert term is not None
            for succ, operands in _get_branches(term):
                circuit_map = dict[SSAValue, SSAValue]()
                for op, arg in zip(operands, succ.args, strict=True):
                    if op.type == qu.BitType():
                        circuit_map[op] = arg
                for value in liveness.live_in(succ):
                    if value.type == qu.BitType():
                        circuit_map[value] = value
                self._circuit_maps[(block, succ)] = circuit_map

        worklist = Worklist[Block]()

        for block in reversed(region.blocks):
            worklist.push(block)

        # Now process branching operations
        while (block := worklist.pop()) is not None:
            # Check predecessors
            for use in block.uses:
                pred = use.operation.parent_block()
                assert pred is not None
                circuit_map = self._circuit_maps[(pred, block)]
                root_dict = dict[SSAValue, SSAValue]()
                new_circuit_map = dict[SSAValue, SSAValue]()
                for src, tgt in circuit_map.items():
                    root = self._circuits[block].find(tgt)
                    if root in root_dict:
                        self._circuits[pred].union_left(root_dict[root], src)
                        worklist.push(pred)
                    else:
                        new_circuit_map[src] = root
                        root_dict[root] = src
                self._circuit_maps[(pred, block)] = new_circuit_map

            # Check successors
            term = block.last_op
            assert term is not None
            for succ in term.successors:
                circuit_map = self._circuit_maps[(block, succ)]
                root_dict = dict[SSAValue, SSAValue]()
                new_circuit_map = dict[SSAValue, SSAValue]()
                for src, tgt in circuit_map.items():
                    root = self._circuits[block].find(src)
                    if root in root_dict:
                        self._circuits[succ].union_left(root_dict[root], tgt)
                        worklist.push(succ)
                    else:
                        new_circuit_map[root] = tgt
                        root_dict[root] = tgt
                self._circuit_maps[(block, succ)] = new_circuit_map

    def circuits(self, block: Block) -> DisjointSet[SSAValue]:
        return self._circuits[block]

    def circuit_map(self, src: Block, dest: Block) -> dict[SSAValue, SSAValue]:
        return self._circuit_maps[(src, dest)]


class MeasurementAnalysis:
    _circuit_deps: dict[Block, dict[SSAValue, set[SSAValue]]]

    def __init__(
        self,
        region: Region,
        *,
        liveness: LiveVariableAnalysis | None = None,
        circuits: CircuitAnalysis | None = None,
    ):
        self._circuit_deps = {}
        if liveness is None:
            liveness = LiveVariableAnalysis(region)
        if circuits is None:
            circuits = CircuitAnalysis(region, liveness=liveness)

        worklist = Worklist[Block]()
        for block in reversed(region.blocks):
            self._circuit_deps[block] = {
                x: set() for x in block.args + tuple(liveness.live_in(block))
            }
            worklist.push(block)

        while (block := worklist.pop()) is not None:
            # Regenerate local circuit dependencies
            block_deps = self._circuit_deps[block]
            for op in block.ops:
                deps = set(x for operand in op.operands for x in block_deps[operand])
                if isinstance(op, qref.MeasureOp):
                    deps.add(circuits.circuits(block).find(op.in_qubits[0]))
                for res in op.results:
                    block_deps[res] = deps

            # Calculate CFG edge dependencies
            term = block.last_op
            assert term is not None
            for succ, operands in _get_branches(term):
                succ_deps = self._circuit_deps[succ]
                circuit_map = circuits.circuit_map(block, succ)
                for o, a in zip(operands, succ.args, strict=True):
                    new_deps = set(
                        circuit_map[dep] for dep in block_deps[o] if dep in circuit_map
                    )
                    if not new_deps.issubset(succ_deps[a]):
                        succ_deps[a] = succ_deps[a].union(new_deps)
                        worklist.push(succ)
                for value in liveness.live_in(succ):
                    new_deps = set(
                        circuit_map[dep]
                        for dep in block_deps[value]
                        if dep in circuit_map
                    )
                    if not new_deps.issubset(succ_deps[value]):
                        succ_deps[value] = succ_deps[value].union(new_deps)
                        worklist.push(succ)

    def circuit_deps(self, block: Block) -> dict[SSAValue, set[SSAValue]]:
        return self._circuit_deps[block]
