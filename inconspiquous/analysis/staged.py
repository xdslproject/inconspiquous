from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Generic, NamedTuple

from bidict import BidirectionalMapping, bidict
from networkx import DiGraph
from typing_extensions import TypeVar
from xdsl.dialects import cf
from xdsl.ir import Block, Operation, Region, SSAValue, SSAValues, dataclass, field
from xdsl.utils.disjoint_set import DisjointSet

from inconspiquous.dialects import qref, qu

WorklistItemInvT = TypeVar("WorklistItemInvT", bound=Block | Operation)


class CFGEdge(NamedTuple):
    from_block: Block
    succ_index: int

    def to_block(self) -> Block:
        term = self.from_block.last_op
        assert term is not None
        return term.successors[self.succ_index]


# Should use BranchOpInterface here
def _get_branches(op: Operation | None) -> tuple[tuple[CFGEdge, SSAValues], ...]:
    if op is None:
        return ()
    parent = op.parent_block()
    if parent is None:
        return ()
    match op:
        case cf.BranchOp():
            return ((CFGEdge(parent, 0), op.arguments),)
        case cf.ConditionalBranchOp():
            return (
                (CFGEdge(parent, 0), op.then_arguments),
                (CFGEdge(parent, 1), op.else_arguments),
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


AnalysisT = TypeVar("AnalysisT")


class BlockAnalysis(ABC, Generic[AnalysisT]):
    _analysis: dict[Block, AnalysisT]
    _worklist: Worklist[Block]

    def __init__(self, region: Region, *, rev: bool = False) -> None:
        self._analysis = {}
        self._worklist = Worklist()

        for block in reversed(region.blocks) if rev else region.blocks:
            self._initialise_block(block)
            self._worklist.push(block)

        while (block := self._worklist.pop()) is not None:
            self._update_block(block)

    @abstractmethod
    def _initialise_block(self, block: Block): ...

    @abstractmethod
    def _update_block(self, block: Block): ...


class LiveVariableAnalysis(BlockAnalysis[set[SSAValue]]):
    _block_defs: dict[Block, set[SSAValue]]

    def __init__(self, region: Region):
        self._block_defs = {}
        super().__init__(region)

    def _initialise_block(self, block: Block):
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
        self._analysis[block] = live
        self._block_defs[block] = defs

    def _update_block(self, block: Block):
        assert block.last_op is not None
        succ_live = set[SSAValue]()
        for succ in block.last_op.successors:
            succ_live |= self._analysis[succ]
        succ_live -= self._block_defs[block]
        if not succ_live <= self._analysis[block]:
            self._analysis[block] |= succ_live
            for predecessor in block.uses:
                parent = predecessor.operation.parent_block()
                assert parent is not None
                self._worklist.push(parent)

    def live_in(self, block: Block) -> AbstractSet[SSAValue]:
        return self._analysis[block]


class CircuitAnalysis(BlockAnalysis[DisjointSet[SSAValue]]):
    _circuit_maps: dict[CFGEdge, bidict[SSAValue, SSAValue]]
    _liveness: LiveVariableAnalysis

    def __init__(self, region: Region, *, liveness: LiveVariableAnalysis | None = None):
        self._liveness = liveness or LiveVariableAnalysis(region)
        self._circuit_maps = {}
        super().__init__(region, rev=True)

    def _initialise_block(self, block: Block):
        ds = DisjointSet[SSAValue]()
        # Add block arguments to the set but don't unify them yet
        for arg in block.args:
            if arg.type == qu.BitType():
                ds.add(arg)
        for i in self._liveness.live_in(block):
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
        self._analysis[block] = ds

        term = block.last_op
        assert term is not None
        for edge, operands in _get_branches(term):
            succ = edge.to_block()
            circuit_map = bidict[SSAValue, SSAValue]()
            for op, arg in zip(operands, succ.args, strict=True):
                if op.type == qu.BitType():
                    circuit_map[op] = arg
            for value in self._liveness.live_in(succ):
                if value.type == qu.BitType():
                    circuit_map[value] = value
            self._circuit_maps[edge] = circuit_map

    def _update_block(self, block: Block):
        # Check predecessors
        for use in block.uses:
            pred = use.operation.parent_block()
            assert pred is not None
            edge = CFGEdge(pred, use.index)
            self._resolve_map(block, pred, self._circuit_maps[edge].inverse)

        # Check successors
        term = block.last_op
        assert term is not None
        for i, succ in enumerate(term.successors):
            edge = CFGEdge(block, i)
            self._resolve_map(block, succ, self._circuit_maps[edge])

    def _resolve_map(
        self, src: Block, tgt: Block, circuit_map: bidict[SSAValue, SSAValue]
    ):
        src_circuits = self._analysis[src]
        tgt_circuits = self._analysis[tgt]
        old_keys = tuple(circuit_map.keys())
        for key in old_keys:
            root = src_circuits.find(key)
            if root == key:
                continue
            value = circuit_map.pop(key)
            if root in circuit_map and tgt_circuits.union_left(
                circuit_map[root], value
            ):
                self._worklist.push(tgt)

    def circuits(self, block: Block) -> DisjointSet[SSAValue]:
        return self._analysis[block]

    def circuit_map(self, edge: CFGEdge) -> BidirectionalMapping[SSAValue, SSAValue]:
        return self._circuit_maps[edge]


class MeasurementAnalysis(BlockAnalysis[dict[SSAValue, set[SSAValue]]]):
    _liveness: LiveVariableAnalysis
    _circuits: CircuitAnalysis

    def __init__(
        self,
        region: Region,
        *,
        liveness: LiveVariableAnalysis | None = None,
        circuits: CircuitAnalysis | None = None,
    ):
        self._liveness = liveness or LiveVariableAnalysis(region)
        self._circuits = circuits or CircuitAnalysis(region, liveness=self._liveness)
        super().__init__(region, rev=True)

    def _initialise_block(self, block: Block):
        self._analysis[block] = {
            x: set() for x in block.args + tuple(self._liveness.live_in(block))
        }

    def _update_block(self, block: Block):
        # Regenerate local circuit dependencies
        block_deps = self._analysis[block]
        for op in block.ops:
            deps = set(x for operand in op.operands for x in block_deps[operand])
            if isinstance(op, qref.MeasureOp):
                deps.add(self._circuits.circuits(block).find(op.in_qubits[0]))
            for res in op.results:
                block_deps[res] = deps

        # Calculate CFG edge dependencies
        term = block.last_op
        assert term is not None
        for edge, operands in _get_branches(term):
            succ = edge.to_block()
            succ_deps = self._analysis[succ]
            circuit_map = self._circuits.circuit_map(edge)
            for o, a in zip(operands, succ.args, strict=True):
                new_deps = set(
                    circuit_map[dep] for dep in block_deps[o] if dep in circuit_map
                )
                if not new_deps.issubset(succ_deps[a]):
                    succ_deps[a] = succ_deps[a].union(new_deps)
                    self._worklist.push(succ)
            for value in self._liveness.live_in(succ):
                new_deps = set(
                    circuit_map[dep] for dep in block_deps[value] if dep in circuit_map
                )
                if not new_deps.issubset(succ_deps[value]):
                    succ_deps[value] = succ_deps[value].union(new_deps)
                    self._worklist.push(succ)

    def circuit_deps(self, block: Block) -> Mapping[SSAValue, set[SSAValue]]:
        return self._analysis[block]


DepGraph = DiGraph[SSAValue] if TYPE_CHECKING else DiGraph


class DependencyAnalysis(BlockAnalysis[DepGraph]):
    _liveness: LiveVariableAnalysis
    _circuits: CircuitAnalysis
    _measurements: MeasurementAnalysis

    def __init__(
        self,
        region: Region,
        *,
        liveness: LiveVariableAnalysis | None = None,
        circuits: CircuitAnalysis | None = None,
        measurements: MeasurementAnalysis | None = None,
    ):
        self._liveness = liveness or LiveVariableAnalysis(region)
        self._circuits = circuits or CircuitAnalysis(region, liveness=self._liveness)
        self._measurements = measurements or MeasurementAnalysis(
            region, liveness=self._liveness, circuits=self._circuits
        )
        super().__init__(region, rev=True)

    def _initialise_block(self, block: Block):
        graph: DiGraph[SSAValue] = DiGraph()
        graph.add_nodes_from(self._circuits.circuits(block).roots())
        self._analysis[block] = graph

    def _update_block(self, block: Block):
        # Regenerate local circuit dependencies
        for op in block.ops:
            match op:
                case qref.DynGateOp():
                    circ = self._circuits.circuits(block).find(op.in_qubits[0])
                    self._analysis[block].add_edges_from(
                        (circ, x)
                        for x in self._measurements.circuit_deps(block)[op.gate]
                    )
                case cf.ConditionalBranchOp():
                    deps = self._measurements.circuit_deps(block)[op.cond]
                    if deps:
                        c = set(
                            self._circuits.circuits(block).find(x)
                            for x in op.then_arguments
                            + op.else_arguments
                            + tuple(self._liveness.live_in(op.then_block))
                            + tuple(self._liveness.live_in(op.else_block))
                            if x.type == qu.BitType()
                        )
                        self._analysis[block].add_edges_from(
                            (x, y) for x in c for y in deps
                        )
                case _:
                    pass

        # Propagate forward
        term = block.last_op
        assert term is not None
        for i, succ in enumerate(term.successors):
            edge = CFGEdge(block, i)
            if self._propagate_edge(block, succ, self._circuits.circuit_map(edge)):
                self._worklist.push(succ)
        # Propagate backwards
        for use in block.uses:
            pred = use.operation.parent_block()
            assert pred is not None
            edge = CFGEdge(pred, use.index)
            if self._propagate_edge(
                block, pred, self._circuits.circuit_map(edge).inverse
            ):
                self._worklist.push(pred)

    def _propagate_edge(
        self, src: Block, tgt: Block, mapping: Mapping[SSAValue, SSAValue]
    ) -> bool:
        graph = self._analysis[tgt]
        edges_before = graph.number_of_edges()
        graph.add_edges_from(
            (mapping[x], mapping[y])
            for x, y in self._analysis[src].edges
            if x in mapping and y in mapping
        )
        return graph.number_of_edges() != edges_before

    def circuit_graph(self, block: Block) -> DiGraph[SSAValue]:
        return self._analysis[block]
