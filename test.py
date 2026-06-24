import pprint

from networkx import recursive_simple_cycles
from xdsl.dialects.func import FuncOp
from xdsl.parser import Parser

from inconspiquous.analysis.staged import (
    CircuitAnalysis,
    DependencyAnalysis,
    LiveVariableAnalysis,
    MeasurementAnalysis,
)
from inconspiquous.tools.quopt_main import QuoptMain

staged = """
func.func @staged() {
  %q0 = qu.alloc
  %q1 = qu.alloc

  %b = qref.measure %q0
  cf.cond_br %b, ^bb0, ^bb1
^bb0:
  qref.gate<#gate.x> %q1
  cf.br ^bb1
^bb1:
  func.return
}
"""

not_staged = """
func.func @not_staged() {
  %q0 = qu.alloc
  %q1 = qu.alloc
  qref.gate<#gate.cx> %q0, %q1
  %b = qref.measure %q0
  cf.cond_br %b, ^bb0, ^bb1
^bb0:
  qref.gate<#gate.x> %q1
  cf.br ^bb1
^bb1:
  func.return
}
"""

is_it_staged = """
func.func @is_it_staged() {
  %q0 = qu.alloc
  cf.br ^bb0(%q0 : !qu.bit)
^bb0(%q1 : !qu.bit):
  %b = qref.measure %q1
  %q2 = qu.alloc
  cf.cond_br %b, ^bb1, ^bb0(%q2 : !qu.bit)
^bb1:
  qref.gate<#gate.x> %q2
  cf.br ^bb0(%q2 : !qu.bit)
}
"""

prog = """
builtin.module {
  func.func @test() {
    %q = qu.alloc
    %q3 = qu.alloc
    %cFalse = arith.constant false
    cf.br ^bb0(%q, %cFalse : !qu.bit, i1)
  ^bb0(%q1 : !qu.bit, %b1: i1):
    %b = qref.measure %q1
    qref.gate<#gate.h> %q3
    %q2 = qu.alloc
    %g1 = gate.constant #gate.h
    %g2 = gate.constant #gate.id<1>
    %g3 = arith.select %b, %g1, %g2 : !gate.type<1>
    qref.dyn_gate<%g3> %q2
    qref.gate<#gate.cz> %q2, %q3
    cf.br ^bb0(%q2, %b : !qu.bit, i1)
  }
}
"""

for prog in (staged, not_staged, is_it_staged):
    print(prog)

    module = Parser(QuoptMain().ctx, prog).parse_module()

    assert module.body.first_block is not None
    func = module.body.first_block.first_op
    assert isinstance(func, FuncOp)

    liveness = LiveVariableAnalysis(func.body)
    circuits = CircuitAnalysis(func.body, liveness=liveness)
    measurements = MeasurementAnalysis(func.body, liveness=liveness, circuits=circuits)
    graph = DependencyAnalysis(
        func.body, liveness=liveness, circuits=circuits, measurements=measurements
    )

    for i, block in enumerate(func.body.blocks):
        print("---------------------")
        print("Block ", i)
        print("Liveness: ", liveness.live_in(block))
        print("Circuits: ", circuits.circuits(block))
        print("Measurements: ", pprint.pformat(measurements.circuit_deps(block)))
        print("Dependencies: ", graph.circuit_graph(block))
        print("Violations: ", recursive_simple_cycles(graph.circuit_graph(block)))
