import pprint

from xdsl.dialects.func import FuncOp
from xdsl.parser import Parser

from inconspiquous.analysis.staged import (
    CircuitAnalysis,
    LiveVariableAnalysis,
    MeasurementAnalysis,
)
from inconspiquous.tools.quopt_main import QuoptMain

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

module = Parser(QuoptMain().ctx, prog).parse_module()

assert module.body.first_block is not None
func = module.body.first_block.first_op
assert isinstance(func, FuncOp)

liveness = LiveVariableAnalysis(func.body)
circuits = CircuitAnalysis(func.body, liveness=liveness)
measurements = MeasurementAnalysis(func.body, liveness=liveness, circuits=circuits)

print("---------------------")
print("Block 0")
block0 = func.body.blocks[0]
print("Liveness: ", liveness.live_in(block0))
print("Circuits: ", circuits.circuits(block0))
print("Measurements: ", pprint.pformat(measurements.circuit_deps(block0)))

print("---------------------")
print("Block 1")
block1 = func.body.blocks[1]
print("Liveness: ", liveness.live_in(block1))
print("Circuits: ", circuits.circuits(block1))
print("Measurements: ", pprint.pformat(measurements.circuit_deps(block1)))

print("---------------------")
print("Circuit maps")
print("0 -> 1: ", circuits.circuit_map(block0, block1))
print("1 -> 1: ", circuits.circuit_map(block1, block1))
