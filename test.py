from xdsl.dialects.func import FuncOp
from xdsl.parser import Parser

from inconspiquous.analysis.staged import CircuitAnalysis, LiveVariableAnalysis
from inconspiquous.tools.quopt_main import QuoptMain

prog = """
builtin.module {
  func.func @test() {
    %q = qu.alloc
    %q3 = qu.alloc
    cf.br ^bb0(%q : !qu.bit)
  ^bb0(%q1 : !qu.bit):
    %b = qref.measure %q1
    qref.gate<#gate.h> %q3
    %q2 = qu.alloc
    qref.gate<#gate.cz> %q2, %q3
    cf.br ^bb0(%q2 : !qu.bit)
  }
}
"""

module = Parser(QuoptMain().ctx, prog).parse_module()

assert module.body.first_block is not None
func = module.body.first_block.first_op
assert isinstance(func, FuncOp)

liveness = LiveVariableAnalysis(func.body)

for block in func.body.blocks:
    print(block)
    print(liveness.live_in(block))

print()

circuits = CircuitAnalysis(func.body)

for block in func.body.blocks:
    print(block)
    print(circuits.circuits(block))

print()
print(circuits.circuit_map(func.body.blocks[0], func.body.blocks[1]))
print()
print(circuits.circuit_map(func.body.blocks[1], func.body.blocks[1]))
