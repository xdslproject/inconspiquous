// RUN: quopt %s -p convert-to-xzs,cse | filecheck %s

// CHECK:      func.func @id(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %g = gate.xz %cFalse, %cFalse
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_gate<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @id(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.id> %q
  %g = gate.constant #gate.id
  %q_2 = qssa.dyn_gate<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @x(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xz %cTrue, %cFalse
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_gate<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @x(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %g = gate.constant #gate.x
  %q_2 = qssa.dyn_gate<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @y(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xz %cTrue, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_gate<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @y(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.y> %q
  %g = gate.constant #gate.y
  %q_2 = qssa.dyn_gate<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @z(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xz %cFalse, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_gate<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @z(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.z> %q
  %g = gate.constant #gate.z
  %q_2 = qssa.dyn_gate<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @phase(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xzs %cFalse, %cFalse, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_gate<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @phase(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.s> %q
  %g = gate.constant #gate.s
  %q_2 = qssa.dyn_gate<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @phase_dagger(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xzs %cFalse, %cTrue, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_gate<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @phase_dagger(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.s_dagger> %q
  %g = gate.constant #gate.s_dagger
  %q_2 = qssa.dyn_gate<%g> %q_1
  func.return %q_2 : !qu.bit
}
