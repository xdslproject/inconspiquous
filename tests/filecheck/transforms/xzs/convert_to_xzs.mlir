// RUN: quopt %s -p convert-to-xzs,cse | filecheck %s

// CHECK:      func.func @id(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %g = gate.xz %cFalse, %cFalse
// CHECK-NEXT:   %q_1 = qssa.dyn_apply<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_apply<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @id(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.apply<#gate.id<1>> %q
  %g = instrument.constant #gate.id<1>
  %q_2 = qssa.dyn_apply<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @x(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xz %cTrue, %cFalse
// CHECK-NEXT:   %q_1 = qssa.dyn_apply<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_apply<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @x(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.apply<#gate.x> %q
  %g = instrument.constant #gate.x
  %q_2 = qssa.dyn_apply<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @y(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xz %cTrue, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_apply<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_apply<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @y(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.apply<#gate.y> %q
  %g = instrument.constant #gate.y
  %q_2 = qssa.dyn_apply<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @z(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xz %cFalse, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_apply<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_apply<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @z(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.apply<#gate.z> %q
  %g = instrument.constant #gate.z
  %q_2 = qssa.dyn_apply<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @phase(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xzs %cFalse, %cFalse, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_apply<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_apply<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @phase(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.apply<#gate.s> %q
  %g = instrument.constant #gate.s
  %q_2 = qssa.dyn_apply<%g> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @phase_dagger(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %g = gate.xzs %cFalse, %cTrue, %cTrue
// CHECK-NEXT:   %q_1 = qssa.dyn_apply<%g> %q
// CHECK-NEXT:   %q_2 = qssa.dyn_apply<%g> %q_1
// CHECK-NEXT:   func.return %q_2 : !qu.bit
// CHECK-NEXT: }
func.func @phase_dagger(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.apply<#gate.s_dagger> %q
  %g = instrument.constant #gate.s_dagger
  %q_2 = qssa.dyn_apply<%g> %q_1
  func.return %q_2 : !qu.bit
}
