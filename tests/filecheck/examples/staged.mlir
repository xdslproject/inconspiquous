// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @rec() -> i32 {
// CHECK-NEXT:   %c0 = arith.constant 0 : i32
// CHECK-NEXT:   %c1 = arith.constant 1 : i32
// CHECK-NEXT:   cf.br ^bb0(%c0 : i32)
// CHECK-NEXT: ^bb0(%i: i32):
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   staged.gate<#gate.h> %q0
// CHECK-NEXT:   %0 = staged.measure %q0
// CHECK-NEXT:   staged.step ^bb1(%0 : !staged.later<i1>)
// CHECK-NEXT: ^bb1(%m: i1):
// CHECK-NEXT:   %j = arith.addi %i, %c1 : i32
// CHECK-NEXT:   cf.cond_br %m, ^bb0(%j : i32), ^bb2
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   func.return %i : i32
// CHECK-NEXT: }
func.func @rec() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  cf.br ^bb0(%c0 : i32)
^bb0(%i: i32):
  %q0 = qu.alloc
  staged.gate<#gate.h> %q0
  %1 = staged.measure %q0
  staged.step ^bb1(%1 : !staged.later<i1>)
^bb1(%m: i1):
  %j = arith.addi %i, %c1 : i32
  cf.cond_br %m, ^bb0(%j : i32), ^bb2
^bb2:
  func.return %i : i32
}

// CHECK:      func.func @vqe_ansatz(%a1: !angle.type, %a2: !angle.type, %a3: !angle.type, %basis: i1) -> (i1, i1) {
// CHECK-NEXT:   %q1 = qu.alloc
// CHECK-NEXT:   %q2 = qu.alloc
// CHECK-NEXT:   %g1 = gate.dyn_rx<%a1>
// CHECK-NEXT:   %g2 = gate.dyn_rx<%a2>
// CHECK-NEXT:   %g3 = gate.dyn_rz<%a3>
// CHECK-NEXT:   %h = gate.constant #gate.h
// CHECK-NEXT:   %id = gate.constant #gate.id<1>
// CHECK-NEXT:   %g4 = arith.select %basis, %h, %id : !gate.type<1>
// CHECK-NEXT:   staged.dyn_gate<%g1> %q1
// CHECK-NEXT:   staged.dyn_gate<%g2> %q2
// CHECK-NEXT:   staged.gate<#gate.cx> %q1, %q2
// CHECK-NEXT:   staged.dyn_gate<%g2> %q1
// CHECK-NEXT:   staged.dyn_gate<%g4> %q1
// CHECK-NEXT:   staged.dyn_gate<%g4> %q2
// CHECK-NEXT:   %m1 = staged.measure %q1
// CHECK-NEXT:   %m2 = staged.measure %q2
// CHECK-NEXT:   staged.step ^bb0(%m1, %m2 : !staged.later<i1>, !staged.later<i1>)
// CHECK-NEXT: ^bb0(%0: i1, %1: i1):
// CHECK-NEXT:   func.return %0, %1 : i1, i1
// CHECK-NEXT: }
func.func @vqe_ansatz(%a1: !angle.type, %a2: !angle.type, %a3: !angle.type, %basis: i1) -> (i1, i1) {
  %q1 = qu.alloc
  %q2 = qu.alloc
  %g1 = gate.dyn_rx<%a1>
  %g2 = gate.dyn_rx<%a2>
  %g3 = gate.dyn_rz<%a3>
  %h = gate.constant #gate.h
  %id = gate.constant #gate.id<1>
  %g4 = arith.select %basis, %h, %id : !gate.type<1>

  staged.dyn_gate<%g1> %q1
  staged.dyn_gate<%g2> %q2
  staged.gate<#gate.cx> %q1, %q2
  staged.dyn_gate<%g2> %q1

  staged.dyn_gate<%g4> %q1
  staged.dyn_gate<%g4> %q2

  %m1 = staged.measure %q1
  %m2 = staged.measure %q2
  staged.step ^bb0(%m1, %m2 : !staged.later<i1>, !staged.later<i1>)
^bb0(%0: i1, %1: i1):
  func.return %0, %1 : i1, i1
}
