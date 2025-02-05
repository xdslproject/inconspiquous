// RUN: quopt %s -p xz-commute,dce | filecheck %s

// CHECK:      func.func @h_commute(%q : !qubit.bit, %x : i1, %z : i1) {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.h> %q
// CHECK-NEXT:   %q_2 = gate.xz %z, %x
// CHECK-NEXT:   %q_3 = qssa.dyn_gate<%q_2> %q_1
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

func.func @h_commute(%q : !qubit.bit, %x : i1, %z : i1) {
  %g = gate.xz %x, %z
  %q_1 = qssa.dyn_gate<%g> %q
  %q_2 = qssa.gate<#gate.h> %q_1
  func.return
}

// CHECK:      func.func @cz_commute(%q1 : !qubit.bit, %q2 : !qubit.bit, %x1 : i1, %z1 : i1, %x2 : i1, %z2 : i1) {
// CHECK-NEXT:   %0 = arith.constant false
// CHECK-NEXT:   %1 = arith.constant false
// CHECK-NEXT:   %2, %3 = qssa.gate<#gate.cz> %q1, %q2
// CHECK-NEXT:   %4 = arith.xori %1, %x1 : i1
// CHECK-NEXT:   %5 = arith.xori %x2, %z1 : i1
// CHECK-NEXT:   %6 = gate.xz %4, %5
// CHECK-NEXT:   %q1_1 = qssa.dyn_gate<%6> %2
// CHECK-NEXT:   %7 = arith.xori %x2, %0 : i1
// CHECK-NEXT:   %8 = arith.xori %z2, %x1 : i1
// CHECK-NEXT:   %g2 = gate.xz %7, %8
// CHECK-NEXT:   %q2_1 = qssa.dyn_gate<%g2> %3
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cz_commute(%q1 : !qubit.bit, %q2 : !qubit.bit, %x1 : i1, %z1 : i1, %x2 : i1, %z2 : i1) {
  %g1 = gate.xz %x1, %z1
  %g2 = gate.xz %x2, %z2
  %q1_1 = qssa.dyn_gate<%g1> %q1
  %q2_1 = qssa.dyn_gate<%g2> %q2
  %q1_2, %q2_2 = qssa.gate<#gate.cz> %q1_1, %q2_1
  func.return
}

// CHECK:      func.func @cnot_commute(%q1 : !qubit.bit, %q2 : !qubit.bit, %x1 : i1, %z1 : i1, %x2 : i1, %z2 : i1) {
// CHECK-NEXT:   %0 = arith.constant false
// CHECK-NEXT:   %1 = arith.constant false
// CHECK-NEXT:   %2, %3 = qssa.gate<#gate.cnot> %q1, %q2
// CHECK-NEXT:   %4 = arith.xori %1, %x1 : i1
// CHECK-NEXT:   %5 = arith.xori %z2, %z1 : i1
// CHECK-NEXT:   %6 = gate.xz %4, %5
// CHECK-NEXT:   %q1_1 = qssa.dyn_gate<%6> %2
// CHECK-NEXT:   %7 = arith.xori %x2, %x1 : i1
// CHECK-NEXT:   %8 = arith.xori %z2, %0 : i1
// CHECK-NEXT:   %g2 = gate.xz %7, %8
// CHECK-NEXT:   %q2_1 = qssa.dyn_gate<%g2> %3
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cnot_commute(%q1 : !qubit.bit, %q2 : !qubit.bit, %x1 : i1, %z1 : i1, %x2 : i1, %z2 : i1) {
  %g1 = gate.xz %x1, %z1
  %g2 = gate.xz %x2, %z2
  %q1_1 = qssa.dyn_gate<%g1> %q1
  %q2_1 = qssa.dyn_gate<%g2> %q2
  %q1_2, %q2_2 = qssa.gate<#gate.cnot> %q1_1, %q2_1
  func.return
}

// CHECK:      func.func @measure_commute(%q : !qubit.bit, %x : i1, %z : i1) -> i1 {
// CHECK-NEXT:   %0 = qssa.measure %q
// CHECK-NEXT:   %1 = arith.addi %0, %x : i1
// CHECK-NEXT:   func.return %1 : i1
// CHECK-NEXT: }
func.func @measure_commute(%q : !qubit.bit, %x : i1, %z : i1) -> i1 {
  %g = gate.xz %x, %z
  %q_1 = qssa.dyn_gate<%g> %q
  %0 = qssa.measure %q_1
  func.return %0 : i1
}

// CHECK:      func.func @xy_measure_commute(%q : !qubit.bit, %x : i1, %z : i1) -> i1 {
// CHECK-NEXT:   %0 = angle.constant<0.5pi>
// CHECK-NEXT:   %1 = angle.cond_negate %x, %0
// CHECK-NEXT:   %2 = measurement.dyn_xy<%1>
// CHECK-NEXT:   %3 = qssa.dyn_measure<%2> %q
// CHECK-NEXT:   %4 = arith.addi %3, %z : i1
// CHECK-NEXT:   func.return %4 : i1
// CHECK-NEXT: }
func.func @xy_measure_commute(%q : !qubit.bit, %x : i1, %z : i1) -> i1 {
  %g = gate.xz %x, %z
  %q_1 = qssa.dyn_gate<%g> %q
  %0 = qssa.measure<#measurement.xy<0.5pi>> %q_1
  func.return %0 : i1
}

// CHECK:      func.func @dyn_xy_measure_commute(%q : !qubit.bit, %x : i1, %z : i1, %a : !angle.type) -> i1 {
// CHECK-NEXT:   %0 = angle.cond_negate %x, %a
// CHECK-NEXT:   %1 = measurement.dyn_xy<%0>
// CHECK-NEXT:   %2 = qssa.dyn_measure<%1> %q
// CHECK-NEXT:   %3 = arith.addi %2, %z : i1
// CHECK-NEXT:   func.return %3 : i1
// CHECK-NEXT: }
func.func @dyn_xy_measure_commute(%q : !qubit.bit, %x : i1, %z : i1, %a : !angle.type) -> i1 {
  %g = gate.xz %x, %z
  %q_1 = qssa.dyn_gate<%g> %q
  %m = measurement.dyn_xy<%a>
  %0 = qssa.dyn_measure<%m> %q_1
  func.return %0 : i1
}
