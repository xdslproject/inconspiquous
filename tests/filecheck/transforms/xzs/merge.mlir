// RUN: quopt %s -p merge-xzs,dce | filecheck %s

// CHECK:      func.func @merge(%q : !qubit.bit, %x1 : i1, %x2 : i1, %z1 : i1, %z2 : i1, %s1 : i1, %s2 : i1) -> !qubit.bit {
// CHECK-NEXT:   %0 = arith.addi %x1, %x2 : i1
// CHECK-NEXT:   %1 = arith.andi %x2, %s1 : i1
// CHECK-NEXT:   %2 = arith.addi %z1, %z2 : i1
// CHECK-NEXT:   %3 = arith.addi %1, %2 : i1
// CHECK-NEXT:   %4 = arith.addi %s1, %s2 : i1
// CHECK-NEXT:   %g1 = gate.xzs %0, %3, %4
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g1> %q : !qubit.bit
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
// CHECK-NEXT: }
func.func @merge(%q: !qubit.bit, %x1: i1, %x2: i1, %z1: i1, %z2: i1, %s1: i1, %s2: i1) -> !qubit.bit {
  %g1 = gate.xzs %x1, %z1, %s1
  %q_1 = qssa.dyn_gate<%g1> %q : !qubit.bit
  %g2 = gate.xzs %x2, %z2, %s2
  %q_2 = qssa.dyn_gate<%g2> %q_1 : !qubit.bit
  func.return %q_2 : !qubit.bit
}
