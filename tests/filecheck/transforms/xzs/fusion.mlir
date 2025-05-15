// RUN: quopt %s -p xzs-fusion,dce | filecheck %s

// CHECK:      func.func @fusion(%q : !qu.bit, %x1 : i1, %x2 : i1, %z1 : i1, %z2 : i1, %s1 : i1, %s2 : i1) -> !qu.bit {
// CHECK-NEXT:   %0 = arith.xori %x1, %x2 : i1
// CHECK-NEXT:   %1 = arith.andi %x2, %s1 : i1
// CHECK-NEXT:   %2 = arith.xori %z1, %z2 : i1
// CHECK-NEXT:   %3 = arith.xori %1, %2 : i1
// CHECK-NEXT:   %4 = arith.xori %s1, %s2 : i1
// CHECK-NEXT:   %g1 = gate.xzs %0, %3, %4
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g1> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
// CHECK-NEXT: }
func.func @fusion(%q: !qu.bit, %x1: i1, %x2: i1, %z1: i1, %z2: i1, %s1: i1, %s2: i1) -> !qu.bit {
  %g1 = gate.xzs %x1, %z1, %s1
  %q_1 = qssa.dyn_gate<%g1> %q
  %g2 = gate.xzs %x2, %z2, %s2
  %q_2 = qssa.dyn_gate<%g2> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @fusion_backwards(%q : !qu.bit, %x1 : i1, %x2 : i1, %z1 : i1, %z2 : i1, %s1 : i1) -> !qu.bit {
// CHECK-NEXT:   %0 = arith.constant false
// CHECK-NEXT:   %1 = arith.xori %x2, %x1 : i1
// CHECK-NEXT:   %2 = arith.andi %x1, %0 : i1
// CHECK-NEXT:   %3 = arith.xori %z2, %z1 : i1
// CHECK-NEXT:   %4 = arith.xori %2, %3 : i1
// CHECK-NEXT:   %5 = arith.xori %0, %s1 : i1
// CHECK-NEXT:   %g2 = gate.xzs %1, %4, %5
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%g2> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
// CHECK-NEXT: }
func.func @fusion_backwards(%q: !qu.bit, %x1: i1, %x2: i1, %z1: i1, %z2: i1, %s1: i1) -> !qu.bit {
  %g1 = gate.xzs %x1, %z1, %s1
  %0 = arith.constant false
  %g2 = gate.xzs %x2, %z2, %0
  %q_1 = qssa.dyn_gate<%g2> %q
  %q_2 = qssa.dyn_gate<%g1> %q_1
  func.return %q_2 : !qu.bit
}
