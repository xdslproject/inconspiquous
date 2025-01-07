// RUN: quopt %s -p xzs-select,dce | filecheck %s

// CHECK:      func.func @select(%b : i1, %x1 : i1, %x2 : i1, %z1 : i1, %z2 : i1, %s1 : i1, %s2 : i1) -> !gate.type<1> {
// CHECK-NEXT:   %g3 = arith.select %b, %x1, %x2 : i1
// CHECK-NEXT:   %g3_1 = arith.select %b, %z1, %z2 : i1
// CHECK-NEXT:   %g3_2 = arith.select %b, %s1, %s2 : i1
// CHECK-NEXT:   %g3_3 = gate.xzs %g3, %g3_1, %g3_2
// CHECK-NEXT:   func.return %g3_3 : !gate.type<1>
// CHECK-NEXT: }
func.func @select(%b: i1, %x1: i1, %x2: i1, %z1: i1, %z2: i1, %s1: i1, %s2: i1) -> !gate.type<1> {
  %g1 = gate.xzs %x1, %z1, %s1
  %g2 = gate.xzs %x2, %z2, %s2
  %g3 = arith.select %b, %g1, %g2 : !gate.type<1>
  func.return %g3 : !gate.type<1>
}
