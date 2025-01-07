// RUN: quopt -p lower-xzs-to-select %s | filecheck %s

// CHECK:      func.func @xzs(%x : i1, %z : i1, %s : i1) -> !gate.type<1> {
// CHECK-NEXT:   %[[#id:]] = gate.constant #gate.id
// CHECK-NEXT:   %[[#gs:]] = gate.constant #gate.s
// CHECK-NEXT:   %[[#gz:]] = gate.constant #gate.z
// CHECK-NEXT:   %[[#gy:]] = gate.constant #gate.y
// CHECK-NEXT:   %[[#gx:]] = gate.constant #gate.x
// CHECK-NEXT:   %[[#sel_1:]] = arith.select %z, %[[#gz]], %[[#id]] : !gate.type<1>
// CHECK-NEXT:   %[[#sel_2:]] = arith.select %z, %[[#gy]], %[[#gx]] : !gate.type<1>
// CHECK-NEXT:   %[[#sel_3:]] = arith.select %x, %[[#sel_2]], %[[#sel_1]] : !gate.type<1>
// CHECK-NEXT:   %[[#sel_4:]] = arith.select %s, %[[#gs]], %[[#id]] : !gate.type<1>
// CHECK-NEXT:   %[[#final:]] = gate.compose %[[#sel_3]], %[[#sel_4]] : !gate.type<1>
// CHECK-NEXT:   func.return %[[#final:]] : !gate.type<1>
// CHECK-NEXT: }
func.func @xzs(%x: i1, %z: i1, %s: i1) -> !gate.type<1> {
  %0 = gate.xzs %x, %z, %s
  func.return %0 : !gate.type<1>
}
