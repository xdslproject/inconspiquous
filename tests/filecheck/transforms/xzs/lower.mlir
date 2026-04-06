// RUN: quopt -p lower-xzs-to-select %s | filecheck %s

// CHECK:      func.func @xzs(%x : i1, %z : i1, %s : i1) -> !instrument.type<1> {
// CHECK-NEXT:   %[[#id:]] = instrument.constant #gate.id<1>
// CHECK-NEXT:   %[[#gs:]] = instrument.constant #gate.s
// CHECK-NEXT:   %[[#gz:]] = instrument.constant #gate.z
// CHECK-NEXT:   %[[#gy:]] = instrument.constant #gate.y
// CHECK-NEXT:   %[[#gx:]] = instrument.constant #gate.x
// CHECK-NEXT:   %[[#sel_1:]] = arith.select %z, %[[#gz]], %[[#id]] : !instrument.type<1>
// CHECK-NEXT:   %[[#sel_2:]] = arith.select %z, %[[#gy]], %[[#gx]] : !instrument.type<1>
// CHECK-NEXT:   %[[#sel_3:]] = arith.select %x, %[[#sel_2]], %[[#sel_1]] : !instrument.type<1>
// CHECK-NEXT:   %[[#sel_4:]] = arith.select %s, %[[#gs]], %[[#id]] : !instrument.type<1>
// CHECK-NEXT:   %[[#final:]] = gate.compose %[[#sel_3]], %[[#sel_4]] : !instrument.type<1>
// CHECK-NEXT:   func.return %[[#final:]] : !instrument.type<1>
// CHECK-NEXT: }
func.func @xzs(%x: i1, %z: i1, %s: i1) -> !instrument.type<1> {
  %0 = gate.xzs %x, %z, %s
  func.return %0 : !instrument.type<1>
}
