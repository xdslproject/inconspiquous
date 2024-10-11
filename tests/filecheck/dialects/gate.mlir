// RUN: QUOPT_ROUNDTRIP

"test.op"() {"angle" = #gate.angle<0>} : () -> ()

// CHECK: "test.op"() {"angle" = #gate.angle<0>} : () -> ()

"test.op"() {"angle" = #gate.angle<pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #gate.angle<pi>} : () -> ()

"test.op"() {"angle" = #gate.angle<2pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #gate.angle<0>} : () -> ()

"test.op"() {"angle" = #gate.angle<0.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #gate.angle<0.5pi>} : () -> ()

"test.op"() {"angle" = #gate.angle<1.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #gate.angle<1.5pi>} : () -> ()

"test.op"() {"angle" = #gate.angle<2.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #gate.angle<0.5pi>} : () -> ()

"test.op"() {"angle" = #gate.angle<-0.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #gate.angle<1.5pi>} : () -> ()

"test.op"() {"angle" = #gate.angle<-pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #gate.angle<pi>} : () -> ()

"test.op"() {"gate" = #gate.h} : () -> ()

// CHECK-NEXT: "test.op"() {"gate" = #gate.h} : () -> ()

"test.op"() {"gate" = #gate.rz<pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"gate" = #gate.rz<pi>} : () -> ()
