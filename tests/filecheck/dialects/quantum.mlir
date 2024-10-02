// RUN: QUOPT_ROUNDTRIP

"test.op"() {"angle" = #quantum.angle<0>} : () -> ()

// CHECK: "test.op"() {"angle" = #quantum.angle<0>} : () -> ()

"test.op"() {"angle" = #quantum.angle<pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #quantum.angle<pi>} : () -> ()

"test.op"() {"angle" = #quantum.angle<2pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #quantum.angle<0>} : () -> ()

"test.op"() {"angle" = #quantum.angle<0.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #quantum.angle<0.5pi>} : () -> ()

"test.op"() {"angle" = #quantum.angle<1.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #quantum.angle<1.5pi>} : () -> ()

"test.op"() {"angle" = #quantum.angle<2.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #quantum.angle<0.5pi>} : () -> ()

"test.op"() {"angle" = #quantum.angle<-0.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #quantum.angle<1.5pi>} : () -> ()

"test.op"() {"angle" = #quantum.angle<-pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"angle" = #quantum.angle<pi>} : () -> ()

"test.op"() {"gate" = #quantum.h} : () -> ()

// CHECK-NEXT: "test.op"() {"gate" = #quantum.h} : () -> ()

"test.op"() {"gate" = #quantum.rz<pi>} : () -> ()

// CHECK-NEXT: "test.op"() {"gate" = #quantum.rz<pi>} : () -> ()
