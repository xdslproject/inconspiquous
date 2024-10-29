// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

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

%0 = gate.constant #gate.h

// CHECK-NEXT: %{{.*}} = gate.constant #gate.h
// CHECK-GENERIC: %{{.*}} = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>

%1 = gate.constant #gate.cz

// CHECK-NEXT: %{{.*}} = gate.constant #gate.cz
// CHECK-GENERIC-NEXT: %{{.*}} = "gate.constant"() <{"gate" = #gate.cz}> : () -> !gate.type<2>

%zero = arith.constant 0 : i64
%one = arith.constant 1 : i64

// CHECK: %{{.*}} = gate.quaternion<i64> %zero + %one i + %zero j + %zero k
// CHECK-GENERIC: %{{.*}} = "gate.quaternion"(%zero, %one, %zero, %zero) : (i64, i64, i64, i64) -> !gate.type<1>
%2 = gate.quaternion<i64> %zero + %one i + %zero j + %zero k
