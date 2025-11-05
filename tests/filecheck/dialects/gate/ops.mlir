// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

"test.op"() {gate = #gate.h} : () -> ()

// CHECK: "test.op"() {gate = #gate.h} : () -> ()

"test.op"() {gate = #gate.id<2>} : () -> ()

// CHECK: "test.op"() {gate = #gate.id<2>} : () -> ()

"test.op"() {gate = #gate.rz<pi>} : () -> ()

// CHECK-NEXT: "test.op"() {gate = #gate.rz<pi>} : () -> ()

%0 = gate.constant #gate.h

// CHECK-NEXT: %{{.*}} = gate.constant #gate.h
// CHECK-GENERIC: %{{.*}} = "gate.constant"() <{gate = #gate.h}> : () -> !gate.type<1>

%1 = gate.constant #gate.cz

// CHECK-NEXT: %{{.*}} = gate.constant #gate.cz
// CHECK-GENERIC-NEXT: %{{.*}} = "gate.constant"() <{gate = #gate.cz}> : () -> !gate.type<2>

%zero = arith.constant 0 : i64
%one = arith.constant 1 : i64

// CHECK: %{{.*}} = gate.quaternion<i64> %zero + %one i + %zero j + %zero k
// CHECK-GENERIC: %{{.*}} = "gate.quaternion"(%zero, %one, %zero, %zero) : (i64, i64, i64, i64) -> !gate.type<1>
%2 = gate.quaternion<i64> %zero + %one i + %zero j + %zero k

%cTrue = arith.constant true
%cFalse = arith.constant false

// CHECK: %{{.*}} = gate.xzs %cTrue, %cFalse, %cTrue
// CHECK-GENERIC: %{{.*}} = "gate.xzs"(%cTrue, %cFalse, %cTrue) : (i1, i1, i1) -> !gate.type<1>
%3 = gate.xzs %cTrue, %cFalse, %cTrue

%phi = "test.op"() : () -> !angle.type

// CHECK: %{{.*}} = gate.dyn_j<%phi>
// CHECK-GENERIC: %{{.*}} = "gate.dyn_j"(%phi) : (!angle.type) -> !gate.type<1>
%4 = gate.dyn_j<%phi>
