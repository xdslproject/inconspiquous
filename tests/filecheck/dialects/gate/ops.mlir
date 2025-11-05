// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

"test.op"() {gate = #gate.h} : () -> ()
// CHECK: "test.op"() {gate = #gate.h} : () -> ()

"test.op"() {gate = #gate.x} : () -> ()
// CHECK: "test.op"() {gate = #gate.x} : () -> ()

"test.op"() {gate = #gate.y} : () -> ()
// CHECK: "test.op"() {gate = #gate.y} : () -> ()

"test.op"() {gate = #gate.z} : () -> ()
// CHECK: "test.op"() {gate = #gate.z} : () -> ()

"test.op"() {gate = #gate.s} : () -> ()
// CHECK: "test.op"() {gate = #gate.s} : () -> ()

"test.op"() {gate = #gate.s_dagger} : () -> ()
// CHECK: "test.op"() {gate = #gate.s_dagger} : () -> ()

"test.op"() {gate = #gate.t} : () -> ()
// CHECK: "test.op"() {gate = #gate.t} : () -> ()

"test.op"() {gate = #gate.t_dagger} : () -> ()
// CHECK: "test.op"() {gate = #gate.t_dagger} : () -> ()

"test.op"() {gate = #gate.rx<pi>} : () -> ()
// CHECK-NEXT: "test.op"() {gate = #gate.rx<pi>} : () -> ()

"test.op"() {gate = #gate.ry<pi>} : () -> ()
// CHECK-NEXT: "test.op"() {gate = #gate.ry<pi>} : () -> ()

"test.op"() {gate = #gate.rz<pi>} : () -> ()
// CHECK-NEXT: "test.op"() {gate = #gate.rz<pi>} : () -> ()

"test.op"() {gate = #gate.j<pi>} : () -> ()
// CHECK-NEXT: "test.op"() {gate = #gate.j<pi>} : () -> ()

"test.op"() {gate = #gate.cx} : () -> ()
// CHECK-NEXT: "test.op"() {gate = #gate.cx} : () -> ()

"test.op"() {gate = #gate.cz} : () -> ()
// CHECK-NEXT: "test.op"() {gate = #gate.cz} : () -> ()

"test.op"() {gate = #gate.toffoli} : () -> ()
// CHECK-NEXT: "test.op"() {gate = #gate.toffoli} : () -> ()

"test.op"() {gate = #gate.id<2>} : () -> ()
// CHECK: "test.op"() {gate = #gate.id<2>} : () -> ()


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

// CHECK: %{{.*}} = gate.dyn_rx<%phi>
// CHECK-GENERIC: %{{.*}} = "gate.dyn_rx"(%phi) : (!angle.type) -> !gate.type<1>
%4 = gate.dyn_rx<%phi>

// CHECK: %{{.*}} = gate.dyn_ry<%phi>
// CHECK-GENERIC: %{{.*}} = "gate.dyn_ry"(%phi) : (!angle.type) -> !gate.type<1>
%5 = gate.dyn_ry<%phi>

// CHECK: %{{.*}} = gate.dyn_rz<%phi>
// CHECK-GENERIC: %{{.*}} = "gate.dyn_rz"(%phi) : (!angle.type) -> !gate.type<1>
%6 = gate.dyn_rz<%phi>

// CHECK: %{{.*}} = gate.dyn_j<%phi>
// CHECK-GENERIC: %{{.*}} = "gate.dyn_j"(%phi) : (!angle.type) -> !gate.type<1>
%7 = gate.dyn_j<%phi>

// CHECK: %{{.*}} = gate.control %4 : !gate.type<1>
// CHECK-GENERIC: %{{.*}} = "gate.control"(%4) : (!gate.type<1>) -> !gate.type<2>
%8 = gate.control %4 : !gate.type<1>
