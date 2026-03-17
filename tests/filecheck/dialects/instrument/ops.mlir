// RUN: QUOPT_ROUNDTRIP

// CHECK: "test.op"() : () -> !instrument.type<1>
"test.op"() : () -> !instrument.type<1>

// CHECK: "test.op"() : () -> !instrument.type<1, i1>
"test.op"() : () -> !instrument.type<1, i1>

// CHECK: "test.op"() : () -> !instrument.type<2, i1, i32>
"test.op"() : () -> !instrument.type<2, i1, i32>

// CHECK-NEXT: %{{.*}} = instrument.constant #gate.h
// CHECK-GENERIC: %{{.*}} = "instrument.constant"() <{gate = #gate.h}> : () -> !instrument.type<1>
%0 = instrument.constant #gate.h

// CHECK-NEXT: %{{.*}} = instrument.constant #gate.cz
// CHECK-GENERIC-NEXT: %{{.*}} = "instrument.constant"() <{gate = #gate.cz}> : () -> !instrument.type<2>
%1 = instrument.constant #gate.cz
