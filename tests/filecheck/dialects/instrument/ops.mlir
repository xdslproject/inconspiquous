// RUN: QUOPT_ROUNDTRIP

// CHECK: "test.op"() : () -> !instrument.type<1>
"test.op"() : () -> !instrument.type<1>

// CHECK: "test.op"() : () -> !instrument.type<1, i1>
"test.op"() : () -> !instrument.type<1, i1>

// CHECK: "test.op"() : () -> !instrument.type<2, i1, i32>
"test.op"() : () -> !instrument.type<2, i1, i32>
