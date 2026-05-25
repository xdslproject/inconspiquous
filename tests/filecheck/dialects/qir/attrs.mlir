// RUN: QUOPT_ROUNDTRIP

// CHECK: %0 = "test.op"() : () -> !qir.qubit
%0 = "test.op"() : () -> !qir.qubit

// CHECK: %1 = "test.op"() : () -> !qir.result
%1 = "test.op"() : () -> !qir.result

// CHECK: %2 = "test.op"() : () -> !qir.array
%2 = "test.op"() : () -> !qir.array
