// RUN: QUOPT_ROUNDTRIP

// CHECK: %0 = "test.op"() : () -> !qir.qubit
%0 = "test.op"() : () -> !qir.qubit

// CHECK: %1 = "test.op"() : () -> !qir.result
%1 = "test.op"() : () -> !qir.result
