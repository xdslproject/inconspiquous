// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %{{.*}} = prob.bernoulli 5.000000e-01 : f64
// CHECK-GENERIC: %{{.*}} = "prob.bernoulli"() <{"prob" = 5.000000e-01 : f64}> : () -> i1
%0 = prob.bernoulli 0.5

// CHECK: %{{.*}} = prob.uniform : i32
// CHECK-GENERIC: %{{.*}} = "prob.uniform"() : () -> i32
%1 = prob.uniform : i32
