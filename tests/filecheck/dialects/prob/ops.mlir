// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %{{.*}} = prob.bernoulli 5.000000e-01
// CHECK-GENERIC: %{{.*}} = "prob.bernoulli"() <{prob = 5.000000e-01 : f64}> : () -> i1
%0 = prob.bernoulli 0.5

// CHECK: %{{.*}} = prob.uniform : i32
// CHECK-GENERIC: %{{.*}} = "prob.uniform"() : () -> i32
%1 = prob.uniform : i32

%2, %3, %4 = "test.op"() : () -> (i64, i64, i64)

// CHECK: %{{.*}} = prob.fin_supp [ 1.000000e-01 of %{{.*}}, 2.000000e-01 of %{{.*}}, else %{{.*}} ] : i64
%5 = prob.fin_supp [
  0.1 of %2,
  0.2 of %3,
  else %4
] : i64

// CHECK: %{{.*}} = prob.fin_supp [ %{{.*}} ] : i64
%6 = prob.fin_supp [ %4 ] : i64
