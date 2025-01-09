// RUN: quopt -p canonicalize %s | filecheck %s

// CHECK: arith.constant false
%0 = prob.bernoulli 0.0

// CHECK: arith.constant true
%1 = prob.bernoulli 1.0

// Stop them being dead code eliminated
"test.op"(%0, %1) : (i1, i1) -> ()

// CHECK: %[[#x1:]] = "test.op"() {fin_supp_test} : () -> i64
%2 = "test.op"() {fin_supp_test} : () -> i64
%3 = prob.fin_supp [ %2 ] : i64
// CHECK-NEXT: "test.op"(%[[#x1]]) : (i64) -> ()
"test.op"(%3) : (i64) -> ()

// CHECK: %[[#first:]], %[[#second:]], %[[#third:]] = "test.op"() : () -> (i32, i32, i32)
%4, %5, %6 = "test.op"() : () -> (i32, i32, i32)
// CHECK-NEXT: %{{.*}} = prob.fin_supp [ 0.375 of %[[#first]], else %[[#second]] ] : i32
%7 = prob.fin_supp [ 0.125 of %4, 0.25 of %4, else %5 ] : i32

// CHECK-NEXT: %{{.*}} = prob.fin_supp [ 0.1 of %[[#first]], else %[[#third]] ] : i32
%8 = prob.fin_supp [ 0.1 of %4, 0.0 of %5, else %6 ] : i32

// CHECK-NEXT: %{{.*}} = prob.fin_supp [ 0.2 of %[[#second]], else %[[#first]] ] : i32
%9 = prob.fin_supp [ 0.1 of %4, 0.2 of %5, else %4 ] : i32

"test.op"(%7, %8, %9) : (i32, i32, i32) -> ()
