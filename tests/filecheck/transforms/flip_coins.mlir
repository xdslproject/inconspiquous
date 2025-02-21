// RUN: quopt %s -p flip-coins{seed=1234} | filecheck %s

// CHECK: [[v0:%.+]] = arith.constant false
%0 = prob.bernoulli 0.1

// CHECK: [[v1:%.+]] = arith.constant false
%1 = prob.bernoulli 0.2

// CHECK: [[v2:%.+]] = arith.constant true
%2 = prob.bernoulli 0.3

// CHECK: [[v3:%.+]] = arith.constant false
%3 = prob.bernoulli 0.4

// CHECK: [[v4:%.+]] = arith.constant false
%4 = prob.bernoulli 0.5

// CHECK: [[v5:%.+]] = arith.constant true
%5 = prob.bernoulli 0.6

// CHECK: [[v6:%.+]] = arith.constant true
%6 = prob.bernoulli 0.7

// CHECK: [[v7:%.+]] = arith.constant -1786971706 : i32
%7 = prob.uniform i32

// CHECK: [[v8:%.+]] = arith.constant 1144526902 : i32
%8 = prob.uniform i32

// CHECK: [[v9:%.+]] = arith.constant -1130393399 : i32
%9 = prob.uniform i32

// CHECK: [[v10:%.+]] = arith.constant -2015138464 : i32
%10 = prob.uniform i32

// CHECK: "test.op"([[v0]], [[v1]], [[v2]], [[v3]], [[v4]], [[v5]], [[v6]], [[v7]], [[v8]], [[v9]], [[v10]]) : (i1, i1, i1, i1, i1, i1, i1, i32, i32, i32, i32) -> ()
"test.op"(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10) : (i1, i1, i1, i1, i1, i1, i1, i32, i32, i32, i32) -> ()
