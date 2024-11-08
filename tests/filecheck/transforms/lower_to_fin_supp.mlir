// RUN: quopt -p lower-to-fin-supp{max_size=2} %s | filecheck %s

// CHECK:    %0 = arith.constant false
// CHECK-NEXT:    %1 = arith.constant true
// CHECK-NEXT:    %2 = prob.fin_supp [ 0.1 of %1, else %0 ] : i1
%0 = prob.bernoulli 0.1 : f64

// CHECK-NEXT:    %3 = arith.constant true
// CHECK-NEXT:    %4 = arith.constant false
// CHECK-NEXT:    %5 = prob.fin_supp [ 0.5 of %3, else %4 ] : i1
%1 = prob.uniform : i1

// CHECK-NEXT:    %6 = arith.constant 1 : i2
// CHECK-NEXT:    %7 = arith.constant 2 : i2
// CHECK-NEXT:    %8 = arith.constant 3 : i2
// CHECK-NEXT:    %9 = arith.constant 0 : i2
// CHECK-NEXT:    %10 = prob.fin_supp [ 0.25 of %6, 0.25 of %7, 0.25 of %8, else %9 ] : i2
%2 = prob.uniform : i2

// CHECK-NEXT:    %11 = prob.uniform : i3
%3 = prob.uniform : i3
