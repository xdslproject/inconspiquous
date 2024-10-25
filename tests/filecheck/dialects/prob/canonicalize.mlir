// RUN: quopt -p canonicalize %s | filecheck %s

// CHECK: arith.constant false
%0 = prob.bernoulli 0.0

// CHECK: arith.constant true
%1 = prob.bernoulli 1.0

// Stop them being dead code eliminated
"test.op"(%0, %1) : (i1, i1) -> ()
