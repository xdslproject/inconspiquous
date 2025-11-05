// RUN: quopt -p canonicalize %s --split-input-file | filecheck %s

%a, %b = "test.op"() : () -> (i1, i1)

%false = arith.constant false

// CHECK: gate.xz %a, %b
%g = gate.xzs %a, %b, %false

"test.op"(%g) : (!gate.type<1>) -> ()

// -----

%phi = angle.constant<0.5pi>

// CHECK: gate.constant #gate.j<0.5pi>
%g = gate.dyn_j<%phi>

"test.op"(%g) : (!gate.type<1>) -> ()

// -----

%cTrue = arith.constant true
%cFalse = arith.constant false

// CHECK: gate.constant #gate.cz
%0 = gate.cond<#gate.cz> %cTrue
// CHECK: gate.constant #gate.id<2>
%1 = gate.cond<#gate.cz> %cFalse

"test.op"(%0, %1) : (!gate.type<2>, !gate.type<2>) -> ()
