// RUN: quopt -p canonicalize %s --split-input-file | filecheck %s

%a, %b = "test.op"() : () -> (i1, i1)

%false = arith.constant false

// CHECK: gate.xz %a, %b
%g = gate.xzs %a, %b, %false

"test.op"(%g) : (!gate.type<1>) -> ()

// -----

%phi = angle.constant<0.5pi>

// CHECK: gate.constant #gate.rx<0.5pi>
%g = gate.dyn_rx<%phi>

"test.op"(%g) : (!gate.type<1>) -> ()

// -----

%phi = angle.constant<0.5pi>

// CHECK: gate.constant #gate.ry<0.5pi>
%g = gate.dyn_ry<%phi>

"test.op"(%g) : (!gate.type<1>) -> ()

// -----

%phi = angle.constant<0.5pi>

// CHECK: gate.constant #gate.rz<0.5pi>
%g = gate.dyn_rz<%phi>

"test.op"(%g) : (!gate.type<1>) -> ()

// -----

%phi = angle.constant<0.5pi>

// CHECK: gate.constant #gate.j<0.5pi>
%g = gate.dyn_j<%phi>

"test.op"(%g) : (!gate.type<1>) -> ()

// -----

%phi = angle.constant<0.5pi>

// CHECK: gate.constant #gate.rzz<0.5pi>
%g = gate.dyn_rzz<%phi>

"test.op"(%g) : (!gate.type<2>) -> ()
