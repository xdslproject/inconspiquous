// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q0 = qubit.alloc
// CHECK-GENERIC: %q0 = "qubit.alloc"() <{alloc = #qubit.zero}> : () -> !qubit.bit
%q0 = qubit.alloc

// CHECK: %q1 = qubit.alloc
// CHECK-GENERIC: %q1 = "qubit.alloc"() <{alloc = #qubit.zero}> : () -> !qubit.bit
%q1 = qubit.alloc

// CHECK: qref.gate<#gate.h> %q0
// CHECK-GENERIC: "qref.gate"(%q0) <{gate = #gate.h}> : (!qubit.bit) -> ()
qref.gate<#gate.h> %q0

// CHECK: qref.gate<#gate.rz<0.5pi>> %q1
// CHECK-GENERIC: "qref.gate"(%q1) <{gate = #gate.rz<0.5pi>}> : (!qubit.bit) -> ()
qref.gate<#gate.rz<0.5pi>> %q1

// CHECK: qref.gate<#gate.cx> %q0, %q1
// CHECK-GENERIC: "qref.gate"(%q0, %q1) <{gate = #gate.cx}> : (!qubit.bit, !qubit.bit)
qref.gate<#gate.cx> %q0, %q1

%g1 = "test.op"() : () -> !gate.type<1>

// CHECK: qref.dyn_gate<%g1> %q1
// CHECK-GENERIC: "qref.dyn_gate"(%g1, %q1) : (!gate.type<1>, !qubit.bit) -> ()
qref.dyn_gate<%g1> %q1

// CHECK: %{{.*}} = qref.measure %q0
// CHECK-GENERIC: %{{.*}} = "qref.measure"(%q0) <{measurement = #measurement.comp_basis}> : (!qubit.bit) -> i1
%0 = qref.measure %q0

// CHECK: %{{.*}} = qref.measure<#measurement.xy<0.5pi>> %q1
// CHECK-GENERIC: %{{.*}} = "qref.measure"(%q1) <{measurement = #measurement.xy<0.5pi>}> : (!qubit.bit) -> i1
%1 = qref.measure<#measurement.xy<0.5pi>> %q1

%q2 = qubit.alloc

%m = "test.op"() : () -> !measurement.type<1>

// CHECK: %{{.*}} = qref.dyn_measure<%m> %q2
// CHECK-GENERIC: %{{.*}} = "qref.dyn_measure"(%m, %q2) : (!measurement.type<1>, !qubit.bit) -> i1
%2 = qref.dyn_measure<%m> %q2
