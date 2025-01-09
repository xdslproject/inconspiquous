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

// CHECK: qref.gate<#gate.cnot> %q0, %q1
// CHECK-GENERIC: "qref.gate"(%q0, %q1) <{gate = #gate.cnot}> : (!qubit.bit, !qubit.bit)
qref.gate<#gate.cnot> %q0, %q1

%g1 = "test.op"() : () -> !gate.type<1>

// CHECK: qref.dyn_gate<%g1> %q1 : !qubit.bit
// CHECK-GENERIC: "qref.dyn_gate"(%q1, %g1) : (!qubit.bit, !gate.type<1>) -> ()
qref.dyn_gate<%g1> %q1 : !qubit.bit

// CHECK: %{{.*}} = qref.measure %q0
// CHECK-GENERIC: %{{.*}} = "qref.measure"(%q0) : (!qubit.bit) -> i1
%0 = qref.measure %q0
