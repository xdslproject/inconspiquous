// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q0 = qssa.alloc
// CHECK-GENERIC: %q0 = "qssa.alloc"() : () -> !qssa.qubit
%q0 = qssa.alloc

// CHECK: %q1 = qssa.alloc
// CHECK-GENERIC: %q1 = "qssa.alloc"() : () -> !qssa.qubit
%q1 = qssa.alloc

// CHECK: %q2 = qssa.gate<#quantum.h> %q0
// CHECK-GENERIC: %q2 = "qssa.gate"(%q0) <{"gate" = #quantum.h}> : (!qssa.qubit) -> !qssa.qubit
%q2 = qssa.gate<#quantum.h> %q0

// CHECK: %q3 = qssa.gate<#quantum.rz<0.5pi>> %q1
// CHECK-GENERIC: %q3 = "qssa.gate"(%q1) <{"gate" = #quantum.rz<0.5pi>}> : (!qssa.qubit) -> !qssa.qubit
%q3 = qssa.gate<#quantum.rz<0.5pi>> %q1

// CHECK: %q4, %q5 = qssa.gate<#quantum.cnot> %q2, %q3
// CHECK-GENERIC: %q4, %q5 = "qssa.gate"(%q2, %q3) <{"gate" = #quantum.cnot}> : (!qssa.qubit, !qssa.qubit) -> (!qssa.qubit, !qssa.qubit)
%q4, %q5 = qssa.gate<#quantum.cnot> %q2, %q3

// CHECK: %{{.*}} = qssa.measure %q4
// CHECK-GENERIC: %{{.*}} = "qssa.measure"(%q4) : (!qssa.qubit) -> i1
%0 = qssa.measure %q4
