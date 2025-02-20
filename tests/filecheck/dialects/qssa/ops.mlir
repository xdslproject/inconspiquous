// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q0 = qubit.alloc
// CHECK-GENERIC: %q0 = "qubit.alloc"() <{alloc = #qubit.zero}> : () -> !qubit.bit
%q0 = qubit.alloc

// CHECK: %q1 = qubit.alloc
// CHECK-GENERIC: %q1 = "qubit.alloc"() <{alloc = #qubit.zero}> : () -> !qubit.bit
%q1 = qubit.alloc

// CHECK: %q2 = qssa.gate<#gate.h> %q0
// CHECK-GENERIC: %q2 = "qssa.gate"(%q0) <{gate = #gate.h}> : (!qubit.bit) -> !qubit.bit
%q2 = qssa.gate<#gate.h> %q0

// CHECK: %q3 = qssa.gate<#gate.rz<0.5pi>> %q1
// CHECK-GENERIC: %q3 = "qssa.gate"(%q1) <{gate = #gate.rz<0.5pi>}> : (!qubit.bit) -> !qubit.bit
%q3 = qssa.gate<#gate.rz<0.5pi>> %q1

// CHECK: %q4, %q5 = qssa.gate<#gate.cx> %q2, %q3
// CHECK-GENERIC: %q4, %q5 = "qssa.gate"(%q2, %q3) <{gate = #gate.cx}> : (!qubit.bit, !qubit.bit) -> (!qubit.bit, !qubit.bit)
%q4, %q5 = qssa.gate<#gate.cx> %q2, %q3

%g1 = "test.op"() : () -> !gate.type<1>

// CHECK: %q6 = qssa.dyn_gate<%g1> %q5
// CHECK-GENERIC: %q6 = "qssa.dyn_gate"(%g1, %q5) : (!gate.type<1>, !qubit.bit) -> !qubit.bit
%q6 = qssa.dyn_gate<%g1> %q5

// CHECK: %{{.*}} = qssa.measure %q4
// CHECK-GENERIC: %{{.*}} = "qssa.measure"(%q4) <{measurement = #measurement.comp_basis}> : (!qubit.bit) -> i1
%0 = qssa.measure %q4

// CHECK: %{{.*}} = qssa.measure<#measurement.xy<0.5pi>> %q6
// CHECK-GENERIC: %{{.*}} = "qssa.measure"(%q6) <{measurement = #measurement.xy<0.5pi>}> : (!qubit.bit) -> i1
%1 = qssa.measure<#measurement.xy<0.5pi>> %q6

%q7 = qubit.alloc

%m = "test.op"() : () -> !measurement.type<1>

// CHECK: %{{.*}} = qssa.dyn_measure<%m> %q7
// CHECK-GENERIC: %{{.*}} = "qssa.dyn_measure"(%m, %q7) : (!measurement.type<1>, !qubit.bit) -> i1
%2 = qssa.dyn_measure<%m> %q7
