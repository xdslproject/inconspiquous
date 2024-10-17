// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q0 = qubit.alloc
// CHECK-GENERIC: %q0 = "qubit.alloc"() <{"alloc" = #qubit.zero}> : () -> !qubit.bit
%q0 = qubit.alloc

// CHECK: %q1 = qubit.alloc
// CHECK-GENERIC: %q1 = "qubit.alloc"() <{"alloc" = #qubit.zero}> : () -> !qubit.bit
%q1 = qubit.alloc

// CHECK: %q2 = qssa.gate<#gate.h> %q0 : !qubit.bit
// CHECK-GENERIC: %q2 = "qssa.gate"(%q0) <{"gate" = #gate.h}> : (!qubit.bit) -> !qubit.bit
%q2 = qssa.gate<#gate.h> %q0 : !qubit.bit

// CHECK: %q3 = qssa.gate<#gate.rz<0.5pi>> %q1
// CHECK-GENERIC: %q3 = "qssa.gate"(%q1) <{"gate" = #gate.rz<0.5pi>}> : (!qubit.bit) -> !qubit.bit
%q3 = qssa.gate<#gate.rz<0.5pi>> %q1 : !qubit.bit

// CHECK: %q4, %q5 = qssa.gate<#gate.cnot> %q2, %q3
// CHECK-GENERIC: %q4, %q5 = "qssa.gate"(%q2, %q3) <{"gate" = #gate.cnot}> : (!qubit.bit, !qubit.bit) -> (!qubit.bit, !qubit.bit)
%q4, %q5 = qssa.gate<#gate.cnot> %q2, %q3 : !qubit.bit , !qubit.bit

// CHECK: %{{.*}}, %q6 = qssa.measure %q4
// CHECK-GENERIC: %{{.*}}, %q6 = "qssa.measure"(%q4) : (!qubit.bit) -> (i1, !qubit.bit)
%0, %q6 = qssa.measure %q4
