// RUN: quopt -p convert-qssa-to-qref %s | filecheck %s
// RUN: quopt -p convert-qssa-to-qref,convert-qref-to-qssa %s | filecheck %s --check-prefix=CHECK-ROUNDTRIP

%q0 = qubit.alloc
%q1 = qubit.alloc
%q2 = qssa.gate<#gate.h> %q0
%q3 = qssa.gate<#gate.rz<0.5pi>> %q1
%q4, %q5 = qssa.gate<#gate.cnot> %q2, %q3
%0, %q6 = qssa.measure %q4
%g = gate.constant #gate.h
%q7 = qssa.dyn_gate<%g> %q6 : !qubit.bit
%1, %q8 = qssa.measure %q7

// CHECK:      %q0 = qubit.alloc
// CHECK-NEXT: %q1 = qubit.alloc
// CHECK-NEXT: qref.gate<#gate.h> %q0
// CHECK-NEXT: qref.gate<#gate.rz<0.5pi>> %q1
// CHECK-NEXT: qref.gate<#gate.cnot> %q0, %q1
// CHECK-NEXT: %{{.*}} = qref.measure %q0
// CHECK-NEXT: %g = gate.constant #gate.h
// CHECK-NEXT: qref.dyn_gate<%g> %q0 : !qubit.bit
// CHECK-NEXT: %{{.*}} = qref.measure %q0

// CHECK-ROUNDTRIP:      %q0 = qubit.alloc
// CHECK-ROUNDTRIP-NEXT: %q1 = qubit.alloc
// CHECK-ROUNDTRIP-NEXT: %q0_1 = qssa.gate<#gate.h> %q0
// CHECK-ROUNDTRIP-NEXT: %q1_1 = qssa.gate<#gate.rz<0.5pi>> %q1
// CHECK-ROUNDTRIP-NEXT: %q0_2, %q1_2 = qssa.gate<#gate.cnot> %q0_1, %q1_1
// CHECK-ROUNDTRIP-NEXT: %{{.*}}, %q0_3 = qssa.measure %q0_2
// CHECK-ROUNDTRIP-NEXT: %g = gate.constant #gate.h
// CHECK-ROUNDTRIP-NEXT: %q0_4 = qssa.dyn_gate<%g> %q0_3 : !qubit.bit
// CHECK-ROUNDTRIP-NEXT: %{{.*}}, %q0_5 = qssa.measure %q0_4
