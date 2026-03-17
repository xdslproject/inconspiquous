// RUN: quopt -p convert-qssa-to-qref %s | filecheck %s
// RUN: quopt -p convert-qssa-to-qref,convert-qref-to-qssa %s | filecheck %s --check-prefix=CHECK-ROUNDTRIP

%q0 = qu.alloc
%q1 = qu.alloc
%q2 = qssa.gate<#gate.h> %q0
%q3 = qssa.gate<#gate.rz<0.5pi>> %q1
%q4, %q5 = qssa.gate<#gate.cx> %q2, %q3
%q6, %0 = qssa.measure %q4
%g = instrument.constant #gate.h
%q7 = qssa.dyn_gate<%g> %q6
%m = instrument.constant #measurement.comp_basis
%q8, %1 = qssa.dyn_measure<%m> %q5

// CHECK:      %q0 = qu.alloc
// CHECK-NEXT: %q1 = qu.alloc
// CHECK-NEXT: qref.gate<#gate.h> %q0
// CHECK-NEXT: qref.gate<#gate.rz<0.5pi>> %q1
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q1
// CHECK-NEXT: %{{.*}} = qref.measure %q0
// CHECK-NEXT: %g = instrument.constant #gate.h
// CHECK-NEXT: qref.dyn_gate<%g> %q0
// CHECK-NEXT: %m = instrument.constant #measurement.comp_basis
// CHECK-NEXT: %{{.*}} = qref.dyn_measure<%m> %q1

// CHECK-ROUNDTRIP:      %q0 = qu.alloc
// CHECK-ROUNDTRIP-NEXT: %q1 = qu.alloc
// CHECK-ROUNDTRIP-NEXT: %q0_1 = qssa.gate<#gate.h> %q0
// CHECK-ROUNDTRIP-NEXT: %q1_1 = qssa.gate<#gate.rz<0.5pi>> %q1
// CHECK-ROUNDTRIP-NEXT: %q0_2, %q1_2 = qssa.gate<#gate.cx> %q0_1, %q1_1
// CHECK-ROUNDTRIP-NEXT: %q0_3, %{{.*}} = qssa.measure %q0_2
// CHECK-ROUNDTRIP-NEXT: %g = instrument.constant #gate.h
// CHECK-ROUNDTRIP-NEXT: %q0_4 = qssa.dyn_gate<%g> %q0_3
// CHECK-ROUNDTRIP-NEXT: %m = instrument.constant #measurement.comp_basis
// CHECK-ROUNDTRIP-NEXT: %{{.*}} = qssa.dyn_measure<%m> %q1_2
