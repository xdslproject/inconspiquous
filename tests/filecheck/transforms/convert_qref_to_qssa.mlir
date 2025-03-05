// RUN: quopt -p convert-qref-to-qssa %s | filecheck %s
// RUN: quopt -p convert-qref-to-qssa,convert-qssa-to-qref %s | filecheck %s --check-prefix=CHECK-ROUNDTRIP

%q0 = qubit.alloc
%q1 = qubit.alloc
qref.gate<#gate.h> %q0
qref.gate<#gate.rz<0.5pi>> %q1
qref.gate<#gate.cx> %q0, %q1
%0 = qref.measure %q0
%g = gate.constant #gate.h
qref.dyn_gate<%g> %q1
%m = measurement.constant #measurement.comp_basis
%1 = qref.dyn_measure<%m> %q1

// CHECK:      %q0 = qubit.alloc
// CHECK-NEXT: %q1 = qubit.alloc
// CHECK-NEXT: %q0_1 = qssa.gate<#gate.h> %q0
// CHECK-NEXT: %q1_1 = qssa.gate<#gate.rz<0.5pi>> %q1
// CHECK-NEXT: %q0_2, %q1_2 = qssa.gate<#gate.cx> %q0_1, %q1_1
// CHECK-NEXT: %{{.*}} = qssa.measure %q0_2
// CHECK-NEXT: %g = gate.constant #gate.h
// CHECK-NEXT: %q1_3 = qssa.dyn_gate<%g> %q1_2
// CHECK-NEXT: %m = measurement.constant #measurement.comp_basis
// CHECK-NEXT: %1 = qssa.dyn_measure<%m> %q1_3

// CHECK-ROUNDTRIP:      %q0 = qubit.alloc
// CHECK-ROUNDTRIP-NEXT: %q1 = qubit.alloc
// CHECK-ROUNDTRIP-NEXT: qref.gate<#gate.h> %q0
// CHECK-ROUNDTRIP-NEXT: qref.gate<#gate.rz<0.5pi>> %q1
// CHECK-ROUNDTRIP-NEXT: qref.gate<#gate.cx> %q0, %q1
// CHECK-ROUNDTRIP-NEXT: %{{.*}} = qref.measure %q0
// CHECK-ROUNDTRIP-NEXT: %g = gate.constant #gate.h
// CHECK-ROUNDTRIP-NEXT: qref.dyn_gate<%g> %q1
// CHECK-ROUNDTRIP-NEXT: %m = measurement.constant #measurement.comp_basis
// CHECK-ROUNDTRIP-NEXT: %{{.*}} = qref.dyn_measure<%m> %q1

func.func @qref_in_region(%q : !qubit.bit, %p: i1) -> !qubit.bit {
  %q2 = scf.if %p -> (!qubit.bit) {
    qref.gate<#gate.z> %q
    scf.yield %q : !qubit.bit
  } else {
    scf.yield %q : !qubit.bit
  }
  func.return %q2 : !qubit.bit
}

// CHECK:      func.func @qref_in_region
// CHECK-NEXT: %q2 = scf.if %p -> (!qubit.bit) {
// CHECK-NEXT:   qref.gate<#gate.z> %q
// CHECK-NEXT:   scf.yield %q : !qubit.bit
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %q : !qubit.bit
// CHECK-NEXT: }
// CHECK-NEXT: func.return %q2 : !qubit.bit
