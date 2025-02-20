// RUN: quopt %s -p mbqc-legalize --verify-diagnostics --split-input-file | filecheck %s

%q0 = qubit.alloc
%q1 = qubit.alloc

%q0_1 = qssa.gate<#gate.x> %q0
// CHECK: A CZ gate can only follow allocations and CZ gates in a valid mbqc program
%q0_2, %q1_1 = qssa.gate<#gate.cz> %q0_1, %q1

// -----

%q0 = qubit.alloc
%q1 = qubit.alloc
// CHECK: Expected only CZ or Pauli gates, found #gate.cnot
%q0_1, %q1_1 = qssa.gate<#gate.cnot> %q0, %q1

// -----

%q0 = qubit.alloc
// CHECK: Expected only XY measurements, found #measurement.comp_basis
%0 = qssa.measure %q0

// -----

%q0 = qubit.alloc
%q0_1 = qssa.gate<#gate.x> %q0
// CHECK: A measurement can only follow allocations and CZ gates in a valid mbqc program.
%0 = qssa.measure<#measurement.xy<0>> %q0_1

// -----
%q0 = qubit.alloc
%q0_1 = qssa.gate<#gate.x> %q0
%a = angle.constant<0>
%m = measurement.dyn_xy<%a>
// CHECK: A measurement can only follow allocations and CZ gates in a valid mbqc program.
%0 = qssa.dyn_measure<%m> %q0_1

// -----

// CHECK: Only expected dynamic Pauli gates, found #gate.cz
%g = gate.constant #gate.cz

// -----

%q0 = qubit.alloc
// CHECK: Unexpected operation qref.gate
qref.gate<#gate.x> %q0
