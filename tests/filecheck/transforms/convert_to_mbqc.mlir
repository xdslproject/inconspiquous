// RUN: quopt %s -p convert-to-mbqc | filecheck %s

// CHECK:      %q1 = qubit.alloc
// CHECK-NEXT: %q1_1 = qubit.alloc<#qubit.plus>
// CHECK-NEXT: %q1_2 = qubit.alloc<#qubit.plus>
// CHECK-NEXT: %q1_3, %q1_4 = qssa.gate<#gate.cz> %q1, %q1_1
// CHECK-NEXT: %0, %1 = qssa.gate<#gate.cz> %q1_4, %q1_2
// CHECK-NEXT: %q1_5 = qssa.measure<#measurement.xy<pi>> %q1_3
// CHECK-NEXT: %q1_6 = qssa.measure<#measurement.xy<0>> %0
// CHECK-NEXT: %2 = gate.xz %q1_6, %q1_5
// CHECK-NEXT: %q1_7 = qssa.dyn_gate<%2> %1
%q1 = qubit.alloc

%q1_1 = qssa.gate<#gate.j<pi>> %q1

%q1_2 = qssa.gate<#gate.j<0>> %q1_1
