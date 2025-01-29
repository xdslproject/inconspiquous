// RUN: quopt %s -p convert-to-cme,cse | filecheck %s

// CHECK:      %q1 = qubit.alloc
// CHECK-NEXT: %q1_1 = qubit.alloc<#qubit.plus>
// CHECK-NEXT: %q1_2, %q1_3 = qssa.gate<#gate.cz> %q1, %q1_1
// CHECK-NEXT: %q1_4 = qssa.measure<#measurement.xy<pi>> %q1_2
// CHECK-NEXT: %q1_5 = gate.constant #gate.x
// CHECK-NEXT: %q1_6 = gate.constant #gate.id
// CHECK-NEXT: %q1_7 = arith.select %q1_4, %q1_5, %q1_6 : !gate.type<1>
// CHECK-NEXT: %q1_8 = qssa.dyn_gate<%q1_7> %q1_3
// CHECK-NEXT: %q1_9 = qubit.alloc<#qubit.plus>
// CHECK-NEXT: %q1_10, %q1_11 = qssa.gate<#gate.cz> %q1_8, %q1_9
// CHECK-NEXT: %q1_12 = qssa.measure<#measurement.xy<0>> %q1_10
// CHECK-NEXT: %q1_13 = arith.select %q1_12, %q1_5, %q1_6 : !gate.type<1>
// CHECK-NEXT: %q1_14 = qssa.dyn_gate<%q1_13> %q1_11
%q1 = qubit.alloc

%q1_1 = qssa.gate<#gate.j<pi>> %q1

%q1_2 = qssa.gate<#gate.j<0>> %q1_1
