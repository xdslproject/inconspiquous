// RUN: quopt %s -p convert-to-cme,cse | filecheck %s

// CHECK:      %q1 = qu.alloc
// CHECK-NEXT: %q1_1 = qu.alloc<#qu.plus>
// CHECK-NEXT: %q1_2, %q1_3 = qssa.gate<#gate.cz> %q1, %q1_1
// CHECK-NEXT: %q1_4, %q1_5 = qssa.measure<#measurement.xy<pi>> %q1_2
// CHECK-NEXT: %q1_6 = instrument.constant #gate.x
// CHECK-NEXT: %q1_7 = instrument.constant #gate.id<1>
// CHECK-NEXT: %q1_8 = arith.select %q1_5, %q1_6, %q1_7 : !instrument.type<1>
// CHECK-NEXT: %q1_9 = qssa.dyn_gate<%q1_8> %q1_3
// CHECK-NEXT: %q1_10 = qu.alloc<#qu.plus>
// CHECK-NEXT: %q1_11, %q1_12 = qssa.gate<#gate.cz> %q1_9, %q1_10
// CHECK-NEXT: %q1_13, %q1_14 = qssa.measure<#measurement.xy<0>> %q1_11
// CHECK-NEXT: %q1_15 = arith.select %q1_14, %q1_6, %q1_7 : !instrument.type<1>
// CHECK-NEXT: %q1_16 = qssa.dyn_gate<%q1_15> %q1_12
// CHECK-NEXT: %a = "test.op"() : () -> !angle.type
// CHECK-NEXT: %q1_17 = qu.alloc<#qu.plus>
// CHECK-NEXT: %q1_18, %q1_19 = qssa.gate<#gate.cz> %q1_16, %q1_17
// CHECK-NEXT: %q1_20 = arith.constant true
// CHECK-NEXT: %q1_21 = angle.cond_negate %q1_20, %a
// CHECK-NEXT: %q1_22 = measurement.dyn_xy<%q1_21>
// CHECK-NEXT: %q1_23, %q1_24 = qssa.dyn_measure<%q1_22> %q1_18
// CHECK-NEXT: %q1_25 = arith.select %q1_24, %q1_6, %q1_7 : !instrument.type<1>
// CHECK-NEXT: %q1_26 = qssa.dyn_gate<%q1_25> %q1_19
%q1 = qu.alloc

%q1_1 = qssa.gate<#gate.j<pi>> %q1

%q1_2 = qssa.gate<#gate.j<0>> %q1_1

%a = "test.op"() : () -> !angle.type
%g = gate.dyn_j<%a>
%q1_3 = qssa.dyn_gate<%g> %q1_2
