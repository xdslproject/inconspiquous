// RUN: quopt %s -p convert-to-mbqc | filecheck %s

// CHECK:      func.func @rotation(%phi : !angle.type, %theta : !angle.type, %lambda : !angle.type, %q1 : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q1_1 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   %q1_2 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   %q1_3 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   %q1_4 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   %q1_5, %q1_6 = qssa.gate<#gate.cz> %q1, %q1_1
// CHECK-NEXT:   %0, %1 = qssa.gate<#gate.cz> %q1_6, %q1_2
// CHECK-NEXT:   %2, %3 = qssa.gate<#gate.cz> %1, %q1_3
// CHECK-NEXT:   %4, %5 = qssa.gate<#gate.cz> %3, %q1_4
// CHECK-NEXT:   %q1_7 = qssa.measure<#measurement.xy<0>> %q1_5
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %q1_8 = arith.xori %q1_7, %cTrue : i1
// CHECK-NEXT:   %q1_9 = angle.cond_negate %q1_8, %lambda
// CHECK-NEXT:   %q1_10 = measurement.dyn_xy<%q1_9>
// CHECK-NEXT:   %q1_11 = qssa.dyn_measure<%q1_10> %0
// CHECK-NEXT:   %q1_12 = arith.xori %q1_11, %cTrue : i1
// CHECK-NEXT:   %q1_13 = angle.cond_negate %q1_12, %theta
// CHECK-NEXT:   %q1_14 = measurement.dyn_xy<%q1_13>
// CHECK-NEXT:   %q1_15 = qssa.dyn_measure<%q1_14> %2
// CHECK-NEXT:   %q1_16 = arith.xori %q1_15, %q1_7 : i1
// CHECK-NEXT:   %q1_17 = arith.xori %q1_16, %cTrue : i1
// CHECK-NEXT:   %q1_18 = angle.cond_negate %q1_17, %phi
// CHECK-NEXT:   %q1_19 = measurement.dyn_xy<%q1_18>
// CHECK-NEXT:   %q1_20 = qssa.dyn_measure<%q1_19> %4
// CHECK-NEXT:   %q1_21 = arith.xori %q1_20, %q1_11 : i1
// CHECK-NEXT:   %6 = gate.xz %q1_21, %q1_16
// CHECK-NEXT:   %q1_22 = qssa.dyn_gate<%6> %5
// CHECK-NEXT:   func.return %q1_22 : !qubit.bit
// CHECK-NEXT: }
func.func @rotation(%phi: !angle.type, %theta: !angle.type, %lambda: !angle.type, %q1: !qubit.bit) -> !qubit.bit {
  %q1_1 = qssa.gate<#gate.j<0>> %q1
  %g1 = gate.dyn_j<%lambda>
  %q1_2 = qssa.dyn_gate<%g1> %q1_1
  %g2 = gate.dyn_j<%theta>
  %q1_3 = qssa.dyn_gate<%g2> %q1_2
  %g3 = gate.dyn_j<%phi>
  %q1_4 = qssa.dyn_gate<%g3> %q1_3
  func.return %q1_4 : !qubit.bit
}
