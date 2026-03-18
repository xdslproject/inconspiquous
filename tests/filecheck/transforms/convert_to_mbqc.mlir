// RUN: quopt %s -p convert-to-mbqc | filecheck %s

// CHECK:      func.func @rotation(%phi : !angle.type, %theta : !angle.type, %lambda : !angle.type, %q1 : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q1_1 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_2 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_3 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_4 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_5, %q1_6 = qssa.gate<#gate.cz> %q1, %q1_1
// CHECK-NEXT:   %0, %1 = qssa.gate<#gate.cz> %q1_6, %q1_2
// CHECK-NEXT:   %2, %3 = qssa.gate<#gate.cz> %1, %q1_3
// CHECK-NEXT:   %4, %5 = qssa.gate<#gate.cz> %3, %q1_4
// CHECK-NEXT:   %q1_7, %q1_8 = qssa.measure<#measurement.xy<0>> %q1_5
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %6 = arith.xori %q1_7, %cTrue : i1
// CHECK-NEXT:   %7 = angle.cond_negate %6, %lambda
// CHECK-NEXT:   %8 = measurement.dyn_xy<%7>
// CHECK-NEXT:   %q1_9, %q1_10 = qssa.dyn_measure<%8> %0
// CHECK-NEXT:   %9 = arith.xori %q1_9, %cTrue : i1
// CHECK-NEXT:   %10 = angle.cond_negate %9, %theta
// CHECK-NEXT:   %11 = measurement.dyn_xy<%10>
// CHECK-NEXT:   %12, %q1_11 = qssa.dyn_measure<%11> %2
// CHECK-NEXT:   %q1_12 = arith.xori %12, %q1_7 : i1
// CHECK-NEXT:   %13 = arith.xori %q1_12, %cTrue : i1
// CHECK-NEXT:   %14 = angle.cond_negate %13, %phi
// CHECK-NEXT:   %15 = measurement.dyn_xy<%14>
// CHECK-NEXT:   %16, %q1_13 = qssa.dyn_measure<%15> %4
// CHECK-NEXT:   %q1_14 = arith.xori %16, %q1_9 : i1
// CHECK-NEXT:   %17 = gate.xz %q1_14, %q1_12
// CHECK-NEXT:   %q1_15 = qssa.dyn_gate<%17> %5
// CHECK-NEXT:   func.return %q1_15 : !qu.bit
// CHECK-NEXT: }
func.func @rotation(%phi: !angle.type, %theta: !angle.type, %lambda: !angle.type, %q1: !qu.bit) -> !qu.bit {
  %q1_1 = qssa.gate<#gate.j<0>> %q1
  %g1 = gate.dyn_j<%lambda>
  %q1_2 = qssa.dyn_gate<%g1> %q1_1
  %g2 = gate.dyn_j<%theta>
  %q1_3 = qssa.dyn_gate<%g2> %q1_2
  %g3 = gate.dyn_j<%phi>
  %q1_4 = qssa.dyn_gate<%g3> %q1_3
  func.return %q1_4 : !qu.bit
}
