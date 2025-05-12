// RUN: quopt %s -p mbqc-legalize | filecheck %s

// CHECK:      func.func @already_correct(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q1 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q_1, %q1_1 = qssa.gate<#gate.cz> %q, %q1
// CHECK-NEXT:   %0 = qssa.measure<#measurement.xy<pi>> %q
// CHECK-NEXT:   %c0 = arith.constant false
// CHECK-NEXT:   %g = gate.xz %0, %c0
// CHECK-NEXT:   %q1_2 = qssa.dyn_gate<%g> %q1_1
// CHECK-NEXT:   func.return %q1_2 : !qu.bit
// CHECK-NEXT: }
func.func @already_correct(%q : !qu.bit) -> !qu.bit {
  %q1 = qu.alloc<#qu.plus>
  %q_1, %q1_1 = qssa.gate<#gate.cz> %q, %q1

  %0 = qssa.measure<#measurement.xy<pi>> %q
  %c0 = arith.constant false
  %g = gate.xz %0, %c0
  %q1_2 = qssa.dyn_gate<%g> %q1_1
  func.return %q1_2 : !qu.bit
}

// CHECK:      func.func @late_alloc(%q1 : !qu.bit, %q2 : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q3 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
// CHECK-NEXT:   func.return %q3 : !qu.bit
// CHECK-NEXT: }
func.func @late_alloc(%q1 : !qu.bit, %q2 : !qu.bit) -> !qu.bit {
  %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
  %q3 = qu.alloc<#qu.plus>
  func.return %q3 : !qu.bit
}

// CHECK:      func.func @late_cz(%q1 : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q2 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q3 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
// CHECK-NEXT:   %q2_2, %q3_1 = qssa.gate<#gate.cz> %q2_1, %q3
// CHECK-NEXT:   %0 = qssa.measure<#measurement.xy<0>> %q1_1
// CHECK-NEXT:   %1 = qssa.measure<#measurement.xy<0>> %q2_2
// CHECK-NEXT:   func.return %q3_1 : !qu.bit
// CHECK-NEXT: }
func.func @late_cz(%q1 : !qu.bit) -> !qu.bit {
  %q2 = qu.alloc<#qu.plus>
  %q3 = qu.alloc<#qu.plus>
  %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
  %0 = qssa.measure<#measurement.xy<0>> %q1_1
  %q2_2, %q3_1 = qssa.gate<#gate.cz> %q2_1, %q3
  %1 = qssa.measure<#measurement.xy<0>> %q2_2
  func.return %q3_1 : !qu.bit
}

// CHECK:      func.func @correction_before_cz(%q1 : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q2 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q3 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
// CHECK-NEXT:   %q2_2, %q3_1 = qssa.gate<#gate.cz> %q2_1, %q3
// CHECK-NEXT:   %q1_2 = qssa.gate<#gate.x> %q1_1
// CHECK-NEXT:   func.return %q3_1 : !qu.bit
// CHECK-NEXT: }
func.func @correction_before_cz(%q1 : !qu.bit) -> !qu.bit {
  %q2 = qu.alloc<#qu.plus>
  %q3 = qu.alloc<#qu.plus>
  %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
  %q1_2 = qssa.gate<#gate.x> %q1_1
  %q2_2, %q3_1 = qssa.gate<#gate.cz> %q2_1, %q3
  func.return %q3_1 : !qu.bit
}

// CHECK:      func.func @correction_before_measure(%q1 : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q2 = qu.alloc<#qu.plus>
// CHECK-NEXT:   %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
// CHECK-NEXT:   %0 = qssa.measure<#measurement.xy<0>> %q1_1
// CHECK-NEXT:   %q2_2 = qssa.gate<#gate.x> %q2_1
// CHECK-NEXT:   func.return %q2_2 : !qu.bit
// CHECK-NEXT: }
func.func @correction_before_measure(%q1 : !qu.bit) -> !qu.bit {
  %q2 = qu.alloc<#qu.plus>
  %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
  %q2_2 = qssa.gate<#gate.x> %q2_1
  %0 = qssa.measure<#measurement.xy<0>> %q1_1
  func.return %q2_2 : !qu.bit
}
