// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @rotation(%phi : !angle.type, %theta : !angle.type, %lambda : !angle.type, %q1 : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q2 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   %q3 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   %q4 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   %q5 = qubit.alloc<#qubit.plus>
// CHECK-NEXT:   qref.gate<#gate.cz> %q1, %q2
// CHECK-NEXT:   qref.gate<#gate.cz> %q2, %q3
// CHECK-NEXT:   qref.gate<#gate.cz> %q3, %q4
// CHECK-NEXT:   qref.gate<#gate.cz> %q4, %q5
// CHECK-NEXT:   %0 = qref.measure<#measurement.xy<0>> %q1
// CHECK-NEXT:   %a2 = angle.cond_negate %0, %lambda
// CHECK-NEXT:   %m2 = measurement.dyn_xy<%a2>
// CHECK-NEXT:   %1 = qref.dyn_measure<%m2> %q2
// CHECK-NEXT:   %a3 = angle.cond_negate %1, %theta
// CHECK-NEXT:   %m3 = measurement.dyn_xy<%a3>
// CHECK-NEXT:   %2 = qref.dyn_measure<%m3> %q3
// CHECK-NEXT:   %z = arith.xori %0, %2 : i1
// CHECK-NEXT:   %a4 = angle.cond_negate %z, %phi
// CHECK-NEXT:   %m4 = measurement.dyn_xy<%a4>
// CHECK-NEXT:   %3 = qref.dyn_measure<%m4> %q4
// CHECK-NEXT:   %x = arith.xori %1, %3 : i1
// CHECK-NEXT:   %g = gate.xz %x, %z
// CHECK-NEXT:   qref.dyn_gate<%g> %q5
// CHECK-NEXT:   func.return %q5 : !qubit.bit
// CHECK-NEXT: }
func.func @rotation(%phi: !angle.type, %theta: !angle.type, %lambda: !angle.type, %q1: !qubit.bit) -> !qubit.bit {
  %q2 = qubit.alloc<#qubit.plus>
  %q3 = qubit.alloc<#qubit.plus>
  %q4 = qubit.alloc<#qubit.plus>
  %q5 = qubit.alloc<#qubit.plus>
  qref.gate<#gate.cz> %q1, %q2
  qref.gate<#gate.cz> %q2, %q3
  qref.gate<#gate.cz> %q3, %q4
  qref.gate<#gate.cz> %q4, %q5
  %1 = qref.measure<#measurement.xy<0>> %q1
  %a2 = angle.cond_negate %1, %lambda
  %m2 = measurement.dyn_xy<%a2>
  %2 = qref.dyn_measure<%m2> %q2
  %a3 = angle.cond_negate %2, %theta
  %m3 = measurement.dyn_xy<%a3>
  %3 = qref.dyn_measure<%m3> %q3
  %z = arith.xori %1, %3 : i1
  %a4 = angle.cond_negate %z, %phi
  %m4 = measurement.dyn_xy<%a4>
  %4 = qref.dyn_measure<%m4> %q4
  %x = arith.xori %2, %4 : i1
  %g = gate.xz %x, %z
  qref.dyn_gate<%g> %q5
  func.return %q5 : !qubit.bit
}
