// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @toffoli(%q1 : !qubit.bit, %q2 : !qubit.bit, %q3 : !qubit.bit) -> (!qubit.bit, !qubit.bit, !qubit.bit) {
// CHECK-NEXT:   qref.gate<#gate.h> %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %q2, %q3
// CHECK-NEXT:   qref.gate<#gate.t_dagger> %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %q1, %q3
// CHECK-NEXT:   qref.gate<#gate.t> %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %q2, %q3
// CHECK-NEXT:   qref.gate<#gate.t> %q2
// CHECK-NEXT:   qref.gate<#gate.t> %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %q1, %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %q1, %q2
// CHECK-NEXT:   qref.gate<#gate.t> %q1
// CHECK-NEXT:   qref.gate<#gate.t_dagger> %q2
// CHECK-NEXT:   qref.gate<#gate.t> %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %q1, %q2
// CHECK-NEXT:   qref.gate<#gate.s> %q2
// CHECK-NEXT:   qref.gate<#gate.h> %q3
// CHECK-NEXT:   func.return %q1, %q2, %q3 : !qubit.bit, !qubit.bit, !qubit.bit
// CHECK-NEXT: }
func.func @toffoli(%q1 : !qubit.bit, %q2 : !qubit.bit, %q3 : !qubit.bit) -> (!qubit.bit, !qubit.bit, !qubit.bit) {
  qref.gate<#gate.h> %q3
  qref.gate<#gate.cx> %q2, %q3
  qref.gate<#gate.t_dagger> %q3
  qref.gate<#gate.cx> %q1, %q3
  qref.gate<#gate.t> %q3
  qref.gate<#gate.cx> %q2, %q3
  qref.gate<#gate.t> %q2
  qref.gate<#gate.t> %q3
  qref.gate<#gate.cx> %q1, %q3
  qref.gate<#gate.cx> %q1, %q2
  qref.gate<#gate.t> %q1
  qref.gate<#gate.t_dagger> %q2
  qref.gate<#gate.t> %q3
  qref.gate<#gate.cx> %q1, %q2
  qref.gate<#gate.s> %q2
  qref.gate<#gate.h> %q3
  func.return %q1, %q2, %q3 : !qubit.bit, !qubit.bit, !qubit.bit
}
