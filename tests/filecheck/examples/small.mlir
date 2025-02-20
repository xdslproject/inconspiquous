// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @small(%q1 : !qubit.bit, %q2 : !qubit.bit) -> (!qubit.bit, !qubit.bit) {
// CHECK-NEXT:   qref.gate<#gate.t> %q1
// CHECK-NEXT:   qref.gate<#gate.cx> %q1, %q2
// CHECK-NEXT:   func.return %q1, %q2 : !qubit.bit, !qubit.bit
// CHECK-NEXT: }
func.func @small(%q1: !qubit.bit, %q2: !qubit.bit) -> (!qubit.bit, !qubit.bit) {
  qref.gate<#gate.t> %q1
  qref.gate<#gate.cx> %q1, %q2
  func.return %q1, %q2 : !qubit.bit, !qubit.bit
}
