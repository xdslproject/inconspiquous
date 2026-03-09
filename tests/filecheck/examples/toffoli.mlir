// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @toffoli(%q1 : !qu.bit, %q2 : !qu.bit, %q3 : !qu.bit) -> (!qu.bit, !qu.bit, !qu.bit) {
// CHECK-NEXT:   qref.apply<#gate.h> %q3
// CHECK-NEXT:   qref.apply<#gate.cx> %q2, %q3
// CHECK-NEXT:   qref.apply<#gate.t_dagger> %q3
// CHECK-NEXT:   qref.apply<#gate.cx> %q1, %q3
// CHECK-NEXT:   qref.apply<#gate.t> %q3
// CHECK-NEXT:   qref.apply<#gate.cx> %q2, %q3
// CHECK-NEXT:   qref.apply<#gate.t> %q2
// CHECK-NEXT:   qref.apply<#gate.t> %q3
// CHECK-NEXT:   qref.apply<#gate.cx> %q1, %q3
// CHECK-NEXT:   qref.apply<#gate.cx> %q1, %q2
// CHECK-NEXT:   qref.apply<#gate.t> %q1
// CHECK-NEXT:   qref.apply<#gate.t_dagger> %q2
// CHECK-NEXT:   qref.apply<#gate.t> %q3
// CHECK-NEXT:   qref.apply<#gate.cx> %q1, %q2
// CHECK-NEXT:   qref.apply<#gate.s> %q2
// CHECK-NEXT:   qref.apply<#gate.h> %q3
// CHECK-NEXT:   func.return %q1, %q2, %q3 : !qu.bit, !qu.bit, !qu.bit
// CHECK-NEXT: }
func.func @toffoli(%q1 : !qu.bit, %q2 : !qu.bit, %q3 : !qu.bit) -> (!qu.bit, !qu.bit, !qu.bit) {
  qref.apply<#gate.h> %q3
  qref.apply<#gate.cx> %q2, %q3
  qref.apply<#gate.t_dagger> %q3
  qref.apply<#gate.cx> %q1, %q3
  qref.apply<#gate.t> %q3
  qref.apply<#gate.cx> %q2, %q3
  qref.apply<#gate.t> %q2
  qref.apply<#gate.t> %q3
  qref.apply<#gate.cx> %q1, %q3
  qref.apply<#gate.cx> %q1, %q2
  qref.apply<#gate.t> %q1
  qref.apply<#gate.t_dagger> %q2
  qref.apply<#gate.t> %q3
  qref.apply<#gate.cx> %q1, %q2
  qref.apply<#gate.s> %q2
  qref.apply<#gate.h> %q3
  func.return %q1, %q2, %q3 : !qu.bit, !qu.bit, !qu.bit
}
