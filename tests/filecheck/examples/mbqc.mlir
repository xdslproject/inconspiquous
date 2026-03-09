// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @mbqc() -> (!qu.bit, !qu.bit) {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %q1 = qu.alloc
// CHECK-NEXT:   qssa.apply<#gate.cx> %q0, %q1
// CHECK-NEXT:   func.return %q0, %q1 : !qu.bit, !qu.bit
// CHECK-NEXT: }

func.func @mbqc() -> (!qu.bit, !qu.bit) {
  %q0 = qu.alloc
  %q1 = qu.alloc
  qssa.apply<#gate.cx> %q0, %q1
  func.return %q0, %q1 : !qu.bit, !qu.bit
}
