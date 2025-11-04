// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @phase_dyn(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   %id = gate.constant #gate.id<1>
// CHECK-NEXT:   %z = gate.constant #gate.z
// CHECK-NEXT:   %g = arith.select %p, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   func.return %q1 : !qu.bit
// CHECK-NEXT: }
func.func @phase_dyn(%q : !qu.bit) -> !qu.bit {
  %p = prob.bernoulli 0.1
  %id = gate.constant #gate.id<1>
  %z = gate.constant #gate.z
  %g = arith.select %p, %z, %id : !gate.type<1>
  %q1 = qssa.dyn_gate<%g> %q
  func.return %q1 : !qu.bit
}

// CHECK:      func.func @phase_scf(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   %q2 = scf.if %p -> (!qu.bit) {
// CHECK-NEXT:     %q1 = qssa.gate<#gate.z> %q
// CHECK-NEXT:     scf.yield %q1 : !qu.bit
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %q : !qu.bit
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %q2 : !qu.bit
// CHECK-NEXT: }
func.func @phase_scf(%q : !qu.bit) -> !qu.bit {
  %p = prob.bernoulli 0.1
  %q2 = scf.if %p -> (!qu.bit) {
    %q1 = qssa.gate<#gate.z> %q
    scf.yield %q1 : !qu.bit
  } else {
    scf.yield %q : !qu.bit
  }
  func.return %q2 : !qu.bit
}

// CHECK:       func.func @phase_cf(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   cf.cond_br %p, ^bb0, ^bb1(%q : !qu.bit)
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   %q1 = qssa.gate<#gate.z> %q
// CHECK-NEXT:   cf.br ^bb1(%q1 : !qu.bit)
// CHECK-NEXT: ^bb1(%q2 : !qu.bit):
// CHECK-NEXT:   func.return %q2 : !qu.bit
// CHECK-NEXT: }
func.func @phase_cf(%q : !qu.bit) -> !qu.bit {
  %p = prob.bernoulli 0.1
  cf.cond_br %p, ^bb0, ^bb1(%q: !qu.bit)
^bb0:
  %q1 = qssa.gate<#gate.z> %q
  cf.br ^bb1(%q1: !qu.bit)
^bb1(%q2 : !qu.bit):
  func.return %q2 : !qu.bit
}
