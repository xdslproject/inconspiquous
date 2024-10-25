// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @phase_dyn(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01 : f64
// CHECK-NEXT:   %id = gate.constant #gate.id
// CHECK-NEXT:   %z = gate.constant #gate.z
// CHECK-NEXT:   %g = arith.select %p, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q1 = qssa.dyn_gate<%g> %q : !qubit.bit
// CHECK-NEXT:   func.return %q1 : !qubit.bit
// CHECK-NEXT: }
func.func @phase_dyn(%q : !qubit.bit) -> !qubit.bit {
  %p = prob.bernoulli 0.1
  %id = gate.constant #gate.id
  %z = gate.constant #gate.z
  %g = arith.select %p, %z, %id : !gate.type<1>
  %q1 = qssa.dyn_gate<%g> %q : !qubit.bit
  func.return %q1 : !qubit.bit
}

// CHECK:      func.func @phase_scf(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01 : f64
// CHECK-NEXT:   %q2 = scf.if %p -> (!qubit.bit) {
// CHECK-NEXT:     %q1 = qssa.gate<#gate.z> %q : !qubit.bit
// CHECK-NEXT:     scf.yield %q1 : !qubit.bit
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %q : !qubit.bit
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %q2 : !qubit.bit
// CHECK-NEXT: }
func.func @phase_scf(%q : !qubit.bit) -> !qubit.bit {
  %p = prob.bernoulli 0.1
  %q2 = scf.if %p -> (!qubit.bit) {
    %q1 = qssa.gate<#gate.z> %q : !qubit.bit
    scf.yield %q1 : !qubit.bit
  } else {
    scf.yield %q : !qubit.bit
  }
  func.return %q2 : !qubit.bit
}

// CHECK:       func.func @phase_cf(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01 : f64
// CHECK-NEXT:   cf.cond_br %p, ^0, ^1(%q : !qubit.bit)
// CHECK-NEXT: ^0:
// CHECK-NEXT:   %q1 = qssa.gate<#gate.z> %q : !qubit.bit
// CHECK-NEXT:   cf.br ^1(%q1 : !qubit.bit)
// CHECK-NEXT: ^1(%q2 : !qubit.bit):
// CHECK-NEXT:   func.return %q2 : !qubit.bit
// CHECK-NEXT: }
func.func @phase_cf(%q : !qubit.bit) -> !qubit.bit {
  %p = prob.bernoulli 0.1
  cf.cond_br %p, ^0, ^1(%q: !qubit.bit)
^0:
  %q1 = qssa.gate<#gate.z> %q : !qubit.bit
  cf.br ^1(%q1: !qubit.bit)
^1(%q2 : !qubit.bit):
  func.return %q2 : !qubit.bit
}
