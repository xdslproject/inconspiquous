// RUN: QUOPT_ROUNDTRIP
// RUN: quopt %s -p xzs-simpl | filecheck %s --check-prefix CHECK-SIMPL

// CHECK:      func.func @double_phase(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   %id = gate.constant #gate.id<1>
// CHECK-NEXT:   %z = gate.constant #gate.z
// CHECK-NEXT:   %g = arith.select %p, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   %p2 = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   %id2 = gate.constant #gate.id<1>
// CHECK-NEXT:   %z2 = gate.constant #gate.z
// CHECK-NEXT:   %g2 = arith.select %p2, %z2, %id2 : !gate.type<1>
// CHECK-NEXT:   %q2 = qssa.dyn_gate<%g2> %q1
// CHECK-NEXT:   func.return %q2 : !qu.bit
// CHECK-NEXT: }

// CHECK-SIMPL:      func.func @double_phase(%q : !qu.bit) -> !qu.bit {
// CHECK-SIMPL-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-SIMPL-NEXT:   %p2 = prob.bernoulli 1.000000e-01
// CHECK-SIMPL-NEXT:   %0 = arith.xori %p, %p2 : i1
// CHECK-SIMPL-NEXT:   %g = gate.constant #gate.id<1>
// CHECK-SIMPL-NEXT:   %g_1 = gate.constant #gate.z
// CHECK-SIMPL-NEXT:   %g_2 = arith.select %0, %g_1, %g : !gate.type<1>
// CHECK-SIMPL-NEXT:   %q2 = qssa.dyn_gate<%g_2> %q
// CHECK-SIMPL-NEXT:   func.return %q2 : !qu.bit
// CHECK-SIMPL-NEXT: }

func.func @double_phase(%q : !qu.bit) -> !qu.bit {
  %p = prob.bernoulli 0.1
  %id = gate.constant #gate.id<1>
  %z = gate.constant #gate.z
  %g = arith.select %p, %z, %id : !gate.type<1>
  %q1 = qssa.dyn_gate<%g> %q

  %p2 = prob.bernoulli 0.1
  %id2 = gate.constant #gate.id<1>
  %z2 = gate.constant #gate.z
  %g2 = arith.select %p2, %z2, %id2 : !gate.type<1>
  %q2 = qssa.dyn_gate<%g2> %q1
  func.return %q2 : !qu.bit
}
