func.func @double_phase(%q : !qubit.bit) -> !qubit.bit {
  %p = prob.bernoulli 0.1
  %id = gate.constant #gate.id
  %z = gate.constant #gate.z
  %g = arith.select %p, %z, %id : !gate.type<1>
  %q1 = qssa.dyn_gate<%g> %q : !qubit.bit

  %p2 = prob.bernoulli 0.1
  %id2 = gate.constant #gate.id
  %z2 = gate.constant #gate.z
  %g2 = arith.select %p2, %z2, %id2 : !gate.type<1>
  %q2 = qssa.dyn_gate<%g2> %q1 : !qubit.bit
  func.return %q2 : !qubit.bit
}
