// RUN: quopt %s -p randomized-comp | filecheck %s

// CHECK:      func.func @t_gate(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %0 = prob.uniform : i1
// CHECK-NEXT:   %1 = prob.uniform : i1
// CHECK-NEXT:   %2 = gate.constant #gate.id
// CHECK-NEXT:   %3 = gate.constant #gate.x
// CHECK-NEXT:   %4 = gate.constant #gate.z
// CHECK-NEXT:   %5 = gate.constant #gate.s
// CHECK-NEXT:   %6 = arith.select %0, %3, %2 : !gate.type<1>
// CHECK-NEXT:   %7 = qssa.dyn_gate<%6> %q : !qubit.bit
// CHECK-NEXT:   %8 = arith.select %1, %4, %2 : !gate.type<1>
// CHECK-NEXT:   %9 = qssa.dyn_gate<%8> %7 : !qubit.bit
// CHECK-NEXT:   %10 = qssa.gate<#gate.t> %9
// CHECK-NEXT:   %11 = arith.select %1, %4, %2 : !gate.type<1>
// CHECK-NEXT:   %12 = qssa.dyn_gate<%11> %10 : !qubit.bit
// CHECK-NEXT:   %13 = qssa.dyn_gate<%6> %12 : !qubit.bit
// CHECK-NEXT:   %14 = arith.select %0, %5, %2 : !gate.type<1>
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%14> %13 : !qubit.bit
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
// CHECK-NEXT: }
func.func @t_gate(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.t> %q
  func.return %q_1 : !qubit.bit
}

// CHECK:     func.func @t_dagger_gate(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %0 = prob.uniform : i1
// CHECK-NEXT:   %1 = prob.uniform : i1
// CHECK-NEXT:   %2 = gate.constant #gate.id
// CHECK-NEXT:   %3 = gate.constant #gate.x
// CHECK-NEXT:   %4 = gate.constant #gate.z
// CHECK-NEXT:   %5 = gate.constant #gate.s_dagger
// CHECK-NEXT:   %6 = arith.select %0, %3, %2 : !gate.type<1>
// CHECK-NEXT:   %7 = qssa.dyn_gate<%6> %q : !qubit.bit
// CHECK-NEXT:   %8 = arith.select %1, %4, %2 : !gate.type<1>
// CHECK-NEXT:   %9 = qssa.dyn_gate<%8> %7 : !qubit.bit
// CHECK-NEXT:   %10 = qssa.gate<#gate.t_dagger> %9
// CHECK-NEXT:   %11 = arith.select %1, %4, %2 : !gate.type<1>
// CHECK-NEXT:   %12 = qssa.dyn_gate<%11> %10 : !qubit.bit
// CHECK-NEXT:   %13 = qssa.dyn_gate<%6> %12 : !qubit.bit
// CHECK-NEXT:   %14 = arith.select %0, %5, %2 : !gate.type<1>
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%14> %13 : !qubit.bit
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
// CHECK-NEXT: }
func.func @t_dagger_gate(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.t_dagger> %q
  func.return %q_1 : !qubit.bit
}

// CHECK:      func.func @h_gate(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %0 = prob.uniform : i1
// CHECK-NEXT:   %1 = prob.uniform : i1
// CHECK-NEXT:   %2 = gate.constant #gate.id
// CHECK-NEXT:   %3 = gate.constant #gate.x
// CHECK-NEXT:   %4 = gate.constant #gate.z
// CHECK-NEXT:   %5 = arith.select %0, %3, %2 : !gate.type<1>
// CHECK-NEXT:   %6 = qssa.dyn_gate<%5> %q : !qubit.bit
// CHECK-NEXT:   %7 = arith.select %1, %4, %2 : !gate.type<1>
// CHECK-NEXT:   %8 = qssa.dyn_gate<%7> %6 : !qubit.bit
// CHECK-NEXT:   %9 = qssa.gate<#gate.h> %8
// CHECK-NEXT:   %10 = arith.select %1, %3, %2 : !gate.type<1>
// CHECK-NEXT:   %11 = qssa.dyn_gate<%10> %9 : !qubit.bit
// CHECK-NEXT:   %12 = arith.select %0, %4, %2 : !gate.type<1>
// CHECK-NEXT:   %q_1 = qssa.dyn_gate<%12> %11 : !qubit.bit
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
// CHECK-NEXT: }
func.func @h_gate(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.h> %q
  func.return %q_1 : !qubit.bit
}

// CHECK:      func.func @cnot_gate(%q1 : !qubit.bit, %q2 : !qubit.bit) -> (!qubit.bit, !qubit.bit) {
// CHECK-NEXT:   %0 = prob.uniform : i1
// CHECK-NEXT:   %1 = prob.uniform : i1
// CHECK-NEXT:   %2 = prob.uniform : i1
// CHECK-NEXT:   %3 = prob.uniform : i1
// CHECK-NEXT:   %4 = gate.constant #gate.id
// CHECK-NEXT:   %5 = gate.constant #gate.x
// CHECK-NEXT:   %6 = gate.constant #gate.z
// CHECK-NEXT:   %7 = arith.select %0, %5, %4 : !gate.type<1>
// CHECK-NEXT:   %8 = arith.select %1, %5, %4 : !gate.type<1>
// CHECK-NEXT:   %9 = arith.select %2, %6, %4 : !gate.type<1>
// CHECK-NEXT:   %10 = arith.select %3, %6, %4 : !gate.type<1>
// CHECK-NEXT:   %11 = qssa.dyn_gate<%7> %q1 : !qubit.bit
// CHECK-NEXT:   %12 = qssa.dyn_gate<%9> %11 : !qubit.bit
// CHECK-NEXT:   %13 = qssa.dyn_gate<%8> %q2 : !qubit.bit
// CHECK-NEXT:   %14 = qssa.dyn_gate<%10> %13 : !qubit.bit
// CHECK-NEXT:   %15, %16 = qssa.gate<#gate.cnot> %12, %14
// CHECK-NEXT:   %17 = qssa.dyn_gate<%9> %15 : !qubit.bit
// CHECK-NEXT:   %18 = qssa.dyn_gate<%10> %17 : !qubit.bit
// CHECK-NEXT:   %19 = qssa.dyn_gate<%10> %16 : !qubit.bit
// CHECK-NEXT:   %20 = qssa.dyn_gate<%7> %19 : !qubit.bit
// CHECK-NEXT:   %q1_1 = qssa.dyn_gate<%7> %18 : !qubit.bit
// CHECK-NEXT:   %q2_1 = qssa.dyn_gate<%8> %20 : !qubit.bit
// CHECK-NEXT:   func.return %q1_1, %q2_1 : !qubit.bit, !qubit.bit
// CHECK-NEXT: }
func.func @cnot_gate(%q1: !qubit.bit, %q2: !qubit.bit) -> (!qubit.bit, !qubit.bit) {
  %q1_1, %q2_1 = qssa.gate<#gate.cnot> %q1, %q2
  func.return %q1_1, %q2_1 : !qubit.bit, !qubit.bit
}

// CHECK:      func.func @measure(%q : !qubit.bit) -> i1 {
// CHECK-NEXT:   %0 = prob.uniform : i1
// CHECK-NEXT:   %1 = prob.uniform : i1
// CHECK-NEXT:   %2 = gate.constant #gate.id
// CHECK-NEXT:   %3 = gate.constant #gate.x
// CHECK-NEXT:   %4 = gate.constant #gate.z
// CHECK-NEXT:   %5 = arith.select %0, %3, %2 : !gate.type<1>
// CHECK-NEXT:   %6 = qssa.dyn_gate<%5> %q : !qubit.bit
// CHECK-NEXT:   %7 = arith.select %1, %4, %2 : !gate.type<1>
// CHECK-NEXT:   %8 = qssa.dyn_gate<%7> %6 : !qubit.bit
// CHECK-NEXT:   %9 = qssa.measure %8
// CHECK-NEXT:   %10 = arith.addi %0, %9 : i1
// CHECK-NEXT:   func.return %10 : i1
// CHECK-NEXT: }
func.func @measure(%q: !qubit.bit) -> i1 {
  %0 = qssa.measure %q
  func.return %0 : i1
}
