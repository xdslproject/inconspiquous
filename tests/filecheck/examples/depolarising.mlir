// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @depolarising_dyn(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   %id = gate.constant #gate.id
// CHECK-NEXT:   %p2 = prob.uniform : i2
// CHECK-NEXT:   %x = gate.constant #gate.x
// CHECK-NEXT:   %y = gate.constant #gate.y
// CHECK-NEXT:   %z = gate.constant #gate.z
// CHECK-NEXT:   %choice = varith.switch %p2 : i2 -> !gate.type<1>, [
// CHECK-NEXT:     default: %id,
// CHECK-NEXT:     1: %x,
// CHECK-NEXT:     -2: %y,
// CHECK-NEXT:     -1: %z
// CHECK-NEXT:   ]
// CHECK-NEXT:   %g = arith.select %p, %choice, %id : !gate.type<1>
// CHECK-NEXT:   %q1 = qssa.dyn_gate<%g> %q : !qubit.bit
// CHECK-NEXT:   func.return %q1 : !qubit.bit
// CHECK-NEXT: }
func.func @depolarising_dyn(%q : !qubit.bit) -> !qubit.bit {
  %p = prob.bernoulli 0.1
  %id = gate.constant #gate.id
  %p2 = prob.uniform : i2
  %x = gate.constant #gate.x
  %y = gate.constant #gate.y
  %z = gate.constant #gate.z
  %choice = varith.switch %p2 : i2 -> !gate.type<1>, [
    default: %id,
    1: %x,
    2: %y,
    3: %z
  ]
  %g = arith.select %p, %choice, %id : !gate.type<1>
  %q1 = qssa.dyn_gate<%g> %q : !qubit.bit
  func.return %q1 : !qubit.bit
}

// CHECK:      func.func @depolarising_scf(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   %q3 = scf.if %p -> (!qubit.bit) {
// CHECK-NEXT:     %p2 = prob.uniform : i4
// CHECK-NEXT:     %p3 = arith.index_cast %p2 : i4 to index
// CHECK-NEXT:     %q2 = scf.index_switch %p3 -> !qubit.bit
// CHECK-NEXT:     case 1 {
// CHECK-NEXT:       %q1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:       scf.yield %q1 : !qubit.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     case 2 {
// CHECK-NEXT:       %q1_1 = qssa.gate<#gate.y> %q
// CHECK-NEXT:       scf.yield %q1_1 : !qubit.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     case 3 {
// CHECK-NEXT:       %q1_2 = qssa.gate<#gate.z> %q
// CHECK-NEXT:       scf.yield %q1_2 : !qubit.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     default {
// CHECK-NEXT:       scf.yield %q : !qubit.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %q2 : !qubit.bit
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %q : !qubit.bit
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %q3 : !qubit.bit
// CHECK-NEXT: }
func.func @depolarising_scf(%q : !qubit.bit) -> !qubit.bit {
  %p = prob.bernoulli 0.1
  %q3 = scf.if %p -> (!qubit.bit) {
    %p2 = prob.uniform : i4
    %p3 = arith.index_cast %p2 : i4 to index
    %q2 = scf.index_switch %p3 -> !qubit.bit
    case 1 {
      %q1 = qssa.gate<#gate.x> %q
      scf.yield %q1 : !qubit.bit
    }
    case 2 {
      %q1 = qssa.gate<#gate.y> %q
      scf.yield %q1 : !qubit.bit
    }
    case 3 {
      %q1 = qssa.gate<#gate.z> %q
      scf.yield %q1 : !qubit.bit
    }
    default {
      scf.yield %q : !qubit.bit
    }
    scf.yield %q2 : !qubit.bit
  } else {
    scf.yield %q : !qubit.bit
  }
  func.return %q3 : !qubit.bit
}

// CHECK:      func.func @depolarising_cf(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   cf.cond_br %p, ^0, ^1(%q : !qubit.bit)
// CHECK-NEXT: ^0:
// CHECK-NEXT:   %p2 = prob.uniform : i4
// CHECK-NEXT:   cf.switch %p2 : i4, [
// CHECK-NEXT:     default: ^2(%q : !qubit.bit),
// CHECK-NEXT:     1: ^1,
// CHECK-NEXT:     2: ^3,
// CHECK-NEXT:     3: ^4
// CHECK-NEXT:   ]
// CHECK-NEXT: ^1:
// CHECK-NEXT:   %q1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:   cf.br ^2(%q1 : !qubit.bit)
// CHECK-NEXT: ^3:
// CHECK-NEXT:   %q2 = qssa.gate<#gate.y> %q
// CHECK-NEXT:   cf.br ^2(%q2 : !qubit.bit)
// CHECK-NEXT: ^4:
// CHECK-NEXT:   %q3 = qssa.gate<#gate.z> %q
// CHECK-NEXT:   cf.br ^2(%q3 : !qubit.bit)
// CHECK-NEXT: ^2(%q4 : !qubit.bit):
// CHECK-NEXT:   func.return %q4 : !qubit.bit
// CHECK-NEXT: }
func.func @depolarising_cf(%q : !qubit.bit) -> !qubit.bit {
  %p = prob.bernoulli 0.1
  cf.cond_br %p, ^0, ^1(%q: !qubit.bit)
^0:
  %p2 = prob.uniform : i4
  cf.switch %p2 : i4, [
    default: ^4(%q: !qubit.bit),
    1: ^1,
    2: ^2,
    3: ^3
  ]
^1:
  %q1 = qssa.gate<#gate.x> %q
  cf.br ^4(%q1: !qubit.bit)
^2:
  %q2 = qssa.gate<#gate.y> %q
  cf.br ^4(%q2: !qubit.bit)
^3:
  %q3 = qssa.gate<#gate.z> %q
  cf.br ^4(%q3: !qubit.bit)
^4(%q4 : !qubit.bit):
  func.return %q4 : !qubit.bit
}
