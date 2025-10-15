// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @depolarising_dyn(%q : !qu.bit) -> !qu.bit {
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
// CHECK-NEXT:   %q1 = qssa.dyn_gate<%g> %q
// CHECK-NEXT:   func.return %q1 : !qu.bit
// CHECK-NEXT: }
func.func @depolarising_dyn(%q : !qu.bit) -> !qu.bit {
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
  %q1 = qssa.dyn_gate<%g> %q
  func.return %q1 : !qu.bit
}

// CHECK:      func.func @depolarising_scf(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   %q3 = scf.if %p -> (!qu.bit) {
// CHECK-NEXT:     %p2 = prob.uniform : i4
// CHECK-NEXT:     %p3 = arith.index_cast %p2 : i4 to index
// CHECK-NEXT:     %q2 = scf.index_switch %p3 -> !qu.bit
// CHECK-NEXT:     case 1 {
// CHECK-NEXT:       %q1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:       scf.yield %q1 : !qu.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     case 2 {
// CHECK-NEXT:       %q1_1 = qssa.gate<#gate.y> %q
// CHECK-NEXT:       scf.yield %q1_1 : !qu.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     case 3 {
// CHECK-NEXT:       %q1_2 = qssa.gate<#gate.z> %q
// CHECK-NEXT:       scf.yield %q1_2 : !qu.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     default {
// CHECK-NEXT:       scf.yield %q : !qu.bit
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %q2 : !qu.bit
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %q : !qu.bit
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %q3 : !qu.bit
// CHECK-NEXT: }
func.func @depolarising_scf(%q : !qu.bit) -> !qu.bit {
  %p = prob.bernoulli 0.1
  %q3 = scf.if %p -> (!qu.bit) {
    %p2 = prob.uniform : i4
    %p3 = arith.index_cast %p2 : i4 to index
    %q2 = scf.index_switch %p3 -> !qu.bit
    case 1 {
      %q1 = qssa.gate<#gate.x> %q
      scf.yield %q1 : !qu.bit
    }
    case 2 {
      %q1 = qssa.gate<#gate.y> %q
      scf.yield %q1 : !qu.bit
    }
    case 3 {
      %q1 = qssa.gate<#gate.z> %q
      scf.yield %q1 : !qu.bit
    }
    default {
      scf.yield %q : !qu.bit
    }
    scf.yield %q2 : !qu.bit
  } else {
    scf.yield %q : !qu.bit
  }
  func.return %q3 : !qu.bit
}

// CHECK:      func.func @depolarising_cf(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %p = prob.bernoulli 1.000000e-01
// CHECK-NEXT:   cf.cond_br %p, ^bb0, ^bb1(%q : !qu.bit)
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   %p2 = prob.uniform : i4
// CHECK-NEXT:   cf.switch %p2 : i4, [
// CHECK-NEXT:     default: ^bb2(%q : !qu.bit),
// CHECK-NEXT:     1: ^bb1,
// CHECK-NEXT:     2: ^bb3,
// CHECK-NEXT:     3: ^bb4
// CHECK-NEXT:   ]
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %q1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:   cf.br ^bb2(%q1 : !qu.bit)
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   %q2 = qssa.gate<#gate.y> %q
// CHECK-NEXT:   cf.br ^bb2(%q2 : !qu.bit)
// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   %q3 = qssa.gate<#gate.z> %q
// CHECK-NEXT:   cf.br ^bb2(%q3 : !qu.bit)
// CHECK-NEXT: ^bb2(%q4 : !qu.bit):
// CHECK-NEXT:   func.return %q4 : !qu.bit
// CHECK-NEXT: }
func.func @depolarising_cf(%q : !qu.bit) -> !qu.bit {
  %p = prob.bernoulli 0.1
  cf.cond_br %p, ^bb0, ^bb1(%q: !qu.bit)
^bb0:
  %p2 = prob.uniform : i4
  cf.switch %p2 : i4, [
    default: ^bb4(%q: !qu.bit),
    1: ^bb1,
    2: ^bb2,
    3: ^bb3
  ]
^bb1:
  %q1 = qssa.gate<#gate.x> %q
  cf.br ^bb4(%q1: !qu.bit)
^bb2:
  %q2 = qssa.gate<#gate.y> %q
  cf.br ^bb4(%q2: !qu.bit)
^bb3:
  %q3 = qssa.gate<#gate.z> %q
  cf.br ^bb4(%q3: !qu.bit)
^bb4(%q4 : !qu.bit):
  func.return %q4 : !qu.bit
}
