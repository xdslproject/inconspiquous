// RUN: quopt %s -p lower-dyn-gate-to-scf

%g1, %g2, %g3, %g4 = "test.op"() : () -> (!gate.type<1>,!gate.type<1>,!gate.type<1>,!gate.type<1>)

%b = "test.op"() : () -> i1
%g5 = arith.select %b, %g1, %g2 : !gate.type<1>

%i = "test.op"() : () -> i2
%g6 = varith.switch %i : i2 -> !gate.type<1>, [
  default: %g1,
  1: %g2,
  2: %g3,
  3: %g4
]

// CHECK-NEXT:    %q = qu.alloc
%q = qu.alloc

// CHECK-NEXT:    scf.if %b {
// CHECK-NEXT:      qref.dyn_gate<%g1> %q
// CHECK-NEXT:    } else {
// CHECK-NEXT:      qref.dyn_gate<%g2> %q
// CHECK-NEXT:    }
qref.dyn_gate<%g5> %q

// CHECK-NEXT:    %0 = arith.index_cast %i : i2 to index
// CHECK-NEXT:    scf.index_switch %0
// CHECK-NEXT:    case 1 {
// CHECK-NEXT:      qref.dyn_gate<%g2> %q
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    case -2 {
// CHECK-NEXT:      qref.dyn_gate<%g3> %q
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    case -1 {
// CHECK-NEXT:      qref.dyn_gate<%g4> %q
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:      qref.dyn_gate<%g1> %q
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
qref.dyn_gate<%g6> %q

// CHECK-NEXT:    %q1 = scf.if %b -> (!qu.bit) {
// CHECK-NEXT:      %1 = qssa.dyn_gate<%g1> %q
// CHECK-NEXT:      scf.yield %1 : !qu.bit
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %2 = qssa.dyn_gate<%g2> %q
// CHECK-NEXT:      scf.yield %2 : !qu.bit
// CHECK-NEXT:    }
%q1 = qssa.dyn_gate<%g5> %q

// CHECK-NEXT:    %q2 = arith.index_cast %i : i2 to index
// CHECK-NEXT:    %q2_1 = scf.index_switch %q2 -> !qu.bit
// CHECK-NEXT:    case 1 {
// CHECK-NEXT:      %3 = qssa.dyn_gate<%g2> %q1
// CHECK-NEXT:      scf.yield %3 : !qu.bit
// CHECK-NEXT:    }
// CHECK-NEXT:    case -2 {
// CHECK-NEXT:      %4 = qssa.dyn_gate<%g3> %q1
// CHECK-NEXT:      scf.yield %4 : !qu.bit
// CHECK-NEXT:    }
// CHECK-NEXT:    case -1 {
// CHECK-NEXT:      %5 = qssa.dyn_gate<%g4> %q1
// CHECK-NEXT:      scf.yield %5 : !qu.bit
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:      %6 = qssa.dyn_gate<%g1> %q1
// CHECK-NEXT:      scf.yield %6 : !qu.bit
// CHECK-NEXT:    }
// CHECK-NEXT:  }
%q2 = qssa.dyn_gate<%g6> %q1
