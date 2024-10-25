// RUN: quopt -p canonicalize %s | filecheck %s

%g = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>

%q1 = qubit.alloc

%q2 = "qssa.dyn_gate"(%q1, %g) : (!qubit.bit, !gate.type<1>) -> !qubit.bit

// CHECK:       builtin.module {
// CHECK-NEXT:    %q1 = qubit.alloc
// CHECK-NEXT:    %q2 = qssa.gate<#gate.h> %q1 : !qubit.bit
// CHECK-NEXT:  }
