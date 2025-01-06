// RUN: quopt -p canonicalize %s | filecheck %s

%g = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>

%q1 = qubit.alloc

"qref.dyn_gate"(%q1, %g) : (!qubit.bit, !gate.type<1>) -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %q1 = qubit.alloc
// CHECK-NEXT:    qref.gate<#gate.h> %q1
// CHECK-NEXT:  }
