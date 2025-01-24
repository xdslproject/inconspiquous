// RUN: quopt -p canonicalize %s | filecheck %s

%g = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>

%q1 = qubit.alloc

"qref.dyn_gate"(%g, %q1) : (!gate.type<1>, !qubit.bit) -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %q1 = qubit.alloc
// CHECK-NEXT:    qref.gate<#gate.h> %q1
// CHECK-NEXT:  }
