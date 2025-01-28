// RUN: quopt -p canonicalize %s | filecheck %s

%g = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>
%m = "measurement.constant"() <{"measurement" = #measurement.comp_basis}> : () -> !measurement.type<1>

%q1 = qubit.alloc

%q2 = "qssa.dyn_gate"(%g, %q1) : (!gate.type<1>, !qubit.bit) -> !qubit.bit

%0 = "qssa.dyn_measure"(%m, %q2) : (!measurement.type<1>, !qubit.bit) -> i1

// CHECK:       builtin.module {
// CHECK-NEXT:    %q1 = qubit.alloc
// CHECK-NEXT:    %q2 = qssa.gate<#gate.h> %q1
// CHECK-NEXT:    %{{.*}} = qssa.measure %q2
// CHECK-NEXT:  }
