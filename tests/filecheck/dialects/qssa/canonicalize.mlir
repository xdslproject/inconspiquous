// RUN: quopt -p canonicalize %s | filecheck %s

%g = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>
%m = "measurement.constant"() <{"measurement" = #measurement.comp_basis}> : () -> !measurement.type<1>

%q1 = qu.alloc

%q2 = "qssa.dyn_gate"(%g, %q1) : (!gate.type<1>, !qu.bit) -> !qu.bit

%0 = "qssa.dyn_measure"(%m, %q2) : (!measurement.type<1>, !qu.bit) -> i1

// CHECK:       builtin.module {
// CHECK-NEXT:    %q1 = qu.alloc
// CHECK-NEXT:    %q2 = qssa.gate<#gate.h> %q1
// CHECK-NEXT:    %{{.*}} = qssa.measure %q2
// CHECK-NEXT:  }
