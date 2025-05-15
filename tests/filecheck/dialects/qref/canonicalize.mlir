// RUN: quopt -p canonicalize %s | filecheck %s

%g = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>
%m = "measurement.constant"() <{"measurement" = #measurement.comp_basis}> : () -> !measurement.type<1>

%q1 = qu.alloc

"qref.dyn_gate"(%g, %q1) : (!gate.type<1>, !qu.bit) -> ()

%0 = "qref.dyn_measure"(%m, %q1) : (!measurement.type<1>, !qu.bit) -> i1

// CHECK:       builtin.module {
// CHECK-NEXT:    %q1 = qu.alloc
// CHECK-NEXT:    qref.gate<#gate.h> %q1
// CHECK-NEXT:    %{{.*}} = qref.measure %q1
// CHECK-NEXT:  }
