// RUN: quopt -p canonicalize %s | filecheck %s

%g = "gate.constant"() <{"gate" = #gate.h}> : () -> !gate.type<1>
%m = "measurement.constant"() <{"measurement" = #measurement.comp_basis}> : () -> !measurement.type<1>

%q1 = qu.alloc

%q2 = "qssa.dyn_gate"(%g, %q1) : (!gate.type<1>, !qu.bit) -> !qu.bit

%0 = "qssa.dyn_measure"(%m, %q2) : (!measurement.type<1>, !qu.bit) -> i1

%q3 = qu.alloc
%q4 = "qssa.gate"(%q3) <{gate = #gate.id<1>}> : (!qu.bit) -> !qu.bit

%g1, %g2 = "test.op"() : () -> (!gate.type<2>, !gate.type<2>)
%g3 = "gate.compose"(%g1, %g2) : (!gate.type<2>, !gate.type<2>) -> !gate.type<2>

%q5 = qu.alloc

"qssa.dyn_gate"(%g3, %q4, %q5) : (!gate.type<2>, !qu.bit, !qu.bit) -> (!qu.bit, !qu.bit)

// CHECK:       builtin.module {
// CHECK-NEXT:    %q1 = qu.alloc
// CHECK-NEXT:    %q2 = qssa.gate<#gate.h> %q1
// CHECK-NEXT:    %{{.*}} = qssa.measure %q2
// CHECK-NEXT:    %q3 = qu.alloc
// CHECK-NOT:     #gate.id
// CHECK:    %g1, %g2 = "test.op"() : () -> (!gate.type<2>, !gate.type<2>)
// CHECK-NEXT:    %q5 = qu.alloc
// CHECK-NEXT:    %1, %2 = qssa.dyn_gate<%g1> %q3, %q5
// CHECK-NEXT:    qssa.dyn_gate<%g2> %1, %2
// CHECK-NEXT:  }
