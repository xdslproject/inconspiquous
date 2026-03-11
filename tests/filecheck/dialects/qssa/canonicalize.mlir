// RUN: quopt -p canonicalize %s | filecheck %s

// CHECK:      %q1 = qu.alloc
// CHECK-NEXT: %q2 = qssa.gate<#gate.h> %q1
// CHECK-NEXT: %q3, %{{.*}} = qssa.measure %q2
// CHECK-NEXT: %q4 = qu.alloc
// CHECK-NOT:  #gate.id
// CHECK:      %g1, %g2 = "test.op"() : () -> (!instrument.type<2>, !instrument.type<2>)
// CHECK-NEXT: %q6 = qu.alloc
// CHECK-NEXT: %1, %2 = qssa.dyn_gate<%g1> %q4, %q6
// CHECK-NEXT: qssa.dyn_gate<%g2> %1, %2

%g = "instrument.constant"() <{instrument = #gate.h}> : () -> !instrument.type<1>
%m = "instrument.constant"() <{instrument = #measurement.comp_basis}> : () -> !instrument.type<1, i1>

%q1 = qu.alloc

%q2 = "qssa.dyn_gate"(%g, %q1) : (!instrument.type<1>, !qu.bit) -> !qu.bit

%q3, %0 = "qssa.dyn_measure"(%m, %q2) : (!instrument.type<1, i1>, !qu.bit) -> (!qu.bit, i1)

%q4 = qu.alloc
%q5 = "qssa.gate"(%q4) <{gate = #gate.id<1>}> : (!qu.bit) -> !qu.bit

%g1, %g2 = "test.op"() : () -> (!instrument.type<2>, !instrument.type<2>)
%g3 = "gate.compose"(%g1, %g2) : (!instrument.type<2>, !instrument.type<2>) -> !instrument.type<2>

%q6 = qu.alloc

"qssa.dyn_gate"(%g3, %q5, %q6) : (!instrument.type<2>, !qu.bit, !qu.bit) -> (!qu.bit, !qu.bit)
