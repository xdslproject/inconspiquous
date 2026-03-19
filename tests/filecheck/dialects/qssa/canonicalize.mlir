// RUN: quopt -p canonicalize %s | filecheck %s

// CHECK:      %q1 = qu.alloc
// CHECK-NEXT: %q2 = qssa.gate<#gate.h> %q1
// CHECK-NEXT: %{{.*}}, %q3 = qssa.measure %q2
// CHECK-NEXT: %q4 = qu.alloc
// CHECK-NEXT: %1, %q5, %q6 = qssa.apply<#qec.stabilizer<XX>> %q3, %q4 : i1
// CHECK-NOT:  #gate.id
// CHECK:      %g1, %g2 = "test.op"() : () -> (!instrument.type<2>, !instrument.type<2>)
// CHECK-NEXT: %q8 = qu.alloc
// CHECK-NEXT: %2, %3 = qssa.dyn_gate<%g1> %q5, %q8
// CHECK-NEXT: qssa.dyn_gate<%g2> %2, %3

%g = "instrument.constant"() <{instrument = #gate.h}> : () -> !instrument.type<1>
%m = "instrument.constant"() <{instrument = #measurement.comp_basis}> : () -> !instrument.type<1, i1>
%i = "instrument.constant"() <{instrument = #qec.stabilizer<XX>}> : () -> !instrument.type<2, i1>

%q1 = qu.alloc

%q2 = "qssa.dyn_gate"(%g, %q1) : (!instrument.type<1>, !qu.bit) -> !qu.bit

%0, %q3 = "qssa.dyn_measure"(%m, %q2) : (!instrument.type<1, i1>, !qu.bit) -> (i1, !qu.bit)

%q4 = qu.alloc

%1, %q5, %q6 = "qssa.dyn_apply"(%i, %q3, %q4) <{resultSegmentSizes = array<i32: 1, 2>}> : (!instrument.type<2, i1>, !qu.bit, !qu.bit) -> (i1, !qu.bit, !qu.bit)

%q7 = "qssa.gate"(%q5) <{gate = #gate.id<1>}> : (!qu.bit) -> !qu.bit

%g1, %g2 = "test.op"() : () -> (!instrument.type<2>, !instrument.type<2>)
%g3 = "gate.compose"(%g1, %g2) : (!instrument.type<2>, !instrument.type<2>) -> !instrument.type<2>

%q8 = qu.alloc

"qssa.dyn_gate"(%g3, %q7, %q8) : (!instrument.type<2>, !qu.bit, !qu.bit) -> (!qu.bit, !qu.bit)
