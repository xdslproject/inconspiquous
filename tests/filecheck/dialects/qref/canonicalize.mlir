// RUN: quopt -p canonicalize %s | filecheck %s

// CHECK:      %q1 = qu.alloc
// CHECK-NEXT: qref.gate<#gate.h> %q1
// CHECK-NEXT: %{{.*}} = qref.measure %q1
// CHECK-NEXT: %q2 = qu.alloc
// CHECK-NEXT: %{{.*}} = qref.apply<#qec.stabilizer<XX>> %q1, %q2 : i1
// CHECK-NOT:  #gate.id
// CHECK:      %g2 = "test.op"() : () -> (!instrument.type<2>, !instrument.type<2>)
// CHECK-NEXT: %q3 = qu.alloc
// CHECK-NEXT: qref.dyn_gate<%g1> %q2, %q3
// CHECK-NEXT: qref.dyn_gate<%g2> %q2, %q3

%g = "instrument.constant"() <{instrument = #gate.h}> : () -> !instrument.type<1>
%m = "instrument.constant"() <{instrument = #measurement.comp_basis}> : () -> !instrument.type<1, i1>
%i = "instrument.constant"() <{instrument = #qec.stabilizer<XX>}> : () -> !instrument.type<2, i1>

%q1 = qu.alloc

"qref.dyn_gate"(%g, %q1) : (!instrument.type<1>, !qu.bit) -> ()

%0 = "qref.dyn_measure"(%m, %q1) : (!instrument.type<1, i1>, !qu.bit) -> i1

%q2 = qu.alloc

%1 = "qref.dyn_apply"(%i, %q1, %q2) : (!instrument.type<2, i1>, !qu.bit, !qu.bit) -> i1

"qref.gate"(%q2) <{gate = #gate.id<1>}> : (!qu.bit) -> ()

%g1, %g2 = "test.op"() : () -> (!instrument.type<2>, !instrument.type<2>)
%g3 = "gate.compose"(%g1, %g2) : (!instrument.type<2>, !instrument.type<2>) -> !instrument.type<2>


%q3 = qu.alloc

"qref.dyn_gate"(%g3, %q2, %q3) : (!instrument.type<2>, !qu.bit, !qu.bit) -> ()
