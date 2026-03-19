// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q0 = qu.alloc
// CHECK-GENERIC: %q0 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
%q0 = qu.alloc

// CHECK: %q1 = qu.alloc
// CHECK-GENERIC: %q1 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
%q1 = qu.alloc

// CHECK: %q2 = qssa.gate<#gate.h> %q0
// CHECK-GENERIC: %q2 = "qssa.gate"(%q0) <{gate = #gate.h}> : (!qu.bit) -> !qu.bit
%q2 = qssa.gate<#gate.h> %q0

// CHECK: %q3 = qssa.gate<#gate.rz<0.5pi>> %q1
// CHECK-GENERIC: %q3 = "qssa.gate"(%q1) <{gate = #gate.rz<0.5pi>}> : (!qu.bit) -> !qu.bit
%q3 = qssa.gate<#gate.rz<0.5pi>> %q1

// CHECK: %q4, %q5 = qssa.gate<#gate.cx> %q2, %q3
// CHECK-GENERIC: %q4, %q5 = "qssa.gate"(%q2, %q3) <{gate = #gate.cx}> : (!qu.bit, !qu.bit) -> (!qu.bit, !qu.bit)
%q4, %q5 = qssa.gate<#gate.cx> %q2, %q3

%g1 = "test.op"() : () -> !instrument.type<1>

// CHECK: %q6 = qssa.dyn_gate<%g1> %q4
// CHECK-GENERIC: %q6 = "qssa.dyn_gate"(%g1, %q4) : (!instrument.type<1>, !qu.bit) -> !qu.bit
%q6 = qssa.dyn_gate<%g1> %q4

// CHECK: %q7, %{{.*}} = qssa.measure %q6
// CHECK-GENERIC: %q7, %{{.*}} = "qssa.measure"(%q6) <{measurement = #measurement.comp_basis}> : (!qu.bit) -> (!qu.bit, i1)
%q7, %0 = qssa.measure %q6

// CHECK: %q8, %{{.*}} = qssa.measure<#measurement.xy<0.5pi>> %q7
// CHECK-GENERIC: %q8, %{{.*}} = "qssa.measure"(%q7) <{measurement = #measurement.xy<0.5pi>}> : (!qu.bit) -> (!qu.bit, i1)
%q8, %1 = qssa.measure<#measurement.xy<0.5pi>> %q7

%m = "test.op"() : () -> !instrument.type<1, i1>

// CHECK: %q9, %{{.*}} = qssa.dyn_measure<%m> %q8
// CHECK-GENERIC: %q9, %{{.*}} = "qssa.dyn_measure"(%m, %q8) : (!instrument.type<1, i1>, !qu.bit) -> (!qu.bit, i1)
%q9, %2 = qssa.dyn_measure<%m> %q8

// CHECK: %q10 = qssa.apply<#gate.h> %q9
// CHECK-GENERIC: %q10 = "qssa.apply"(%q9) <{instrument = #gate.h, resultSegmentSizes = array<i32: 1, 0>}> : (!qu.bit) -> !qu.bit
%q10 = qssa.apply<#gate.h> %q9

// CHECK: %q11, %3 = qssa.apply<#measurement.comp_basis> %q10 : i1
// CHECK-GENERIC: %q11, %3 = "qssa.apply"(%q10) <{instrument = #measurement.comp_basis, resultSegmentSizes = array<i32: 1, 1>}> : (!qu.bit) -> (!qu.bit, i1)
%q11, %3 = qssa.apply<#measurement.comp_basis> %q10 : i1

// CHECK: %q12, %q13, %4 = qssa.apply<#qec.stabilizer<XX>> %q5, %q11 : i1
// CHECK-GENERIC: %q12, %q13, %4 = "qssa.apply"(%q5, %q11) <{instrument = #qec.stabilizer<XX>, resultSegmentSizes = array<i32: 2, 1>}> : (!qu.bit, !qu.bit) -> (!qu.bit, !qu.bit, i1)
%q12, %q13, %4 = qssa.apply<#qec.stabilizer<XX>> %q5, %q11 : i1

%i = "test.op"() : () -> !instrument.type<0, i1, i32>

// CHECK: %5, %6 = qssa.dyn_apply<%i> : i1, i32
// CHECK-GENERIC: %5, %6 = "qssa.dyn_apply"(%i) <{resultSegmentSizes = array<i32: 0, 2>}> : (!instrument.type<0, i1, i32>) -> (i1, i32)
%5, %6 = qssa.dyn_apply<%i> : i1, i32

// CHECK: %{{.*}} = qssa.circuit() ({
// CHECK-NEXT: ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:   qssa.return %{{.*}}
// CHECK-NEXT: }) : () -> !instrument.type<1>
// CHECK-GENERIC: %{{.*}} = "qssa.circuit"() ({
// CHECK-GENERIC-NEXT: ^{{.+}}(%{{.*}} : !qu.bit):
// CHECK-GENERIC-NEXT:   "qssa.return"(%{{.*}}) : (!qu.bit) -> ()
// CHECK-GENERIC-NEXT: }) : () -> !instrument.type<1>
%circuit1 = qssa.circuit() ({
^bb0(%arg0 : !qu.bit):
  qssa.return %arg0
}) : () -> !instrument.type<1>

// CHECK: %{{.*}} = qssa.circuit() ({
// CHECK-NEXT: ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %{{.*}}
// CHECK-NEXT:   qssa.return %{{.*}}
// CHECK-NEXT: }) : () -> !instrument.type<1>
// CHECK-GENERIC: %{{.*}} = "qssa.circuit"() ({
// CHECK-GENERIC-NEXT: ^{{.+}}(%{{.*}} : !qu.bit):
// CHECK-GENERIC-NEXT:   %{{.*}} = "qssa.gate"(%{{.*}}) <{gate = #gate.x}> : (!qu.bit) -> !qu.bit
// CHECK-GENERIC-NEXT:   "qssa.return"(%{{.*}}) : (!qu.bit) -> ()
// CHECK-GENERIC-NEXT: }) : () -> !instrument.type<1>
%circuit2 = qssa.circuit() ({
^bb0(%arg0 : !qu.bit):
  %q14 = qssa.gate<#gate.x> %arg0
  qssa.return %q14
}) : () -> !instrument.type<1>

// CHECK: %{{.*}} = qssa.circuit() ({
// CHECK-NEXT: ^{{.*}}(%{{.*}} : !qu.bit, %{{.*}} : !qu.bit):
// CHECK-NEXT:   %{{.*}}, %{{.*}} = qssa.gate<#gate.cx> %{{.*}}, %{{.*}}
// CHECK-NEXT:   qssa.return %{{.*}}, %{{.*}}
// CHECK-NEXT: }) : () -> !instrument.type<2>
// CHECK-GENERIC: %{{.*}} = "qssa.circuit"() ({
// CHECK-GENERIC-NEXT: ^{{.+}}(%{{.*}} : !qu.bit, %{{.*}} : !qu.bit):
// CHECK-GENERIC-NEXT:   %{{.*}}, %{{.*}} = "qssa.gate"(%{{.*}}, %{{.*}}) <{gate = #gate.cx}> : (!qu.bit, !qu.bit) -> (!qu.bit, !qu.bit)
// CHECK-GENERIC-NEXT:   "qssa.return"(%{{.*}}, %{{.*}}) : (!qu.bit, !qu.bit) -> ()
// CHECK-GENERIC-NEXT: }) : () -> !instrument.type<2>
%circuit3 = qssa.circuit() ({
^bb0(%arg0 : !qu.bit, %arg1 : !qu.bit):
  %q15, %q16 = qssa.gate<#gate.cx> %arg0, %arg1
  qssa.return %q15, %q16
}) : () -> !instrument.type<2>
