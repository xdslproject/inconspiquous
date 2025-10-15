// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q0 = qu.alloc
// CHECK-GENERIC: %q0 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
%q0 = qu.alloc

// CHECK: %q1 = qu.alloc
// CHECK-GENERIC: %q1 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
%q1 = qu.alloc

// CHECK: qref.gate<#gate.h> %q0
// CHECK-GENERIC: "qref.gate"(%q0) <{gate = #gate.h}> : (!qu.bit) -> ()
qref.gate<#gate.h> %q0

// CHECK: qref.gate<#gate.rz<0.5pi>> %q1
// CHECK-GENERIC: "qref.gate"(%q1) <{gate = #gate.rz<0.5pi>}> : (!qu.bit) -> ()
qref.gate<#gate.rz<0.5pi>> %q1

// CHECK: qref.gate<#gate.cx> %q0, %q1
// CHECK-GENERIC: "qref.gate"(%q0, %q1) <{gate = #gate.cx}> : (!qu.bit, !qu.bit)
qref.gate<#gate.cx> %q0, %q1

%g1 = "test.op"() : () -> !gate.type<1>

// CHECK: qref.dyn_gate<%g1> %q1
// CHECK-GENERIC: "qref.dyn_gate"(%g1, %q1) : (!gate.type<1>, !qu.bit) -> ()
qref.dyn_gate<%g1> %q1

// CHECK: %{{.*}} = qref.measure %q0
// CHECK-GENERIC: %{{.*}} = "qref.measure"(%q0) <{measurement = #measurement.comp_basis}> : (!qu.bit) -> i1
%0 = qref.measure %q0

// CHECK: %{{.*}} = qref.measure<#measurement.xy<0.5pi>> %q1
// CHECK-GENERIC: %{{.*}} = "qref.measure"(%q1) <{measurement = #measurement.xy<0.5pi>}> : (!qu.bit) -> i1
%1 = qref.measure<#measurement.xy<0.5pi>> %q1

%q2 = qu.alloc

%m = "test.op"() : () -> !measurement.type<1>

// CHECK: %{{.*}} = qref.dyn_measure<%m> %q2
// CHECK-GENERIC: %{{.*}} = "qref.dyn_measure"(%m, %q2) : (!measurement.type<1>, !qu.bit) -> i1
%2 = qref.dyn_measure<%m> %q2

// CHECK: %{{.*}} = qref.circuit() ({
// CHECK-NEXT: ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:   qref.return
// CHECK-NEXT: }) : () -> !gate.type<1>
// CHECK-GENERIC: %{{.*}} = "qref.circuit"() ({
// CHECK-GENERIC-NEXT: ^{{.+}}(%{{.*}} : !qu.bit):
// CHECK-GENERIC-NEXT:   "qref.return"() : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> !gate.type<1>
%circuit1 = qref.circuit() ({
^bb0(%arg0 : !qu.bit):
  qref.return
}) : () -> !gate.type<1>

// CHECK: %{{.*}} = qref.circuit() ({
// CHECK-NEXT: ^{{.*}}(%{{.*}} : !qu.bit, %{{.*}} : !qu.bit):
// CHECK-NEXT:   qref.gate<#gate.cx> %{{.*}}, %{{.*}}
// CHECK-NEXT:   qref.return
// CHECK-NEXT: }) : () -> !gate.type<2>
// CHECK-GENERIC: %{{.*}} = "qref.circuit"() ({
// CHECK-GENERIC-NEXT: ^{{.+}}(%{{.*}} : !qu.bit, %{{.*}} : !qu.bit):
// CHECK-GENERIC-NEXT:   "qref.gate"(%{{.*}}, %{{.*}}) <{gate = #gate.cx}> : (!qu.bit, !qu.bit) -> ()
// CHECK-GENERIC-NEXT:   "qref.return"() : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> !gate.type<2>
%circuit2 = qref.circuit() ({
^bb0(%arg0 : !qu.bit, %arg1 : !qu.bit):
  qref.gate<#gate.cx> %arg0, %arg1
  qref.return
}) : () -> !gate.type<2>
