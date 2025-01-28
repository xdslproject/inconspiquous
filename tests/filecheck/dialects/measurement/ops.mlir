// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: "test.op"() {measurement = #measurement.comp_basis} : () -> ()
"test.op"() {measurement = #measurement.comp_basis} : () -> ()

// CHECK: "test.op"() {measurement = #measurement.xy<pi>} : () -> ()
"test.op"() {measurement = #measurement.xy<pi>} : () -> ()

// CHECK: %m = measurement.constant #measurement.comp_basis
// CHECK-GENERIC: %m = "measurement.constant"() <{measurement = #measurement.comp_basis}> : () -> !measurement.type<1>
%m = measurement.constant #measurement.comp_basis

%a = "test.op"() : () -> !gate.angle_type

// CHECK: %m2 = measurement.dyn_xy<%a>
// CHECK-GENERIC: %m2 = "measurement.dyn_xy"(%a) : (!gate.angle_type) -> !measurement.type<1>
%m2 = measurement.dyn_xy<%a>
