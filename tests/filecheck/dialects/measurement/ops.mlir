// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: "test.op"() {measurement = #measurement.comp_basis} : () -> ()
"test.op"() {measurement = #measurement.comp_basis} : () -> ()

// CHECK: "test.op"() {measurement = #measurement.x_basis} : () -> ()
"test.op"() {measurement = #measurement.x_basis} : () -> ()

// CHECK: "test.op"() {measurement = #measurement.xy<pi>} : () -> ()
"test.op"() {measurement = #measurement.xy<pi>} : () -> ()

%a = "test.op"() : () -> !angle.type

// CHECK: %m2 = measurement.dyn_xy<%a>
// CHECK-GENERIC: %m2 = "measurement.dyn_xy"(%a) : (!angle.type) -> !instrument.type<1, i1>
%m2 = measurement.dyn_xy<%a>
