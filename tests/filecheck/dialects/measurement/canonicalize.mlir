// RUN: quopt -p canonicalize %s | filecheck %s

%a = gate.constant_angle<0>

%m = measurement.dyn_xy<%a>

"test.op"(%m) : (!measurement.type<1>) -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = measurement.constant #measurement.xy<0>
// CHECK-NEXT:    "test.op"(%m) : (!measurement.type<1>) -> ()
// CHECK-NEXT:  }
