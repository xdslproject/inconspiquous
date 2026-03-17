// RUN: quopt -p canonicalize %s | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = instrument.constant #measurement.xy<0>
// CHECK-NEXT:    "test.op"(%m) : (!instrument.type<1, i1>) -> ()
// CHECK-NEXT:  }
%a = angle.constant<0>

%m = measurement.dyn_xy<%a>

"test.op"(%m) : (!instrument.type<1, i1>) -> ()
