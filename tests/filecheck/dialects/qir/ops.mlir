// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %{{.*}} = qir.result_get_one
// CHECK-GENERIC: %{{.*}} = "qir.result_get_one"() : () -> !qir.result
%0 = qir.result_get_one

// CHECK-NEXT: %{{.*}} = qir.result_equal %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: %{{.*}} = "qir.result_equal"(%{{.*}}, %{{.*}}) : (!qir.result, !qir.result) -> i1
%1 = qir.result_equal %0, %0

// CHECK-NEXT: %{{.*}} = qir.qubit_allocate
// CHECK-GENERIC-NEXT: %{{.*}} = "qir.qubit_allocate"() : () -> !qir.qubit
%2 = qir.qubit_allocate

// CHECK-NEXT: %{{.*}} = qir.m %{{.*}}
// CHECK-GENERIC-NEXT: %{{.*}} = "qir.m"(%{{.*}}) : (!qir.qubit) -> !qir.result
%3 = qir.m %2

// CHECK-NEXT: qir.qubit_release %{{.*}}
// CHECK-GENERIC-NEXT: "qir.qubit_release"(%{{.*}}) : (!qir.qubit) -> ()
qir.qubit_release %2

%4, %5, %6 = "test.op"() : () -> (!qir.qubit, !qir.qubit, !qir.qubit)

// CHECK: qir.cnot %{{.*}}, %{{.*}}
// CHECK-GENERIC: "qir.cnot"(%{{.*}}, %{{.*}}) : (!qir.qubit, !qir.qubit) -> ()
qir.cnot %4, %5

// CHECK-NEXT: qir.cz %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.cz"(%{{.*}}, %{{.*}}) : (!qir.qubit, !qir.qubit) -> ()
qir.cz %4, %5

// CHECK-NEXT: qir.h %{{.*}}
// CHECK-GENERIC-NEXT: "qir.h"(%{{.*}}) : (!qir.qubit) -> ()
qir.h %4

%7 = arith.constant 1.000000e-01 : f64

// CHECK: qir.rx<%{{.*}}> %{{.*}}
// CHECK-GENERIC: "qir.rx"(%{{.*}}, %{{.*}}) : (f64, !qir.qubit) -> ()
qir.rx<%7> %4

// CHECK-NEXT: qir.ry<%{{.*}}> %{{.*}}
// CHECK-GENERIC-NEXT: "qir.ry"(%{{.*}}, %{{.*}}) : (f64, !qir.qubit) -> ()
qir.ry<%7> %4

// CHECK-NEXT: qir.rz<%{{.*}}> %{{.*}}
// CHECK-GENERIC-NEXT: "qir.rz"(%{{.*}}, %{{.*}}) : (f64, !qir.qubit) -> ()
qir.rz<%7> %4

// CHECK: qir.crx<%{{.*}}> %{{.*}}, %{{.*}}
// CHECK-GENERIC: "qir.crx"(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, !qir.qubit, !qir.qubit) -> ()
qir.crx<%7> %4, %5

// CHECK-NEXT: qir.cry<%{{.*}}> %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.cry"(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, !qir.qubit, !qir.qubit) -> ()
qir.cry<%7> %4, %5

// CHECK-NEXT: qir.crz<%{{.*}}> %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.crz"(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, !qir.qubit, !qir.qubit) -> ()
qir.crz<%7> %4, %5

// CHECK-NEXT: qir.s %{{.*}}
// CHECK-GENERIC-NEXT: "qir.s"(%{{.*}}) : (!qir.qubit) -> ()
qir.s %4

// CHECK-NEXT: qir.s_adj %{{.*}}
// CHECK-GENERIC-NEXT: "qir.s_adj"(%{{.*}}) : (!qir.qubit) -> ()
qir.s_adj %4

// CHECK-NEXT: qir.t %{{.*}}
// CHECK-GENERIC-NEXT: "qir.t"(%{{.*}}) : (!qir.qubit) -> ()
qir.t %4

// CHECK-NEXT: qir.t_adj %{{.*}}
// CHECK-GENERIC-NEXT: "qir.t_adj"(%{{.*}}) : (!qir.qubit) -> ()
qir.t_adj %4

// CHECK-NEXT: qir.x %{{.*}}
// CHECK-GENERIC-NEXT: "qir.x"(%{{.*}}) : (!qir.qubit) -> ()
qir.x %4

// CHECK-NEXT: qir.y %{{.*}}
// CHECK-GENERIC-NEXT: "qir.y"(%{{.*}}) : (!qir.qubit) -> ()
qir.y %4

// CHECK-NEXT: qir.z %{{.*}}
// CHECK-GENERIC-NEXT: "qir.z"(%{{.*}}) : (!qir.qubit) -> ()
qir.z %4

// CHECK-NEXT: qir.ccx %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.ccx"(%{{.*}}, %{{.*}}, %{{.*}}) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
qir.ccx %4, %5, %6
