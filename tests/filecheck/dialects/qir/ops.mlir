// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %{{.*}} = qir.result_get_one
// CHECK-GENERIC: %{{.*}} = "qir.result_get_one"() : () -> !qir.result
%0 = qir.result_get_one

// CHECK-NEXT: %{{.*}} = qir.result_equal %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: %{{.*}} = "qir.result_equal"(%{{.*}}, %{{.*}}) : (!qir.result, !qir.result) -> i1
%1 = qir.result_equal %0, %0

// CHECK-NEXT: %{{.*}} = qir.read_result %{{.*}}
// CHECK-GENERIC-NEXT: %{{.*}} = "qir.read_result"(%{{.*}}) : (!qir.result) -> i1
%2 = qir.read_result %0

// CHECK-NEXT: %{{.*}} = qir.qubit_allocate
// CHECK-GENERIC-NEXT: %{{.*}} = "qir.qubit_allocate"() : () -> !qir.qubit
%3 = qir.qubit_allocate

// CHECK-NEXT: %{{.*}} = qir.m %{{.*}}
// CHECK-GENERIC-NEXT: %{{.*}} = "qir.m"(%{{.*}}) : (!qir.qubit) -> !qir.result
%4 = qir.m %3

// CHECK-NEXT: qir.qubit_release %{{.*}}
// CHECK-GENERIC-NEXT: "qir.qubit_release"(%{{.*}}) : (!qir.qubit) -> ()
qir.qubit_release %3

%5, %6, %7 = "test.op"() : () -> (!qir.qubit, !qir.qubit, !qir.qubit)

// CHECK: qir.cnot %{{.*}}, %{{.*}}
// CHECK-GENERIC: "qir.cnot"(%{{.*}}, %{{.*}}) : (!qir.qubit, !qir.qubit) -> ()
qir.cnot %5, %6

// CHECK-NEXT: qir.cz %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.cz"(%{{.*}}, %{{.*}}) : (!qir.qubit, !qir.qubit) -> ()
qir.cz %5, %6

// CHECK-NEXT: qir.h %{{.*}}
// CHECK-GENERIC-NEXT: "qir.h"(%{{.*}}) : (!qir.qubit) -> ()
qir.h %5

%8 = arith.constant 1.000000e-01 : f64

// CHECK: qir.rx<%{{.*}}> %{{.*}}
// CHECK-GENERIC: "qir.rx"(%{{.*}}, %{{.*}}) : (f64, !qir.qubit) -> ()
qir.rx<%8> %5

// CHECK-NEXT: qir.ry<%{{.*}}> %{{.*}}
// CHECK-GENERIC-NEXT: "qir.ry"(%{{.*}}, %{{.*}}) : (f64, !qir.qubit) -> ()
qir.ry<%8> %5

// CHECK-NEXT: qir.rz<%{{.*}}> %{{.*}}
// CHECK-GENERIC-NEXT: "qir.rz"(%{{.*}}, %{{.*}}) : (f64, !qir.qubit) -> ()
qir.rz<%8> %5

// CHECK-NEXT: qir.rzz<%{{.*}}> %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.rzz"(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, !qir.qubit, !qir.qubit) -> ()
qir.rzz<%8> %5, %6

// CHECK-NEXT: qir.s %{{.*}}
// CHECK-GENERIC-NEXT: "qir.s"(%{{.*}}) : (!qir.qubit) -> ()
qir.s %5

// CHECK-NEXT: qir.s_adj %{{.*}}
// CHECK-GENERIC-NEXT: "qir.s_adj"(%{{.*}}) : (!qir.qubit) -> ()
qir.s_adj %5

// CHECK-NEXT: qir.t %{{.*}}
// CHECK-GENERIC-NEXT: "qir.t"(%{{.*}}) : (!qir.qubit) -> ()
qir.t %5

// CHECK-NEXT: qir.t_adj %{{.*}}
// CHECK-GENERIC-NEXT: "qir.t_adj"(%{{.*}}) : (!qir.qubit) -> ()
qir.t_adj %5

// CHECK-NEXT: qir.x %{{.*}}
// CHECK-GENERIC-NEXT: "qir.x"(%{{.*}}) : (!qir.qubit) -> ()
qir.x %5

// CHECK-NEXT: qir.y %{{.*}}
// CHECK-GENERIC-NEXT: "qir.y"(%{{.*}}) : (!qir.qubit) -> ()
qir.y %5

// CHECK-NEXT: qir.z %{{.*}}
// CHECK-GENERIC-NEXT: "qir.z"(%{{.*}}) : (!qir.qubit) -> ()
qir.z %5

// CHECK-NEXT: qir.ccx %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.ccx"(%{{.*}}, %{{.*}}, %{{.*}}) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
qir.ccx %5, %6, %7

%c8 = arith.constant 8 : i32
%c1 = arith.constant 1 : i64
%c0 = arith.constant 0 : i64

// CHECK: %9 = qir.array_create_1d %c8, %c1
// CHECK-GENERIC: %9 = "qir.array_create_1d"(%c8, %c1) : (i32, i64) -> !qir.array
%9 = qir.array_create_1d %c8, %c1

// CHECK-NEXT: %10 = qir.get_element_ptr %9[%c0]
// CHECK-GENERIC-NEXT: "qir.get_element_ptr"(%9, %c0) : (!qir.array, i64) -> !llvm.ptr
%10 = qir.get_element_ptr %9[%c0]

// CHECK: qir.crx<%{{.*}}> %{{.*}}, %{{.*}}
// CHECK-GENERIC: "qir.crx"(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, !qir.array, !qir.qubit) -> ()
qir.crx<%8> %9, %6

// CHECK-NEXT: qir.cry<%{{.*}}> %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.cry"(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, !qir.array, !qir.qubit) -> ()
qir.cry<%8> %9, %6

// CHECK-NEXT: qir.crz<%{{.*}}> %{{.*}}, %{{.*}}
// CHECK-GENERIC-NEXT: "qir.crz"(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, !qir.array, !qir.qubit) -> ()
qir.crz<%8> %9, %6
