// RUN: quopt -p convert-qref-to-qir %s | filecheck %s

// CHECK:      %q0 = qir.qubit_allocate
%q0 = qu.alloc
// CHECK-NEXT: %q1 = qir.qubit_allocate
// CHECK-NEXT: qir.h %q1
%q1 = qu.alloc<#qu.plus>
// CHECK-NEXT: qir.cnot %q0, %q1
qref.gate<#gate.cx> %q0, %q1
// CHECK-NEXT: qir.cz %q0, %q1
qref.gate<#gate.cz> %q0, %q1
// CHECK-NEXT: qir.h %q0
qref.gate<#gate.h> %q0
// CHECK-NEXT: qir.s %q0
qref.gate<#gate.s> %q0
// CHECK-NEXT: qir.s_adj %q0
qref.gate<#gate.s_dagger> %q0
// CHECK-NEXT: qir.t %q0
qref.gate<#gate.t> %q0
// CHECK-NEXT: qir.t_adj %q0
qref.gate<#gate.t_dagger> %q0
// CHECK-NEXT: qir.x %q0
qref.gate<#gate.x> %q0
// CHECK-NEXT: qir.y %q0
qref.gate<#gate.y> %q0
// CHECK-NEXT: qir.z %q0
qref.gate<#gate.z> %q0
// CHECK-NEXT: [[angle:%.*]] = arith.constant
// CHECK-NEXT: qir.rz<[[angle]]> %q1
qref.gate<#gate.rz<0.5pi>> %q1
// CHECK-NEXT: [[lhs:%.*]] = qir.m %q0
// CHECK-NEXT: qir.qubit_release %q0
// CHECK-NEXT: [[rhs:%.*]] = qir.result_get_one
// CHECK-NEXT: [[meas:%.+]] = qir.result_equal [[lhs]], [[rhs]]
%0 = qref.measure %q0

// CHECK-NEXT "test.op"([[meas]])
"test.op"(%0) : (i1) -> ()

func.func @qref_in_region(%q : !qu.bit, %p: i1) -> !qu.bit {
  %q2 = scf.if %p -> (!qu.bit) {
    qref.gate<#gate.z> %q
    scf.yield %q : !qu.bit
  } else {
    scf.yield %q : !qu.bit
  }
  func.return %q2 : !qu.bit
}

// CHECK:      func.func @qref_in_region
// CHECK-NEXT: %q2 = scf.if %p -> (!qir.qubit) {
// CHECK-NEXT:   qir.z %q
// CHECK-NEXT:   scf.yield %q : !qir.qubit
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %q : !qir.qubit
// CHECK-NEXT: }
// CHECK-NEXT: func.return %q2 : !qir.qubit
