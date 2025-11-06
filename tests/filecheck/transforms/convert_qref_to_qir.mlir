// RUN: quopt -p convert-qref-to-qir %s --split-input-file | filecheck %s

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
// CHECK-NEXT: %q2 = qir.qubit_allocate
%q2 = qu.alloc
// CHECK-NEXT: qir.ccx %q0, %q1, %q2
qref.gate<#gate.toffoli> %q0, %q1, %q2
// CHECK-NEXT: [[angle:%.*]] = arith.constant
// CHECK-NEXT: qir.rx<[[angle]]> %q1
qref.gate<#gate.rx<0.5pi>> %q1
// CHECK-NEXT: [[angle2:%.*]] = arith.constant
// CHECK-NEXT: qir.ry<[[angle2]]> %q1
qref.gate<#gate.ry<0.5pi>> %q1
// CHECK-NEXT: [[angle3:%.*]] = arith.constant
// CHECK-NEXT: qir.rz<[[angle3]]> %q1
qref.gate<#gate.rz<0.5pi>> %q1

// CHECK-NEXT: [[angle4:%.*]] = arith.constant
%a = angle.constant<0.5pi>
// CHECK-NEXT: [[angle5:%.*]] = arith.negf [[angle4]]
%negate = angle.negate %a
%b = "test.op"() : () -> i1
// CHECK: [[angle6:%.*]] = arith.negf [[angle4]]
// CHECK-NEXT: [[angle7:%.*]] = arith.select %b, [[angle6]], [[angle4]]
%cond_negate = angle.cond_negate %b, %a

%cHalf = arith.constant 0.5 : f64
// CHECK: [[angle8:%.*]] = arith.mulf [[angle7]], {{.*}}
%scale = angle.scale %cond_negate, %cHalf

// CHECK: [[angle9:%.*]] = arith.addf [[angle8]], [[angle7]]
%add = angle.add %scale, %cond_negate

%rx = gate.dyn_rx<%a>
// CHECK-NEXT: qir.rx<[[angle4]]> %q0
qref.dyn_gate<%rx> %q0

%ry = gate.dyn_ry<%negate>
// CHECK-NEXT: qir.ry<[[angle5]]> %q0
qref.dyn_gate<%ry> %q0

%rz = gate.dyn_rz<%add>
// CHECK-NEXT: qir.rz<[[angle9]]> %q0
qref.dyn_gate<%rz> %q0

%crx = gate.control %rx : !gate.type<1>
// CHECK-NEXT: qir.crx<[[angle4]]> %q0, %q1
qref.dyn_gate<%crx> %q0, %q1

%cry = gate.control %ry : !gate.type<1>
// CHECK-NEXT: qir.cry<[[angle5]]> %q0, %q1
qref.dyn_gate<%cry> %q0, %q1

%crz = gate.control %rz : !gate.type<1>
// CHECK-NEXT: qir.crz<[[angle9]]> %q0, %q1
qref.dyn_gate<%crz> %q0, %q1

// CHECK-NEXT: [[lhs:%.*]] = qir.m %q0
// CHECK-NEXT: qir.qubit_release %q0
// CHECK-NEXT: [[rhs:%.*]] = qir.result_get_one
// CHECK-NEXT: [[meas:%.+]] = qir.result_equal [[lhs]], [[rhs]]
%0 = qref.measure %q0
// CHECK-NEXT: qir.h %q1
// CHECK-NEXT: [[lhs2:%.*]] = qir.m %q1
// CHECK-NEXT: qir.qubit_release %q1
// CHECK-NEXT: [[rhs2:%.*]] = qir.result_get_one
// CHECK-NEXT: [[meas2:%.+]] = qir.result_equal [[lhs2]], [[rhs2]]
%1 = qref.measure<#measurement.x_basis> %q1
// CHECK-NEXT: [[angle10:%.*]] = arith.constant
// CHECK-NEXT: qir.rz<[[angle10]]> %q2
// CHECK-NEXT: qir.h %q2
// CHECK-NEXT: [[lhs3:%.*]] = qir.m %q2
// CHECK-NEXT: qir.qubit_release %q2
// CHECK-NEXT: [[rhs3:%.*]] = qir.result_get_one
// CHECK-NEXT: [[meas3:%.+]] = qir.result_equal [[lhs3]], [[rhs3]]
%2 = qref.measure<#measurement.xy<0.1pi>> %q2

// CHECK-NEXT "test.op"([[meas]], [[meas2]], [[meas3]])
"test.op"(%0, %1, %2) : (i1, i1, i1) -> ()

// -----

func.func @xy_measurement(%a : !angle.type) -> i1 {
  %q = qu.alloc
  %m = measurement.dyn_xy<%a>
  %0 = qref.dyn_measure<%m> %q
  func.return %0 : i1
}
// CHECK-LABEL: @xy_measurement
// CHECK-NEXT: %q = qir.qubit_allocate
// CHECK-NEXT: [[negation:%.*]] = arith.negf %a
// CHECK-NEXT: qir.rz<[[negation]]> %q
// CHECK-NEXT: qir.h %q
// CHECK-NEXT: [[lhs4:%.*]] = qir.m %q
// CHECK-NEXT: qir.qubit_release %q
// CHECK-NEXT: [[rhs4:%.*]] = qir.result_get_one
// CHECK-NEXT: [[meas4:%.+]] = qir.result_equal [[lhs4]], [[rhs4]]
// CHECK-NEXT: func.return [[meas4]]

// -----

func.func @qref_in_region(%q : !qu.bit, %p: i1) -> !qu.bit {
  %q3 = scf.if %p -> (!qu.bit) {
    qref.gate<#gate.z> %q
    scf.yield %q : !qu.bit
  } else {
    scf.yield %q : !qu.bit
  }
  func.return %q3 : !qu.bit
}

// CHECK-LABEL: @qref_in_region
// CHECK-NEXT: %q3 = scf.if %p -> (!qir.qubit) {
// CHECK-NEXT:   qir.z %q
// CHECK-NEXT:   scf.yield %q : !qir.qubit
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %q : !qir.qubit
// CHECK-NEXT: }
// CHECK-NEXT: func.return %q3 : !qir.qubit
