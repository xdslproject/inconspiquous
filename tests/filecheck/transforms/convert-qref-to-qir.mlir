// RUN: inconspiquous-opt --convert-qref-to-qir %s | FileCheck %s

// CHECK-LABEL: func.func @test_basic
func.func @test_basic() {
  // CHECK: %[[Q:.*]] = qu.alloc
  %q = qu.alloc
  // CHECK: qref.gate<#gate.h> %[[Q]]
  qref.gate<#gate.h> %q
  // CHECK: %[[M:.*]] = qref.measure %[[Q]]
  %m = qref.measure %q
  return
}

// CHECK-LABEL: func.func @test_all_gates
func.func @test_all_gates() {
  // CHECK: %[[Q:.*]] = qu.alloc
  %q = qu.alloc
  // CHECK: qir.h %[[Q]]
  qref.gate<#gate.h> %q
  // CHECK: qref.gate<#gate.x> %[[Q]]
  qref.gate<#gate.x> %q
  // CHECK: qref.gate<#gate.y> %[[Q]]
  qref.gate<#gate.y> %q
  // CHECK: qref.gate<#gate.z> %[[Q]]
  qref.gate<#gate.z> %q
  // CHECK: qref.gate<#gate.s> %[[Q]]
  qref.gate<#gate.s> %q
  // CHECK: qref.gate<#gate.t> %[[Q]]
  qref.gate<#gate.t> %q
  return
}

// CHECK-LABEL: func.func @test_two_qubit_gates
func.func @test_two_qubit_gates() {
  // CHECK: %[[Q0:.*]] = qu.alloc
  %q0 = qu.alloc
  // CHECK: %[[Q1:.*]] = qu.alloc
  %q1 = qu.alloc
  // CHECK: qref.gate<#gate.cx> %[[Q0]], %[[Q1]]
  qref.gate<#gate.cx> %q0, %q1
  // CHECK: qref.gate<#gate.cz> %[[Q0]], %[[Q1]]
  qref.gate<#gate.cz> %q0, %q1
  return
}

// CHECK-LABEL: func.func @test_bell_state
func.func @test_bell_state() {
  // CHECK: %[[Q0:.*]] = qu.alloc
  %q0 = qu.alloc
  // CHECK: %[[Q1:.*]] = qu.alloc
  %q1 = qu.alloc
  // CHECK: qref.gate<#gate.h> %[[Q0]]
  qref.gate<#gate.h> %q0
  // CHECK: qref.gate<#gate.cx> %[[Q0]], %[[Q1]]
  qref.gate<#gate.cx> %q0, %q1
  // CHECK: %[[M0:.*]] = qref.measure %[[Q0]]
  %m0 = qref.measure %q0
  // CHECK: %[[M1:.*]] = qref.measure %[[Q1]]
  %m1 = qref.measure %q1
  return
} 
