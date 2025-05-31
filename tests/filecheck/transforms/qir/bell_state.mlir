// RUN: quopt %s -p convert-qref-to-qir | filecheck %s

func.func @bell_state() {
    // CHECK: %[[Q0:.*]] = qir.qubit_allocate
    // CHECK: %[[Q1:.*]] = qir.qubit_allocate
    %q0 = qu.alloc
    %q1 = qu.alloc
    
    // CHECK: qir.h %[[Q0]]
    qref.gate<#gate.h> %q0
    
    // CHECK: qir.cx %[[Q0]], %[[Q1]]
    qref.gate<#gate.cx> %q0, %q1
    
    // CHECK: %[[M0:.*]] = qir.measure %[[Q0]]
    // CHECK: %{{.*}} = qir.read_result %[[M0]]
    %m0 = qref.measure %q0
    
    // CHECK: %[[M1:.*]] = qir.measure %[[Q1]]
    // CHECK: %{{.*}} = qir.read_result %[[M1]]
    %m1 = qref.measure %q1
    
    return
}
