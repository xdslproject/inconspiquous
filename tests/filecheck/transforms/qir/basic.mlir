// RUN: quopt %s -p convert-qref-to-qir | filecheck %s

func.func @simple_circuit() {
    // CHECK: %{{.*}} = qir.qubit_allocate
    %q0 = qu.alloc
    
    // CHECK: qir.h %{{.*}}
    qref.gate<#gate.h> %q0
    
    // CHECK: qir.x %{{.*}}
    qref.gate<#gate.x> %q0
    
    // CHECK: %{{.*}} = qir.measure %{{.*}}
    // CHECK: %{{.*}} = qir.read_result %{{.*}}
    %m = qref.measure %q0
    
    return
}
