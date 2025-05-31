// RUN: quopt %s -p convert-qref-to-qir | filecheck %s

func.func @test() {
    %q = qu.alloc
    qref.gate<#gate.h> %q
    %m = qref.measure %q
    return
}

// CHECK: qir.qubit_allocate
// CHECK: qir.h
// CHECK: qir.measure
// CHECK: qir.read_result
