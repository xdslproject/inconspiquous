// RUN: QUOPT_ROUNDTRIP

"test.op"() {gate = #qft.n<5>} : () -> ()

// CHECK: "test.op"() {gate = #qft.n<5>} : () -> ()

// CHECK: %q0 = qubit.alloc
%q0 = qubit.alloc

// CHECK: qref.gate<#qft.h> %q0
qref.gate<#qft.h> %q0
