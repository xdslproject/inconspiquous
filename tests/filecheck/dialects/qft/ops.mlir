// RUN: QUOPT_ROUNDTRIP

"test.op"() {gate = #qft.n<5>} : () -> ()

// CHECK: "test.op"() {gate = #qft.n<5>} : () -> ()
