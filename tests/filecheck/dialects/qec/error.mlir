// RUN: quopt %s --parsing-diagnostics | filecheck %s

// CHECK: Stabilizer should be one of 'I', 'X', 'Y', or 'Z'
"test.op"() {attr = #qec.stabilizer<A>} : () -> ()
