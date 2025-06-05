// RUN: QUOPT_ROUNDTRIP


// CHECK-LABEL: func.func @test_qubit_allocs() {
// CHECK-NEXT:    %q = "qu.alloc"() : () -> !qu.bit
// CHECK-NEXT:    %q2 = "qu.alloc"() <{alloc = #qu.plus}> : () -> !qu.bit
// CHECK-NEXT:    "func.return"() : () -> ()
// CHECK-NEXT:  }
func.func @test_qubit_allocs() {
  %q = "qu.alloc"() : () -> !qu.bit
  %q2 = "qu.alloc"() <{alloc = #qu.plus}> : () -> !qu.bit
  "func.return"() : () -> ()
}