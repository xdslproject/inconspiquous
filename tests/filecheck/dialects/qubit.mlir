// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK-LABEL: func.func @test_qubit_allocs() {
// CHECK-NEXT:    %q = qu.alloc
// CHECK-NEXT:    %q2 = qu.alloc<#qu.plus>
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "func.func"() <{sym_name = "test_qubit_allocs", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:     %q = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:     %q2 = "qu.alloc"() <{alloc = #qu.plus}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()

func.func @test_qubit_allocs() {
  %q = "qu.alloc"() : () -> !qu.bit
  %q2 = "qu.alloc"() <{alloc = #qu.plus}> : () -> !qu.bit
  func.return
}
