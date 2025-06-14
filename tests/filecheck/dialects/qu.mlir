// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

//--- Qubit Allocation Tests ---//

// CHECK-LABEL: func.func @test_qubit_allocs() {
// CHECK-NEXT:    %q = qu.alloc
// CHECK-NEXT:    %q2 = qu.alloc<#qu.plus>
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }
// CHECK-GENERIC-LABEL: "func.func"() <{sym_name = "test_qubit_allocs", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:    %q = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:    %q2 = "qu.alloc"() <{alloc = #qu.plus}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:    "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()

func.func @test_qubit_allocs() {
  %q = qu.alloc
  %q2 = qu.alloc<#qu.plus>
  func.return
}

//---

//--- Qubit Register Tests ---//

// CHECK-LABEL: func.func @test_qreg_ops() {
// CHECK-NEXT:    %q0 = qu.alloc
// CHECK-NEXT:    %q1 = qu.alloc
// CHECK-NEXT:    %q2 = qu.alloc
// CHECK-NEXT:    %reg = qu.from_bits %q0, %q1, %q2 : !qu.reg<3>
// CHECK-NEXT:    %split1, %split2 = qu.split %reg : !qu.reg<3> -> !qu.reg<1>, !qu.reg<2>
// CHECK-NEXT:    %combined = qu.combine %split1, %split2 : !qu.reg<1>, !qu.reg<2> -> !qu.reg<3>
// CHECK-NEXT:    %bits, %bits_1, %bits_2 = qu.to_bits %combined : !qu.reg<3>
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

// CHECK-GENERIC-LABEL: "func.func"() <{sym_name = "test_qreg_ops", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:    %q0 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:    %q1 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:    %q2 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:    %reg = "qu.from_bits"(%q0, %q1, %q2) : (!qu.bit, !qu.bit, !qu.bit) -> !qu.reg<3>
// CHECK-GENERIC-NEXT:    %split1, %split2 = "qu.split"(%reg) : (!qu.reg<3>) -> (!qu.reg<1>, !qu.reg<2>)
// CHECK-GENERIC-NEXT:    %combined = "qu.combine"(%split1, %split2) : (!qu.reg<1>, !qu.reg<2>) -> !qu.reg<3>
// CHECK-GENERIC-NEXT:    %bits, %bits_1, %bits_2 = "qu.to_bits"(%combined) : (!qu.reg<3>) -> (!qu.bit, !qu.bit, !qu.bit)
// CHECK-GENERIC-NEXT:    "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()

func.func @test_qreg_ops() {
  %q0 = qu.alloc
  %q1 = qu.alloc
  %q2 = qu.alloc
  %reg = qu.from_bits %q0, %q1, %q2 : !qu.reg<3>
  %split1, %split2 = qu.split %reg : !qu.reg<3> -> !qu.reg<1>, !qu.reg<2>
  %combined = qu.combine %split1, %split2 : !qu.reg<1>, !qu.reg<2> -> !qu.reg<3>
  %bits, %bits_1, %bits_2 = qu.to_bits %combined : !qu.reg<3>
  func.return
}
