// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK-LABEL: func.func @test_qreg_ops() {
// CHECK-NEXT:    %q0 = qu.alloc
// CHECK-NEXT:    %q1 = qu.alloc
// CHECK-NEXT:    %q2 = qu.alloc
// CHECK-NEXT:    %reg = "qu.from_bits"(%q0, %q1, %q2) : (!qu.bit, !qu.bit, !qu.bit) -> !qu.reg<3 : i64>
// CHECK-NEXT:    %split1, %split2 = "qu.split"(%reg) <{split_index = 1 : i64}> : (!qu.reg<3 : i64>) -> (!qu.reg<1 : i64>, !qu.reg<2 : i64>)
// CHECK-NEXT:    %combined = "qu.combine"(%split1, %split2) : (!qu.reg<1 : i64>, !qu.reg<2 : i64>) -> !qu.reg<3 : i64>
// CHECK-NEXT:    %bits, %bits_1, %bits_2 = "qu.to_bits"(%combined) : (!qu.reg<3 : i64>) -> (!qu.bit, !qu.bit, !qu.bit)
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "func.func"() <{sym_name = "test_qreg_ops", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:     %q0 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:     %q1 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:     %q2 = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
// CHECK-GENERIC-NEXT:     %reg = "qu.from_bits"(%q0, %q1, %q2) : (!qu.bit, !qu.bit, !qu.bit) -> !qu.reg<3 : i64>
// CHECK-GENERIC-NEXT:     %split1, %split2 = "qu.split"(%reg) <{split_index = 1 : i64}> : (!qu.reg<3 : i64>) -> (!qu.reg<1 : i64>, !qu.reg<2 : i64>)
// CHECK-GENERIC-NEXT:     %combined = "qu.combine"(%split1, %split2) : (!qu.reg<1 : i64>, !qu.reg<2 : i64>) -> !qu.reg<3 : i64>
// CHECK-GENERIC-NEXT:     %bits, %bits_1, %bits_2 = "qu.to_bits"(%combined) : (!qu.reg<3 : i64>) -> (!qu.bit, !qu.bit, !qu.bit)
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()

func.func @test_qreg_ops() {
  %q0 = "qu.alloc"() : () -> !qu.bit
  %q1 = "qu.alloc"() : () -> !qu.bit
  %q2 = "qu.alloc"() : () -> !qu.bit
  %reg = "qu.from_bits"(%q0, %q1, %q2) : (!qu.bit, !qu.bit, !qu.bit) -> !qu.reg<3>
  %split1, %split2 = "qu.split"(%reg) <{split_index = 1 : i64}> : (!qu.reg<3>) -> (!qu.reg<1>, !qu.reg<2>)
  %combined = "qu.combine"(%split1, %split2) : (!qu.reg<1>, !qu.reg<2>) -> !qu.reg<3>
  %bits, %bits_1, %bits_2 = "qu.to_bits"(%combined) : (!qu.reg<3>) -> (!qu.bit, !qu.bit, !qu.bit)
  func.return
}
