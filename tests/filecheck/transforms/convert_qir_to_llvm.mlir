// RUN: quopt %s -p convert-qir-to-llvm --split-input-file | filecheck %s

// CHECK: "llvm.func"
// CHECK-SAME: sym_name = "__quantum__rt__qubit_allocate"
// CHECK: "llvm.call"
// CHECK-SAME: callee = @__quantum__rt__qubit_allocate

%0 = qir.qubit_allocate

// -----

// CHECK: "llvm.func"
// CHECK-NOT: "llvm.func"
// CHECK-COUNT-2: "llvm.call"

%0 = qir.qubit_allocate
%1 = qir.qubit_allocate

// -----

func.func @test(%q : !qir.qubit) {
  qir.h %q
  func.return
}
// CHECK-COUNT-1: "llvm.func"
// CHECK: func.func @test(%q : !llvm.ptr)
// CHECK-NEXT: llvm.call
// CHECK-SAME: @__quantum__qis__h__body

// -----

// CHECK-COUNT-5: "llvm.func"

// CHECK: "llvm.call"
// CHECK-SAME: @__quantum__rt__qubit_allocate
%0 = qir.qubit_allocate
// CHECK: "llvm.call"
// CHECK-SAME: @__quantum__rt__qubit_allocate
%1 = qir.qubit_allocate
// CHECK: "llvm.call"
// CHECK-SAME: @__quantum__qis__h__body
qir.h %0
// CHECK: "llvm.call"
// CHECK-SAME: @__quantum__qis__cnot__body
qir.cnot %0, %1
// CHECK: "llvm.call"
// CHECK-SAME: @__quantum__qis__m__body
%2 = qir.m %0
// CHECK: "llvm.call"
// CHECK-SAME: @__quantum__qis__m__body
%3 = qir.m %1
// CHECK: "llvm.call"
// CHECK-SAME: @__quantum__rt__result_equal
%4 = qir.result_equal %2, %3
