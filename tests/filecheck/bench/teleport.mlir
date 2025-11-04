// RUN: quopt %s -p convert-qssa-to-qref,lower-xzs-to-select,cse,canonicalize,lower-dyn-gate-to-scf,canonicalize,convert-qref-to-qir,convert-qir-to-llvm | mlir-opt -p 'builtin.module(convert-scf-to-cf,canonicalize,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm)' | mlir-translate --mlir-to-llvmir | filecheck %s

// From https://arxiv.org/pdf/2204.04185

func.func @teleport(%q : !qu.bit) -> !qu.bit {
  %a1 = qu.alloc
  %a2 = qu.alloc
  %a3 = qu.alloc
  %a4 = qu.alloc
  %a5 = qu.alloc
  %a6 = qu.alloc

  %a1_1 = qssa.gate<#gate.h> %a1
  %a3_1 = qssa.gate<#gate.h> %a3
  %a5_1 = qssa.gate<#gate.h> %a5

  %a1_2, %a2_1 = qssa.gate<#gate.cx> %a1_1, %a2
  %a3_2, %a4_1 = qssa.gate<#gate.cx> %a3_1, %a4
  %a5_2, %a6_1 = qssa.gate<#gate.cx> %a5_1, %a6

  %q_1, %a1_3 = qssa.gate<#gate.cx> %q, %a1_2
  %a2_2, %a3_3 = qssa.gate<#gate.cx> %a2_1, %a3_2
  %a4_2, %a5_3 = qssa.gate<#gate.cx> %a4_1, %a5_2

  %q_2 = qssa.gate<#gate.h> %q_1
  %a2_3 = qssa.gate<#gate.h> %a2_2
  %a4_3 = qssa.gate<#gate.h> %a4_2

  %m0 = qssa.measure %q_2
  %m1 = qssa.measure %a1_3
  %m2 = qssa.measure %a2_3
  %m3 = qssa.measure %a3_3
  %m4 = qssa.measure %a4_3
  %m5 = qssa.measure %a5_3

  %x1 = arith.xori %m1, %m3 : i1
  %x = arith.xori %x1, %m5 : i1

  %z1 = arith.xori %m0, %m2 : i1
  %z = arith.xori %z1, %m4 : i1

  %c0 = arith.constant false
  %g1 = gate.xz %x, %c0
  %a6_2 = qssa.dyn_gate<%g1> %a6_1
  %g2 = gate.xz %c0, %z
  %a6_3 = qssa.dyn_gate<%g2> %a6_2

  func.return %a6_3 : !qu.bit
}
// CHECK-LABEL:  @teleport
// CHECK-NEXT:    %2 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %3 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %4 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %5 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %6 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %7 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %2, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %4, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %6, ptr %7)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %0, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %3, ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %5, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %0)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %8 = call ptr @__quantum__qis__m__body(ptr %0)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %0)
// CHECK-NEXT:    %9 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %10 = call i1 @__quantum__rt__result_equal(ptr %8, ptr %9)
// CHECK-NEXT:    %11 = call ptr @__quantum__qis__m__body(ptr %2)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %2)
// CHECK-NEXT:    %12 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %13 = call i1 @__quantum__rt__result_equal(ptr %11, ptr %12)
// CHECK-NEXT:    %14 = call ptr @__quantum__qis__m__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %3)
// CHECK-NEXT:    %15 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %16 = call i1 @__quantum__rt__result_equal(ptr %14, ptr %15)
// CHECK-NEXT:    %17 = call ptr @__quantum__qis__m__body(ptr %4)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %4)
// CHECK-NEXT:    %18 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %19 = call i1 @__quantum__rt__result_equal(ptr %17, ptr %18)
// CHECK-NEXT:    %20 = call ptr @__quantum__qis__m__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %5)
// CHECK-NEXT:    %21 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %22 = call i1 @__quantum__rt__result_equal(ptr %20, ptr %21)
// CHECK-NEXT:    %23 = call ptr @__quantum__qis__m__body(ptr %6)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %6)
// CHECK-NEXT:    %24 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %25 = call i1 @__quantum__rt__result_equal(ptr %23, ptr %24)
// CHECK-NEXT:    %26 = xor i1 %13, %19
// CHECK-NEXT:    %27 = xor i1 %26, %25
// CHECK-NEXT:    %28 = xor i1 %10, %16
// CHECK-NEXT:    %29 = xor i1 %28, %22
// CHECK-NEXT:    br i1 %27, label %30, label %31
// CHECK-EMPTY:
// CHECK-NEXT:  30:                                               ; preds = %1
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %7)
// CHECK-NEXT:    br label %31
// CHECK-EMPTY:
// CHECK-NEXT:  31:                                               ; preds = %30, %1
// CHECK-NEXT:    br i1 %29, label %32, label %33
// CHECK-EMPTY:
// CHECK-NEXT:  32:                                               ; preds = %31
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %7)
// CHECK-NEXT:    br label %33
// CHECK-EMPTY:
// CHECK-NEXT:  33:                                               ; preds = %32, %31
// CHECK-NEXT:    ret ptr %7
// CHECK-NEXT:  }
