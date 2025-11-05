// RUN: quopt %s -p convert-qssa-to-qref,lower-xzs-to-select,cse,canonicalize,lower-dyn-gate-to-scf,canonicalize,convert-qref-to-qir,convert-qir-to-llvm | mlir-opt -p 'builtin.module(convert-scf-to-cf,canonicalize,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm)' | mlir-translate --mlir-to-llvmir | filecheck %s

// MBQC cx gate.

func.func @cx(%ctrl: !qu.bit, %tgt: !qu.bit) -> (!qu.bit, !qu.bit) {
  %a1 = qu.alloc<#qu.plus>
  %a2 = qu.alloc<#qu.plus>
  %tgt_1, %a1_1 = qssa.gate<#gate.cz> %tgt, %a1
  %ctrl_1, %a1_2 = qssa.gate<#gate.cz> %ctrl, %a1_1
  %a1_3, %a2_1 = qssa.gate<#gate.cz> %a1_2, %a2
  %m1 = qssa.measure<#measurement.x_basis> %tgt_1
  %cFalse = arith.constant false
  %g1 = gate.xz %cFalse, %m1
  %m2 = qssa.measure<#measurement.x_basis> %a1_3
  %g2 = gate.xz %m2, %m1
  %ctrl_2 = qssa.dyn_gate<%g1> %ctrl_1
  %a2_2 = qssa.dyn_gate<%g2> %a2_1
  func.return %ctrl_2, %a2_2 : !qu.bit, !qu.bit
}
// CHECK-LABEL: cx
// CHECK-NEXT:    %3 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %3)
// CHECK-NEXT:    %4 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %1, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %0, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %3, ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %1)
// CHECK-NEXT:    %5 = call ptr @__quantum__qis__m__body(ptr %1)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %1)
// CHECK-NEXT:    %6 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %7 = call i1 @__quantum__rt__result_equal(ptr %5, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %3)
// CHECK-NEXT:    %8 = call ptr @__quantum__qis__m__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %3)
// CHECK-NEXT:    %9 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %10 = call i1 @__quantum__rt__result_equal(ptr %8, ptr %9)
// CHECK-NEXT:    br i1 %7, label %11, label %12
// CHECK-EMPTY:
// CHECK-NEXT:  11:                                               ; preds = %2
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %0)
// CHECK-NEXT:    br label %12
// CHECK-EMPTY:
// CHECK-NEXT:  12:                                               ; preds = %11, %2
// CHECK-NEXT:    br i1 %10, label %13, label %16
// CHECK-EMPTY:
// CHECK-NEXT:  13:                                               ; preds = %12
// CHECK-NEXT:    br i1 %7, label %14, label %15
// CHECK-EMPTY:
// CHECK-NEXT:  14:                                               ; preds = %13
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %4)
// CHECK-NEXT:    br label %18
// CHECK-EMPTY:
// CHECK-NEXT:  15:                                               ; preds = %13
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %4)
// CHECK-NEXT:    br label %18
// CHECK-EMPTY:
// CHECK-NEXT:  16:                                               ; preds = %12
// CHECK-NEXT:    br i1 %7, label %17, label %18
// CHECK-EMPTY:
// CHECK-NEXT:  17:                                               ; preds = %16
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %4)
// CHECK-NEXT:    br label %18
// CHECK-EMPTY:
// CHECK-NEXT:  18:                                               ; preds = %14, %15, %17, %16
// CHECK-NEXT:    %19 = insertvalue { ptr, ptr } undef, ptr %0, 0
// CHECK-NEXT:    %20 = insertvalue { ptr, ptr } %19, ptr %4, 1
// CHECK-NEXT:    ret { ptr, ptr } %20
// CHECK-NEXT:  }
