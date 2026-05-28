// RUN: TO_QIR

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
// CHECK-NEXT:    %6 = call i1 @__quantum__rt__read_result__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %1)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %3)
// CHECK-NEXT:    %7 = call ptr @__quantum__qis__m__body(ptr %3)
// CHECK-NEXT:    %8 = call i1 @__quantum__rt__read_result__body(ptr %7)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %3)
// CHECK-NEXT:    br i1 %6, label %9, label %10
// CHECK-EMPTY:
// CHECK-NEXT:  9:                                                ; preds = %2
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %0)
// CHECK-NEXT:    br label %10
// CHECK-EMPTY:
// CHECK-NEXT:  10:                                               ; preds = %9, %2
// CHECK-NEXT:    br i1 %8, label %11, label %14
// CHECK-EMPTY:
// CHECK-NEXT:  11:                                               ; preds = %10
// CHECK-NEXT:    br i1 %6, label %12, label %13
// CHECK-EMPTY:
// CHECK-NEXT:  12:                                               ; preds = %11
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %4)
// CHECK-NEXT:    br label %16
// CHECK-EMPTY:
// CHECK-NEXT:  13:                                               ; preds = %11
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %4)
// CHECK-NEXT:    br label %16
// CHECK-EMPTY:
// CHECK-NEXT:  14:                                               ; preds = %10
// CHECK-NEXT:    br i1 %6, label %15, label %16
// CHECK-EMPTY:
// CHECK-NEXT:  15:                                               ; preds = %14
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %4)
// CHECK-NEXT:    br label %16
// CHECK-EMPTY:
// CHECK-NEXT:  16:                                               ; preds = %12, %13, %15, %14
// CHECK-NEXT:    %17 = insertvalue { ptr, ptr } poison, ptr %0, 0
// CHECK-NEXT:    %18 = insertvalue { ptr, ptr } %17, ptr %4, 1
// CHECK-NEXT:    ret { ptr, ptr } %18
// CHECK-NEXT:  }
