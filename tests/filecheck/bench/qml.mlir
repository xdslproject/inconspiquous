// RUN: TO_QIR

// From https://arxiv.org/pdf/2212.10742 Figure 1a

func.func @qml(%ql: !qu.bit, %qo: !qu.bit, %w: !angle.type, %b: !angle.type) -> (!qu.bit, !qu.bit) {
  %wn = angle.negate %w
  %bn = angle.negate %b

  %rxw = gate.dyn_rx<%w>
  %crxw = gate.control %rxw : !gate.type<1>

  %rxb = gate.dyn_rx<%b>

  %rxbn = gate.dyn_rx<%bn>

  %rxwn = gate.dyn_rx<%wn>
  %crxwn = gate.control %rxwn : !gate.type<1>

  %ql_res, %qo_res = scf.while(%ql_1 = %ql, %qo_1 = %qo)
    : (!qu.bit, !qu.bit) -> (!qu.bit, !qu.bit) {
    %qa = qu.alloc
    %ql_2, %qa_1 = qssa.dyn_gate<%crxw> %ql_1, %qa
    %qa_2 = qssa.dyn_gate<%rxb> %qa_1
    %qa_3, %qo_2 = qssa.gate<#gate.cx> %qa_2, %qo_1
    %qa_4 = qssa.gate<#gate.s_dagger> %qa_3
    %qa_5 = qssa.dyn_gate<%rxbn> %qa_4
    %ql_3, %qa_6 = qssa.dyn_gate<%crxwn> %ql_2, %qa_5

    %m = qssa.measure %qa_6

    scf.condition(%m) %ql_3, %qo_2 : !qu.bit, !qu.bit
  } do {
    ^bb0(%ql_1 : !qu.bit, %qo_1 : !qu.bit):
    %qo_2 = qssa.gate<#gate.rx<0.5pi>> %qo_1
    scf.yield %ql_1, %qo_2 : !qu.bit, !qu.bit
  }
  func.return %ql_res, %qo_res : !qu.bit, !qu.bit
}
// CHECK-LABEL: qml
// CHECK-NEXT:    %5 = fneg double %2
// CHECK-NEXT:    %6 = fneg double %3
// CHECK-NEXT:    br label %7
// CHECK-EMPTY:
// CHECK-NEXT:  7:                                                ; preds = %14, %4
// CHECK-NEXT:    %8 = phi ptr [ %15, %14 ], [ %0, %4 ]
// CHECK-NEXT:    %9 = phi ptr [ %16, %14 ], [ %1, %4 ]
// CHECK-NEXT:    %10 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__rx__ctl(double %2, ptr %8, ptr %10)
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double %3, ptr %10)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %10, ptr %9)
// CHECK-NEXT:    call void @__quantum__qis__s__adj(ptr %10)
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double %6, ptr %10)
// CHECK-NEXT:    call void @__quantum__qis__rx__ctl(double %5, ptr %8, ptr %10)
// CHECK-NEXT:    %11 = call ptr @__quantum__qis__m__body(ptr %10)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %10)
// CHECK-NEXT:    %12 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %13 = call i1 @__quantum__rt__result_equal(ptr %11, ptr %12)
// CHECK-NEXT:    br i1 %13, label %14, label %17
// CHECK-EMPTY:
// CHECK-NEXT:  14:                                               ; preds = %7
// CHECK-NEXT:    %15 = phi ptr [ %8, %7 ]
// CHECK-NEXT:    %16 = phi ptr [ %9, %7 ]
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double 0x3FF921FB54442D18, ptr %16)
// CHECK-NEXT:    br label %7
// CHECK-EMPTY:
// CHECK-NEXT:  17:                                               ; preds = %7
// CHECK-NEXT:    %18 = insertvalue { ptr, ptr } undef, ptr %8, 0
// CHECK-NEXT:    %19 = insertvalue { ptr, ptr } %18, ptr %9, 1
// CHECK-NEXT:    ret { ptr, ptr } %19
// CHECK-NEXT:  }
