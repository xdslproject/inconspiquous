// RUN: TO_QIR

// Ansatz for Max-cut QAOA on the following graph:

//    1----4
//   /|    |
//  5 |    |
//   \|    |
//    2----3

// See https://arxiv.org/pdf/1411.4028

func.func @qaoa_problem(
  %theta : !angle.type,
  %q1 : !qu.bit,
  %q2 : !qu.bit,
  %q3 : !qu.bit,
  %q4 : !qu.bit,
  %q5 : !qu.bit
) -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit) {
  %g = gate.dyn_rzz<%theta>

  %q1_1, %q2_1 = qssa.dyn_gate<%g> %q1, %q2
  %q2_2, %q3_1 = qssa.dyn_gate<%g> %q2_1, %q3
  %q3_2, %q4_1 = qssa.dyn_gate<%g> %q3_1, %q4
  %q4_2, %q1_2 = qssa.dyn_gate<%g> %q4_1, %q1_1
  %q1_3, %q5_1 = qssa.dyn_gate<%g> %q1_2, %q5
  %q2_3, %q5_2 = qssa.dyn_gate<%g> %q2_2, %q5_1

  func.return %q1_3, %q2_3, %q3_2, %q4_2, %q5_2 : !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit
}
// CHECK-LABEL: define { ptr, ptr, ptr, ptr, ptr } @qaoa_problem
// CHECK-NEXT:    call void @__quantum__qis__rzz__body(double %0, ptr %1, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__rzz__body(double %0, ptr %2, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__rzz__body(double %0, ptr %3, ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__rzz__body(double %0, ptr %4, ptr %1)
// CHECK-NEXT:    call void @__quantum__qis__rzz__body(double %0, ptr %1, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__rzz__body(double %0, ptr %2, ptr %5)
// CHECK-NEXT:    %7 = insertvalue { ptr, ptr, ptr, ptr, ptr } undef, ptr %1, 0
// CHECK-NEXT:    %8 = insertvalue { ptr, ptr, ptr, ptr, ptr } %7, ptr %2, 1
// CHECK-NEXT:    %9 = insertvalue { ptr, ptr, ptr, ptr, ptr } %8, ptr %3, 2
// CHECK-NEXT:    %10 = insertvalue { ptr, ptr, ptr, ptr, ptr } %9, ptr %4, 3
// CHECK-NEXT:    %11 = insertvalue { ptr, ptr, ptr, ptr, ptr } %10, ptr %5, 4
// CHECK-NEXT:    ret { ptr, ptr, ptr, ptr, ptr } %11
// CHECK-NEXT:  }

func.func @qaoa_mixer(
  %beta : !angle.type,
  %q1 : !qu.bit,
  %q2 : !qu.bit,
  %q3 : !qu.bit,
  %q4 : !qu.bit,
  %q5 : !qu.bit
) -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit) {
  %g = gate.dyn_rx<%beta>

  %q1_1 = qssa.dyn_gate<%g> %q1
  %q2_1 = qssa.dyn_gate<%g> %q2
  %q3_1 = qssa.dyn_gate<%g> %q3
  %q4_1 = qssa.dyn_gate<%g> %q4
  %q5_1 = qssa.dyn_gate<%g> %q5

  func.return %q1_1, %q2_1, %q3_1, %q4_1, %q5_1 : !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit
}
// CHECK-LABEL: define { ptr, ptr, ptr, ptr, ptr } @qaoa_mixer
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double %0, ptr %1)
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double %0, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double %0, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double %0, ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__rx__body(double %0, ptr %5)
// CHECK-NEXT:    %7 = insertvalue { ptr, ptr, ptr, ptr, ptr } undef, ptr %1, 0
// CHECK-NEXT:    %8 = insertvalue { ptr, ptr, ptr, ptr, ptr } %7, ptr %2, 1
// CHECK-NEXT:    %9 = insertvalue { ptr, ptr, ptr, ptr, ptr } %8, ptr %3, 2
// CHECK-NEXT:    %10 = insertvalue { ptr, ptr, ptr, ptr, ptr } %9, ptr %4, 3
// CHECK-NEXT:    %11 = insertvalue { ptr, ptr, ptr, ptr, ptr } %10, ptr %5, 4
// CHECK-NEXT:    ret { ptr, ptr, ptr, ptr, ptr } %11
// CHECK-NEXT:  }

func.func @qaoa_ansatz(
  %theta1 : !angle.type,
  %theta2 : !angle.type,
  %beta1 : !angle.type,
  %beta2 : !angle.type
) -> (i1, i1, i1, i1, i1) {
  %q1 = qu.alloc<#qu.plus>
  %q2 = qu.alloc<#qu.plus>
  %q3 = qu.alloc<#qu.plus>
  %q4 = qu.alloc<#qu.plus>
  %q5 = qu.alloc<#qu.plus>

  %q1_1, %q2_1, %q3_1, %q4_1, %q5_1 = func.call @qaoa_problem(%theta1, %q1, %q2, %q3, %q4, %q5)
    : (!angle.type, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)
    -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)

  %q1_2, %q2_2, %q3_2, %q4_2, %q5_2 = func.call @qaoa_mixer(%beta1, %q1_1, %q2_1, %q3_1, %q4_1, %q5_1)
    : (!angle.type, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)
    -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)

  %q1_3, %q2_3, %q3_3, %q4_3, %q5_3 = func.call @qaoa_problem(%theta2, %q1_2, %q2_2, %q3_2, %q4_2, %q5_2)
    : (!angle.type, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)
    -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)

  %q1_4, %q2_4, %q3_4, %q4_4, %q5_4 = func.call @qaoa_mixer(%beta2, %q1_3, %q2_3, %q3_3, %q4_3, %q5_3)
    : (!angle.type, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)
    -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit)

  %m1 = qssa.measure %q1_4
  %m2 = qssa.measure %q2_4
  %m3 = qssa.measure %q3_4
  %m4 = qssa.measure %q4_4
  %m5 = qssa.measure %q5_4

  func.return %m1, %m2, %m3, %m4, %m5 : i1, i1, i1, i1, i1
}
// CHECK-LABEL: define { i1, i1, i1, i1, i1 } @qaoa_ansatz
// CHECK-NEXT:    %5 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %6 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %6)
// CHECK-NEXT:    %7 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %7)
// CHECK-NEXT:    %8 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %8)
// CHECK-NEXT:    %9 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %9)
// CHECK-NEXT:    %10 = call { ptr, ptr, ptr, ptr, ptr } @qaoa_problem(double %0, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9)
// CHECK-NEXT:    %11 = extractvalue { ptr, ptr, ptr, ptr, ptr } %10, 0
// CHECK-NEXT:    %12 = extractvalue { ptr, ptr, ptr, ptr, ptr } %10, 1
// CHECK-NEXT:    %13 = extractvalue { ptr, ptr, ptr, ptr, ptr } %10, 2
// CHECK-NEXT:    %14 = extractvalue { ptr, ptr, ptr, ptr, ptr } %10, 3
// CHECK-NEXT:    %15 = extractvalue { ptr, ptr, ptr, ptr, ptr } %10, 4
// CHECK-NEXT:    %16 = call { ptr, ptr, ptr, ptr, ptr } @qaoa_mixer(double %2, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15)
// CHECK-NEXT:    %17 = extractvalue { ptr, ptr, ptr, ptr, ptr } %16, 0
// CHECK-NEXT:    %18 = extractvalue { ptr, ptr, ptr, ptr, ptr } %16, 1
// CHECK-NEXT:    %19 = extractvalue { ptr, ptr, ptr, ptr, ptr } %16, 2
// CHECK-NEXT:    %20 = extractvalue { ptr, ptr, ptr, ptr, ptr } %16, 3
// CHECK-NEXT:    %21 = extractvalue { ptr, ptr, ptr, ptr, ptr } %16, 4
// CHECK-NEXT:    %22 = call { ptr, ptr, ptr, ptr, ptr } @qaoa_problem(double %1, ptr %17, ptr %18, ptr %19, ptr %20, ptr %21)
// CHECK-NEXT:    %23 = extractvalue { ptr, ptr, ptr, ptr, ptr } %22, 0
// CHECK-NEXT:    %24 = extractvalue { ptr, ptr, ptr, ptr, ptr } %22, 1
// CHECK-NEXT:    %25 = extractvalue { ptr, ptr, ptr, ptr, ptr } %22, 2
// CHECK-NEXT:    %26 = extractvalue { ptr, ptr, ptr, ptr, ptr } %22, 3
// CHECK-NEXT:    %27 = extractvalue { ptr, ptr, ptr, ptr, ptr } %22, 4
// CHECK-NEXT:    %28 = call { ptr, ptr, ptr, ptr, ptr } @qaoa_mixer(double %3, ptr %23, ptr %24, ptr %25, ptr %26, ptr %27)
// CHECK-NEXT:    %29 = extractvalue { ptr, ptr, ptr, ptr, ptr } %28, 0
// CHECK-NEXT:    %30 = extractvalue { ptr, ptr, ptr, ptr, ptr } %28, 1
// CHECK-NEXT:    %31 = extractvalue { ptr, ptr, ptr, ptr, ptr } %28, 2
// CHECK-NEXT:    %32 = extractvalue { ptr, ptr, ptr, ptr, ptr } %28, 3
// CHECK-NEXT:    %33 = extractvalue { ptr, ptr, ptr, ptr, ptr } %28, 4
// CHECK-NEXT:    %34 = call ptr @__quantum__qis__m__body(ptr %29)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %29)
// CHECK-NEXT:    %35 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %36 = call i1 @__quantum__rt__result_equal(ptr %34, ptr %35)
// CHECK-NEXT:    %37 = call ptr @__quantum__qis__m__body(ptr %30)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %30)
// CHECK-NEXT:    %38 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %39 = call i1 @__quantum__rt__result_equal(ptr %37, ptr %38)
// CHECK-NEXT:    %40 = call ptr @__quantum__qis__m__body(ptr %31)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %31)
// CHECK-NEXT:    %41 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %42 = call i1 @__quantum__rt__result_equal(ptr %40, ptr %41)
// CHECK-NEXT:    %43 = call ptr @__quantum__qis__m__body(ptr %32)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %32)
// CHECK-NEXT:    %44 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %45 = call i1 @__quantum__rt__result_equal(ptr %43, ptr %44)
// CHECK-NEXT:    %46 = call ptr @__quantum__qis__m__body(ptr %33)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %33)
// CHECK-NEXT:    %47 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %48 = call i1 @__quantum__rt__result_equal(ptr %46, ptr %47)
// CHECK-NEXT:    %49 = insertvalue { i1, i1, i1, i1, i1 } undef, i1 %36, 0
// CHECK-NEXT:    %50 = insertvalue { i1, i1, i1, i1, i1 } %49, i1 %39, 1
// CHECK-NEXT:    %51 = insertvalue { i1, i1, i1, i1, i1 } %50, i1 %42, 2
// CHECK-NEXT:    %52 = insertvalue { i1, i1, i1, i1, i1 } %51, i1 %45, 3
// CHECK-NEXT:    %53 = insertvalue { i1, i1, i1, i1, i1 } %52, i1 %48, 4
// CHECK-NEXT:    ret { i1, i1, i1, i1, i1 } %53
// CHECK-NEXT:  }
