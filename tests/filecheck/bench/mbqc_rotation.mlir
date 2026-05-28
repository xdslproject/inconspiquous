// RUN: TO_QIR

// MBQC arbitrary single qubit rotation.

func.func @rotation(%phi: !angle.type, %theta: !angle.type, %lambda: !angle.type, %q1: !qu.bit) -> !qu.bit {
  %q2 = qu.alloc<#qu.plus>
  %q3 = qu.alloc<#qu.plus>
  %q4 = qu.alloc<#qu.plus>
  %q5 = qu.alloc<#qu.plus>
  %q1_1, %q2_1 = qssa.gate<#gate.cz> %q1, %q2
  %q2_2, %q3_1 = qssa.gate<#gate.cz> %q2_1, %q3
  %q3_2, %q4_1 = qssa.gate<#gate.cz> %q3_1, %q4
  %q4_2, %q5_1 = qssa.gate<#gate.cz> %q4_1, %q5
  %0 = qssa.measure<#measurement.x_basis> %q1_1
  %a2 = angle.cond_negate %0, %lambda
  %m2 = measurement.dyn_xy<%a2>
  %1 = qssa.dyn_measure<%m2> %q2_2
  %a3 = angle.cond_negate %1, %theta
  %m3 = measurement.dyn_xy<%a3>
  %2 = qssa.dyn_measure<%m3> %q3_2
  %z = arith.xori %0, %2 : i1
  %a4 = angle.cond_negate %z, %phi
  %m4 = measurement.dyn_xy<%a4>
  %3 = qssa.dyn_measure<%m4> %q4_2
  %x = arith.xori %1, %3 : i1
  %g = gate.xz %x, %z
  %q5_2 = qssa.dyn_gate<%g> %q5_1
  func.return %q5_2 : !qu.bit
}
// CHECK-LABEL: rotation
// CHECK-NEXT:    %5 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %6 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %6)
// CHECK-NEXT:    %7 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %7)
// CHECK-NEXT:    %8 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %8)
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %3, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %5, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %6, ptr %7)
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %7, ptr %8)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %3)
// CHECK-NEXT:    %9 = call ptr @__quantum__qis__m__body(ptr %3)
// CHECK-NEXT:    %10 = call i1 @__quantum__rt__read_result__body(ptr %9)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %3)
// CHECK-NEXT:    %11 = fneg double %2
// CHECK-NEXT:    %12 = select i1 %10, double %11, double %2
// CHECK-NEXT:    %13 = fneg double %12
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %13, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %14 = call ptr @__quantum__qis__m__body(ptr %5)
// CHECK-NEXT:    %15 = call i1 @__quantum__rt__read_result__body(ptr %14)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %5)
// CHECK-NEXT:    %16 = fneg double %1
// CHECK-NEXT:    %17 = select i1 %15, double %16, double %1
// CHECK-NEXT:    %18 = fneg double %17
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %18, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %6)
// CHECK-NEXT:    %19 = call ptr @__quantum__qis__m__body(ptr %6)
// CHECK-NEXT:    %20 = call i1 @__quantum__rt__read_result__body(ptr %19)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %6)
// CHECK-NEXT:    %21 = xor i1 %10, %20
// CHECK-NEXT:    %22 = fneg double %0
// CHECK-NEXT:    %23 = select i1 %21, double %22, double %0
// CHECK-NEXT:    %24 = fneg double %23
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %24, ptr %7)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %7)
// CHECK-NEXT:    %25 = call ptr @__quantum__qis__m__body(ptr %7)
// CHECK-NEXT:    %26 = call i1 @__quantum__rt__read_result__body(ptr %25)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %7)
// CHECK-NEXT:    %27 = xor i1 %15, %26
// CHECK-NEXT:    br i1 %27, label %28, label %31
// CHECK-EMPTY:
// CHECK-NEXT:  28:                                               ; preds = %4
// CHECK-NEXT:    br i1 %21, label %29, label %30
// CHECK-EMPTY:
// CHECK-NEXT:  29:                                               ; preds = %28
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %8)
// CHECK-NEXT:    br label %33
// CHECK-EMPTY:
// CHECK-NEXT:  30:                                               ; preds = %28
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %8)
// CHECK-NEXT:    br label %33
// CHECK-EMPTY:
// CHECK-NEXT:  31:                                               ; preds = %4
// CHECK-NEXT:    br i1 %21, label %32, label %33
// CHECK-EMPTY:
// CHECK-NEXT:  32:                                               ; preds = %31
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %8)
// CHECK-NEXT:    br label %33
// CHECK-EMPTY:
// CHECK-NEXT:  33:                                               ; preds = %29, %30, %32, %31
// CHECK-NEXT:    ret ptr %8
// CHECK-NEXT:  }
