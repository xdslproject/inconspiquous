// RUN: quopt %s -p convert-qssa-to-qref,lower-xzs-to-select,cse,canonicalize,lower-dyn-gate-to-scf,canonicalize,convert-qref-to-qir,convert-qir-to-llvm | mlir-opt -p 'builtin.module(convert-scf-to-cf,canonicalize,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm)' | mlir-translate --mlir-to-llvmir | filecheck %s

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
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %3)
// CHECK-NEXT:    %10 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %11 = call i1 @__quantum__rt__result_equal(ptr %9, ptr %10)
// CHECK-NEXT:    %12 = fneg double %2
// CHECK-NEXT:    %13 = select i1 %11, double %12, double %2
// CHECK-NEXT:    %14 = fneg double %13
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %14, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %15 = call ptr @__quantum__qis__m__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %5)
// CHECK-NEXT:    %16 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %17 = call i1 @__quantum__rt__result_equal(ptr %15, ptr %16)
// CHECK-NEXT:    %18 = fneg double %1
// CHECK-NEXT:    %19 = select i1 %17, double %18, double %1
// CHECK-NEXT:    %20 = fneg double %19
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %20, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %6)
// CHECK-NEXT:    %21 = call ptr @__quantum__qis__m__body(ptr %6)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %6)
// CHECK-NEXT:    %22 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %23 = call i1 @__quantum__rt__result_equal(ptr %21, ptr %22)
// CHECK-NEXT:    %24 = xor i1 %11, %23
// CHECK-NEXT:    %25 = fneg double %0
// CHECK-NEXT:    %26 = select i1 %24, double %25, double %0
// CHECK-NEXT:    %27 = fneg double %26
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %27, ptr %7)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %7)
// CHECK-NEXT:    %28 = call ptr @__quantum__qis__m__body(ptr %7)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %7)
// CHECK-NEXT:    %29 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %30 = call i1 @__quantum__rt__result_equal(ptr %28, ptr %29)
// CHECK-NEXT:    %31 = xor i1 %17, %30
// CHECK-NEXT:    br i1 %31, label %32, label %35
// CHECK-EMPTY:
// CHECK-NEXT:  32:                                               ; preds = %4
// CHECK-NEXT:    br i1 %24, label %33, label %34
// CHECK-EMPTY:
// CHECK-NEXT:  33:                                               ; preds = %32
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %8)
// CHECK-NEXT:    br label %37
// CHECK-EMPTY:
// CHECK-NEXT:  34:                                               ; preds = %32
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %8)
// CHECK-NEXT:    br label %37
// CHECK-EMPTY:
// CHECK-NEXT:  35:                                               ; preds = %4
// CHECK-NEXT:    br i1 %24, label %36, label %37
// CHECK-EMPTY:
// CHECK-NEXT:  36:                                               ; preds = %35
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %8)
// CHECK-NEXT:    br label %37
// CHECK-EMPTY:
// CHECK-NEXT:  37:                                               ; preds = %33, %34, %36, %35
// CHECK-NEXT:    ret ptr %8
// CHECK-NEXT:  }
