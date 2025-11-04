// RUN: quopt %s -p convert-qssa-to-qref,lower-xzs-to-select,cse,canonicalize,lower-dyn-gate-to-scf,canonicalize,convert-qref-to-qir,convert-qir-to-llvm | mlir-opt -p 'builtin.module(convert-scf-to-cf,canonicalize,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm)' | mlir-translate --mlir-to-llvmir | filecheck %s

// From https://arxiv.org/pdf/2302.03029

//            a3            a6
//           /  \          /  \
// q01 -- q02 -- q03 -- q04 -- q05 -- q06
//  | \  / |      | \  / |      | \  / |
//  |  a1  |      |  a4  |      |  a7  |
//  | /  \ |      | /  \ |      | /  \ |
// q07 -- q08 -- q09 -- q10 -- q11 -- q12
//           \  /          \  /
//            a2            a5

func.func @logical_plus() {
  // Data qubits

  %q01 = qu.alloc<#qu.plus>
  %q02 = qu.alloc<#qu.plus>
  %q03 = qu.alloc<#qu.plus>
  %q04 = qu.alloc<#qu.plus>
  %q05 = qu.alloc<#qu.plus>
  %q06 = qu.alloc<#qu.plus>
  %q07 = qu.alloc<#qu.plus>
  %q08 = qu.alloc<#qu.plus>
  %q09 = qu.alloc<#qu.plus>
  %q10 = qu.alloc<#qu.plus>
  %q11 = qu.alloc<#qu.plus>
  %q12 = qu.alloc<#qu.plus>

  // Ancilla qubits

  %a1 = qu.alloc
  %a2 = qu.alloc
  %a3 = qu.alloc
  %a4 = qu.alloc
  %a5 = qu.alloc
  %a6 = qu.alloc
  %a7 = qu.alloc

  // Measure Z stabilisers (using order in https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.033074)

  %q02_1, %a1_1 = qssa.gate<#gate.cx> %q02, %a1
  %q09_1, %a2_1 = qssa.gate<#gate.cx> %q09, %a2
  %q04_1, %a4_1 = qssa.gate<#gate.cx> %q04, %a4
  %q11_1, %a5_1 = qssa.gate<#gate.cx> %q11, %a5
  %q06_1, %a7_1 = qssa.gate<#gate.cx> %q06, %a7
  %q08_1, %a1_2 = qssa.gate<#gate.cx> %q08, %a1_1
  %q03_1, %a3_1 = qssa.gate<#gate.cx> %q03, %a3
  %q10_1, %a4_2 = qssa.gate<#gate.cx> %q10, %a4_1
  %q05_1, %a6_1 = qssa.gate<#gate.cx> %q05, %a6
  %q12_1, %a7_2 = qssa.gate<#gate.cx> %q12, %a7_1
  %q01_1, %a1_3 = qssa.gate<#gate.cx> %q01, %a1_2
  %q08_2, %a2_2 = qssa.gate<#gate.cx> %q08_1, %a2_1
  %q03_2, %a4_3 = qssa.gate<#gate.cx> %q03_1, %a4_2
  %q10_2, %a5_2 = qssa.gate<#gate.cx> %q10_1, %a5_1
  %q05_2, %a7_3 = qssa.gate<#gate.cx> %q05_1, %a7_2
  %q07_1, %a1_4 = qssa.gate<#gate.cx> %q07, %a1_3
  %q02_2, %a3_2 = qssa.gate<#gate.cx> %q02_1, %a3_1
  %q09_2, %a4_4 = qssa.gate<#gate.cx> %q09_1, %a4_3
  %q04_2, %a6_2 = qssa.gate<#gate.cx> %q04_1, %a6_1
  %q11_2, %a7_4 = qssa.gate<#gate.cx> %q11_1, %a7_3

  %s1 = qssa.measure %a1_4
  %s2 = qssa.measure %a2_2
  %s3 = qssa.measure %a3_2
  %s4 = qssa.measure %a4_4
  %s5 = qssa.measure %a5_2
  %s6 = qssa.measure %a6_2
  %s7 = qssa.measure %a7_4

  // Correct along chain q02 -> q09 -> q03 -> q04 -> q11 -> q05 -> q06
  %c0 = arith.constant false

  %g1 = gate.xz %s1, %c0
  %q02_3 = qssa.dyn_gate<%g1> %q02_2

  %g2 = gate.xz %s2, %c0
  %q09_3 = qssa.dyn_gate<%g2> %q09_2

  %x3 = arith.xori %s1, %s3 : i1
  %g3 = gate.xz %x3, %c0
  %q03_3 = qssa.dyn_gate<%g3> %q03_2

  %t4 = arith.xori %s2, %x3 : i1
  %x4 = arith.xori %t4, %s4 : i1
  %g4 = gate.xz %x4, %c0
  %q04_3 = qssa.dyn_gate<%g4> %q04_2

  %g5 = gate.xz %s5, %c0
  %q11_3 = qssa.dyn_gate<%g5> %q11_2

  %x6 = arith.xori %x4, %s6 : i1
  %g6 = gate.xz %x6, %c0
  %q05_3 = qssa.dyn_gate<%g6> %q05_2

  %t7 = arith.xori %s5, %x6 : i1
  %x7 = arith.xori %t7, %s7 : i1
  %g7 = gate.xz %x7, %c0
  %q06_2 = qssa.dyn_gate<%g7> %q06_1

  func.return
}
// CHECK-LABEL: @logical_plus
// CHECK-NEXT:    %1 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %1)
// CHECK-NEXT:    %2 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %2)
// CHECK-NEXT:    %3 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %3)
// CHECK-NEXT:    %4 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
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
// CHECK-NEXT:    %10 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %10)
// CHECK-NEXT:    %11 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %11)
// CHECK-NEXT:    %12 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %12)
// CHECK-NEXT:    %13 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %14 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %15 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %16 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %17 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %18 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %19 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %2, ptr %13)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %9, ptr %14)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %4, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %11, ptr %17)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %6, ptr %19)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %8, ptr %13)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %3, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %10, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %5, ptr %18)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %12, ptr %19)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %1, ptr %13)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %8, ptr %14)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %3, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %10, ptr %17)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %5, ptr %19)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %7, ptr %13)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %2, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %9, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %4, ptr %18)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %11, ptr %19)
// CHECK-NEXT:    %20 = call ptr @__quantum__qis__m__body(ptr %13)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %13)
// CHECK-NEXT:    %21 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %22 = call i1 @__quantum__rt__result_equal(ptr %20, ptr %21)
// CHECK-NEXT:    %23 = call ptr @__quantum__qis__m__body(ptr %14)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %14)
// CHECK-NEXT:    %24 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %25 = call i1 @__quantum__rt__result_equal(ptr %23, ptr %24)
// CHECK-NEXT:    %26 = call ptr @__quantum__qis__m__body(ptr %15)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %15)
// CHECK-NEXT:    %27 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %28 = call i1 @__quantum__rt__result_equal(ptr %26, ptr %27)
// CHECK-NEXT:    %29 = call ptr @__quantum__qis__m__body(ptr %16)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %16)
// CHECK-NEXT:    %30 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %31 = call i1 @__quantum__rt__result_equal(ptr %29, ptr %30)
// CHECK-NEXT:    %32 = call ptr @__quantum__qis__m__body(ptr %17)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %17)
// CHECK-NEXT:    %33 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %34 = call i1 @__quantum__rt__result_equal(ptr %32, ptr %33)
// CHECK-NEXT:    %35 = call ptr @__quantum__qis__m__body(ptr %18)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %18)
// CHECK-NEXT:    %36 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %37 = call i1 @__quantum__rt__result_equal(ptr %35, ptr %36)
// CHECK-NEXT:    %38 = call ptr @__quantum__qis__m__body(ptr %19)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %19)
// CHECK-NEXT:    %39 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %40 = call i1 @__quantum__rt__result_equal(ptr %38, ptr %39)
// CHECK-NEXT:    br i1 %22, label %41, label %42
// CHECK-EMPTY:
// CHECK-NEXT:  41:                                               ; preds = %0
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %2)
// CHECK-NEXT:    br label %42
// CHECK-EMPTY:
// CHECK-NEXT:  42:                                               ; preds = %41, %0
// CHECK-NEXT:    br i1 %25, label %43, label %44
// CHECK-EMPTY:
// CHECK-NEXT:  43:                                               ; preds = %42
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %9)
// CHECK-NEXT:    br label %44
// CHECK-EMPTY:
// CHECK-NEXT:  44:                                               ; preds = %43, %42
// CHECK-NEXT:    %45 = xor i1 %22, %28
// CHECK-NEXT:    br i1 %45, label %46, label %47
// CHECK-EMPTY:
// CHECK-NEXT:  46:                                               ; preds = %44
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %3)
// CHECK-NEXT:    br label %47
// CHECK-EMPTY:
// CHECK-NEXT:  47:                                               ; preds = %46, %44
// CHECK-NEXT:    %48 = xor i1 %25, %45
// CHECK-NEXT:    %49 = xor i1 %48, %31
// CHECK-NEXT:    br i1 %49, label %50, label %51
// CHECK-EMPTY:
// CHECK-NEXT:  50:                                               ; preds = %47
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %4)
// CHECK-NEXT:    br label %51
// CHECK-EMPTY:
// CHECK-NEXT:  51:                                               ; preds = %50, %47
// CHECK-NEXT:    br i1 %34, label %52, label %53
// CHECK-EMPTY:
// CHECK-NEXT:  52:                                               ; preds = %51
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %11)
// CHECK-NEXT:    br label %53
// CHECK-EMPTY:
// CHECK-NEXT:  53:                                               ; preds = %52, %51
// CHECK-NEXT:    %54 = xor i1 %49, %37
// CHECK-NEXT:    br i1 %54, label %55, label %56
// CHECK-EMPTY:
// CHECK-NEXT:  55:                                               ; preds = %53
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %5)
// CHECK-NEXT:    br label %56
// CHECK-EMPTY:
// CHECK-NEXT:  56:                                               ; preds = %55, %53
// CHECK-NEXT:    %57 = xor i1 %34, %54
// CHECK-NEXT:    %58 = xor i1 %57, %40
// CHECK-NEXT:    br i1 %58, label %59, label %60
// CHECK-EMPTY:
// CHECK-NEXT:  59:                                               ; preds = %56
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %6)
// CHECK-NEXT:    br label %60
// CHECK-EMPTY:
// CHECK-NEXT:  60:                                               ; preds = %59, %56
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
