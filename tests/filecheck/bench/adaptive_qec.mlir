// RUN: quopt %s -p convert-qssa-to-qref,lower-xzs-to-select,cse,canonicalize,lower-dyn-gate-to-scf,canonicalize,convert-qref-to-qir,convert-qir-to-llvm | mlir-opt -p 'builtin.module(convert-scf-to-cf,canonicalize,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm)' | mlir-translate --mlir-to-llvmir | filecheck %s

// Adaptive QEC cycle from https://arxiv.org/pdf/2107.07505
// We perform the corrections instead of updating a pauli frame.

func.func @adaptive_qec_cycle(
  %q1 : !qu.bit,
  %q2 : !qu.bit,
  %q3 : !qu.bit,
  %q4 : !qu.bit,
  %q5 : !qu.bit,
  %q6 : !qu.bit,
  %q7 : !qu.bit,
  %prev_s1 : i1,
  %prev_s2 : i1,
  %prev_s3 : i1,
  %prev_s4 : i1,
  %prev_s5 : i1,
  %prev_s6 : i1
) -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1, i1, i1, i1) {
  %a1 = qu.alloc
  %a2 = qu.alloc
  %a3 = qu.alloc

  %a1_1 = qssa.gate<#gate.h> %a1
  %a1_2, %q4_1 = qssa.gate<#gate.cx> %a1_1, %q4
  %q3_1, %a3_1 = qssa.gate<#gate.cx> %q3, %a3
  %q6_1, %a2_1 = qssa.gate<#gate.cx> %q6, %a2
  %a1_3, %a2_2 = qssa.gate<#gate.cx> %a1_2, %a2_1
  %a1_4, %q1_1 = qssa.gate<#gate.cx> %a1_3, %q1
  %q4_2, %a3_2 = qssa.gate<#gate.cx> %q4_1, %a3_1
  %q5_1, %a2_3 = qssa.gate<#gate.cx> %q5, %a2_2
  %a1_5, %q2_1 = qssa.gate<#gate.cx> %a1_4, %q2
  %q7_1, %a3_3 = qssa.gate<#gate.cx> %q7, %a3_2
  %q3_2, %a2_4 = qssa.gate<#gate.cx> %q3_1, %a2_3
  %a1_6, %a3_4 = qssa.gate<#gate.cx> %a1_5, %a3_3
  %a1_7, %q3_3 = qssa.gate<#gate.cx> %a1_6, %q3_2
  %q6_2, %a3_5 = qssa.gate<#gate.cx> %q6_1, %a3_4
  %q2_2, %a2_5 = qssa.gate<#gate.cx> %q2_1, %a2_4
  %a1_8 = qssa.gate<#gate.h> %a1_7

  %f1 = qssa.measure %a1_8
  %fd1 = arith.xori %f1, %prev_s1 : i1

  %f5 = qssa.measure %a2_5
  %fd5 = arith.xori %f5, %prev_s5 : i1

  %f6 = qssa.measure %a3_5
  %fd6 = arith.xori %f6, %prev_s6 : i1

  %0 = arith.ori %fd1, %fd5 : i1
  %1 = arith.ori %0, %fd6 : i1

  %q1_2, %q2_3, %q3_4, %q4_3, %q5_2, %q6_3, %q7_2, %fd2, %fd3, %fd4 = scf.if %1
    -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1) {
    // Flagged circuit for S2, S3, S4
    %a4 = qu.alloc
    %a5 = qu.alloc
    %a6 = qu.alloc

    %a5_1 = qssa.gate<#gate.h> %a5
    %a6_1 = qssa.gate<#gate.h> %a6
    %q4_4, %a4_1 = qssa.gate<#gate.cx> %q4_2, %a4
    %a6_2, %q3_5 = qssa.gate<#gate.cx> %a6_1, %q3_3
    %a5_2, %q6_4 = qssa.gate<#gate.cx> %a5_1, %q6_2
    %a5_3, %a4_2 = qssa.gate<#gate.cx> %a5_2, %a4_1
    %q1_3, %a4_3 = qssa.gate<#gate.cx> %q1_1, %a4_2
    %a6_3, %q4_5 = qssa.gate<#gate.cx> %a6_2, %q4_4
    %a5_4, %q5_3 = qssa.gate<#gate.cx> %a5_3, %q5_1
    %q2_4, %a4_4 = qssa.gate<#gate.cx> %q2_2, %a4_3
    %a6_4, %q7_3 = qssa.gate<#gate.cx> %a6_3, %q7_1
    %a5_5, %q3_6 = qssa.gate<#gate.cx> %a5_4, %q3_5
    %a6_5, %a4_5 = qssa.gate<#gate.cx> %a6_4, %a4_4
    %q3_7, %a4_6 = qssa.gate<#gate.cx> %q3_6, %a4_5
    %a6_6, %q6_5 = qssa.gate<#gate.cx> %a6_5, %q6_4
    %a5_6, %q2_5 = qssa.gate<#gate.cx> %a5_5, %q2_4
    %a5_7 = qssa.gate<#gate.h> %a5_6
    %a6_7 = qssa.gate<#gate.h> %a6_6

    %f2 = qssa.measure %a4_6
    %fd2_1 = arith.xori %f2, %prev_s2 : i1

    %f3 = qssa.measure %a5_7
    %fd3_1 = arith.xori %f3, %prev_s3 : i1

    %f4 = qssa.measure %a6_7
    %fd4_1 = arith.xori %f4, %prev_s4 : i1

    scf.yield %q1_3, %q2_5, %q3_7, %q4_5, %q5_3, %q6_5, %q7_3, %fd2_1, %fd3_1, %fd4_1
      : !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1
  } else {
    %cFalse = arith.constant false
    scf.yield %q1_1, %q2_2, %q3_3, %q4_2, %q5_1, %q6_2, %q7_1, %cFalse, %cFalse, %cFalse
      : !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1
  }

  %2 = arith.ori %fd2, %fd3 : i1
  %3 = arith.ori %2, %fd4 : i1

  %any_flag = arith.ori %1, %3 : i1

  %q1_4, %q2_6, %q3_8, %q4_6, %q5_4, %q6_6, %q7_4, %s1_final, %s2_final, %s3_final, %s4_final, %s5_final, %s6_final = scf.if %any_flag
    -> (!qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1, i1, i1, i1) {
    // S1 syndrome
    %as1 = qu.alloc
    %q1_5, %as1_1 = qssa.gate<#gate.cx> %q1_2, %as1
    %q2_7, %as1_2 = qssa.gate<#gate.cx> %q2_3, %as1_1
    %q3_9, %as1_3 = qssa.gate<#gate.cx> %q3_4, %as1_2
    %q4_7, %as1_4 = qssa.gate<#gate.cx> %q4_3, %as1_3
    %s1 = qssa.measure %as1_4

    %s1_diff = arith.xori %prev_s1, %s1 : i1

    // S2 syndrome
    %as2 = qu.alloc
    %q2_8, %as2_1 = qssa.gate<#gate.cx> %q2_7, %as2
    %q3_10, %as2_2 = qssa.gate<#gate.cx> %q3_9, %as2_1
    %q5_5, %as2_3 = qssa.gate<#gate.cx> %q5_2, %as2_2
    %q6_7, %as2_4 = qssa.gate<#gate.cx> %q6_3, %as2_3
    %s2 = qssa.measure %as2_4

    %s2_diff = arith.xori %prev_s2, %s2 : i1

    // S3 syndrome
    %as3 = qu.alloc
    %q3_11, %as3_1 = qssa.gate<#gate.cx> %q3_10, %as3
    %q4_8, %as3_2 = qssa.gate<#gate.cx> %q4_7, %as3_1
    %q6_8, %as3_3 = qssa.gate<#gate.cx> %q6_7, %as3_2
    %q7_5, %as3_4 = qssa.gate<#gate.cx> %q7_2, %as3_3
    %s3 = qssa.measure %as3_4

    %s3_diff = arith.xori %prev_s3, %s3 : i1

    // S4 syndrome
    %as4 = qu.alloc
    %as4_1 = qssa.gate<#gate.h> %as4
    %as4_2, %q1_6 = qssa.gate<#gate.cx> %as4_1, %q1_5
    %as4_3, %q2_9 = qssa.gate<#gate.cx> %as4_2, %q2_8
    %as4_4, %q3_12 = qssa.gate<#gate.cx> %as4_3, %q3_11
    %as4_5, %q4_9 = qssa.gate<#gate.cx> %as4_4, %q4_8
    %as4_6 = qssa.gate<#gate.h> %as4_5
    %s4 = qssa.measure %as4_6

    %s4_diff = arith.xori %prev_s4, %s4 : i1

    // S5 syndrome
    %as5 = qu.alloc
    %as5_1 = qssa.gate<#gate.h> %as5
    %as5_2, %q2_10 = qssa.gate<#gate.cx> %as5_1, %q2_9
    %as5_3, %q3_13 = qssa.gate<#gate.cx> %as5_2, %q3_12
    %as5_4, %q5_6 = qssa.gate<#gate.cx> %as5_3, %q5_5
    %as5_5, %q6_9 = qssa.gate<#gate.cx> %as5_4, %q6_8
    %as5_6 = qssa.gate<#gate.h> %as5_5
    %s5 = qssa.measure %as5_6

    %s5_diff = arith.xori %prev_s5, %s5 : i1

    // S6 syndrome
    %as6 = qu.alloc
    %as6_1 = qssa.gate<#gate.h> %as6
    %as6_2, %q3_14 = qssa.gate<#gate.cx> %as6_1, %q3_13
    %as6_3, %q4_10 = qssa.gate<#gate.cx> %as6_2, %q4_9
    %as6_4, %q6_10 = qssa.gate<#gate.cx> %as6_3, %q6_9
    %as6_5, %q7_6 = qssa.gate<#gate.cx> %as6_4, %q7_5
    %as6_6 = qssa.gate<#gate.h> %as6_5
    %s6 = qssa.measure %as6_6

    %s6_diff = arith.xori %prev_s6, %s6 : i1

    // Decode X logical
    %x = func.call @decode(%fd1, %fd2, %fd3, %s1_diff, %s2_diff, %s3_diff) : (i1, i1, i1, i1, i1, i1) -> i1

    // Decode Z logical
    %z = func.call @decode(%fd4, %fd5, %fd6, %s4_diff, %s5_diff, %s6_diff) : (i1, i1, i1, i1, i1, i1) -> i1

    // Perform correction
    %g = gate.xz %x, %z
    %q5_7 = qssa.dyn_gate<%g> %q5_6
    %q6_11 = qssa.dyn_gate<%g> %q6_10
    %q7_7 = qssa.dyn_gate<%g> %q7_6

    scf.yield %q1_6, %q2_10, %q3_14, %q4_10, %q5_7, %q6_11, %q7_7, %s1, %s2, %s3, %s4, %s5, %s6
      : !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1, i1, i1, i1
  } else {
    scf.yield %q1_2, %q2_3, %q3_4, %q4_3, %q5_2, %q6_3, %q7_2, %prev_s1, %prev_s2, %prev_s3, %prev_s4, %prev_s5, %prev_s6
      : !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1, i1, i1, i1
  }

  func.return %q1_4, %q2_6, %q3_8, %q4_6, %q5_4, %q6_6, %q7_4, %s1_final, %s2_final, %s3_final, %s4_final, %s5_final, %s6_final
    : !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, !qu.bit, i1, i1, i1, i1, i1, i1
}
// CHECK-LABEL: @adaptive_qec_cycle
// CHECK-NEXT:    %14 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %15 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %16 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %14)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %14, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %2, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %5, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %14, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %14, ptr %0)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %3, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %4, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %14, ptr %1)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %6, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %2, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %14, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %14, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %5, ptr %16)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %1, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %14)
// CHECK-NEXT:    %17 = call ptr @__quantum__qis__m__body(ptr %14)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %14)
// CHECK-NEXT:    %18 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %19 = call i1 @__quantum__rt__result_equal(ptr %17, ptr %18)
// CHECK-NEXT:    %20 = xor i1 %19, %7
// CHECK-NEXT:    %21 = call ptr @__quantum__qis__m__body(ptr %15)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %15)
// CHECK-NEXT:    %22 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %23 = call i1 @__quantum__rt__result_equal(ptr %21, ptr %22)
// CHECK-NEXT:    %24 = xor i1 %23, %11
// CHECK-NEXT:    %25 = call ptr @__quantum__qis__m__body(ptr %16)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %16)
// CHECK-NEXT:    %26 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %27 = call i1 @__quantum__rt__result_equal(ptr %25, ptr %26)
// CHECK-NEXT:    %28 = xor i1 %27, %12
// CHECK-NEXT:    %29 = or i1 %20, %24
// CHECK-NEXT:    %30 = or i1 %29, %28
// CHECK-NEXT:    br i1 %30, label %31, label %47
// CHECK-EMPTY:
// CHECK-NEXT:  31:                                               ; preds = %13
// CHECK-NEXT:    %32 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %33 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %34 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %33)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %34)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %3, ptr %32)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %34, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %33, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %33, ptr %32)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %0, ptr %32)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %34, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %33, ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %1, ptr %32)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %34, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %33, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %34, ptr %32)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %2, ptr %32)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %34, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %33, ptr %1)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %33)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %34)
// CHECK-NEXT:    %35 = call ptr @__quantum__qis__m__body(ptr %32)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %32)
// CHECK-NEXT:    %36 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %37 = call i1 @__quantum__rt__result_equal(ptr %35, ptr %36)
// CHECK-NEXT:    %38 = xor i1 %37, %8
// CHECK-NEXT:    %39 = call ptr @__quantum__qis__m__body(ptr %33)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %33)
// CHECK-NEXT:    %40 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %41 = call i1 @__quantum__rt__result_equal(ptr %39, ptr %40)
// CHECK-NEXT:    %42 = xor i1 %41, %9
// CHECK-NEXT:    %43 = call ptr @__quantum__qis__m__body(ptr %34)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %34)
// CHECK-NEXT:    %44 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %45 = call i1 @__quantum__rt__result_equal(ptr %43, ptr %44)
// CHECK-NEXT:    %46 = xor i1 %45, %10
// CHECK-NEXT:    br label %47
// CHECK-EMPTY:
// CHECK-NEXT:  47:                                               ; preds = %31, %13
// CHECK-NEXT:    %48 = phi ptr [ %0, %31 ], [ %0, %13 ]
// CHECK-NEXT:    %49 = phi ptr [ %1, %31 ], [ %1, %13 ]
// CHECK-NEXT:    %50 = phi ptr [ %2, %31 ], [ %2, %13 ]
// CHECK-NEXT:    %51 = phi ptr [ %3, %31 ], [ %3, %13 ]
// CHECK-NEXT:    %52 = phi ptr [ %4, %31 ], [ %4, %13 ]
// CHECK-NEXT:    %53 = phi ptr [ %5, %31 ], [ %5, %13 ]
// CHECK-NEXT:    %54 = phi ptr [ %6, %31 ], [ %6, %13 ]
// CHECK-NEXT:    %55 = phi i1 [ %38, %31 ], [ false, %13 ]
// CHECK-NEXT:    %56 = phi i1 [ %42, %31 ], [ false, %13 ]
// CHECK-NEXT:    %57 = phi i1 [ %46, %31 ], [ false, %13 ]
// CHECK-NEXT:    %58 = or i1 %55, %56
// CHECK-NEXT:    %59 = or i1 %58, %57
// CHECK-NEXT:    %60 = or i1 %30, %59
// CHECK-NEXT:    br i1 %60, label %61, label %111
// CHECK-EMPTY:
// CHECK-NEXT:  61:                                               ; preds = %47
// CHECK-NEXT:    %62 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %48, ptr %62)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %49, ptr %62)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %50, ptr %62)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %51, ptr %62)
// CHECK-NEXT:    %63 = call ptr @__quantum__qis__m__body(ptr %62)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %62)
// CHECK-NEXT:    %64 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %65 = call i1 @__quantum__rt__result_equal(ptr %63, ptr %64)
// CHECK-NEXT:    %66 = xor i1 %7, %65
// CHECK-NEXT:    %67 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %49, ptr %67)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %50, ptr %67)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %52, ptr %67)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %53, ptr %67)
// CHECK-NEXT:    %68 = call ptr @__quantum__qis__m__body(ptr %67)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %67)
// CHECK-NEXT:    %69 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %70 = call i1 @__quantum__rt__result_equal(ptr %68, ptr %69)
// CHECK-NEXT:    %71 = xor i1 %8, %70
// CHECK-NEXT:    %72 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %50, ptr %72)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %51, ptr %72)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %53, ptr %72)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %54, ptr %72)
// CHECK-NEXT:    %73 = call ptr @__quantum__qis__m__body(ptr %72)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %72)
// CHECK-NEXT:    %74 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %75 = call i1 @__quantum__rt__result_equal(ptr %73, ptr %74)
// CHECK-NEXT:    %76 = xor i1 %9, %75
// CHECK-NEXT:    %77 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %77)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %77, ptr %48)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %77, ptr %49)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %77, ptr %50)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %77, ptr %51)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %77)
// CHECK-NEXT:    %78 = call ptr @__quantum__qis__m__body(ptr %77)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %77)
// CHECK-NEXT:    %79 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %80 = call i1 @__quantum__rt__result_equal(ptr %78, ptr %79)
// CHECK-NEXT:    %81 = xor i1 %10, %80
// CHECK-NEXT:    %82 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %82)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %82, ptr %49)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %82, ptr %50)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %82, ptr %52)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %82, ptr %53)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %82)
// CHECK-NEXT:    %83 = call ptr @__quantum__qis__m__body(ptr %82)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %82)
// CHECK-NEXT:    %84 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %85 = call i1 @__quantum__rt__result_equal(ptr %83, ptr %84)
// CHECK-NEXT:    %86 = xor i1 %11, %85
// CHECK-NEXT:    %87 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %87)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %87, ptr %50)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %87, ptr %51)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %87, ptr %53)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %87, ptr %54)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %87)
// CHECK-NEXT:    %88 = call ptr @__quantum__qis__m__body(ptr %87)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %87)
// CHECK-NEXT:    %89 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %90 = call i1 @__quantum__rt__result_equal(ptr %88, ptr %89)
// CHECK-NEXT:    %91 = xor i1 %12, %90
// CHECK-NEXT:    %92 = call i1 @decode(i1 %20, i1 %55, i1 %56, i1 %66, i1 %71, i1 %76)
// CHECK-NEXT:    %93 = call i1 @decode(i1 %57, i1 %24, i1 %28, i1 %81, i1 %86, i1 %91)
// CHECK-NEXT:    br i1 %92, label %94, label %97
// CHECK-EMPTY:
// CHECK-NEXT:  94:                                               ; preds = %61
// CHECK-NEXT:    br i1 %93, label %95, label %96
// CHECK-EMPTY:
// CHECK-NEXT:  95:                                               ; preds = %94
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %52)
// CHECK-NEXT:    br label %99
// CHECK-EMPTY:
// CHECK-NEXT:  96:                                               ; preds = %94
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %52)
// CHECK-NEXT:    br label %99
// CHECK-EMPTY:
// CHECK-NEXT:  97:                                               ; preds = %61
// CHECK-NEXT:    br i1 %93, label %98, label %99
// CHECK-EMPTY:
// CHECK-NEXT:  98:                                               ; preds = %97
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %52)
// CHECK-NEXT:    br label %99
// CHECK-EMPTY:
// CHECK-NEXT:  99:                                               ; preds = %95, %96, %98, %97
// CHECK-NEXT:    br i1 %92, label %100, label %103
// CHECK-EMPTY:
// CHECK-NEXT:  100:                                              ; preds = %99
// CHECK-NEXT:    br i1 %93, label %101, label %102
// CHECK-EMPTY:
// CHECK-NEXT:  101:                                              ; preds = %100
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %53)
// CHECK-NEXT:    br label %105
// CHECK-EMPTY:
// CHECK-NEXT:  102:                                              ; preds = %100
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %53)
// CHECK-NEXT:    br label %105
// CHECK-EMPTY:
// CHECK-NEXT:  103:                                              ; preds = %99
// CHECK-NEXT:    br i1 %93, label %104, label %105
// CHECK-EMPTY:
// CHECK-NEXT:  104:                                              ; preds = %103
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %53)
// CHECK-NEXT:    br label %105
// CHECK-EMPTY:
// CHECK-NEXT:  105:                                              ; preds = %101, %102, %104, %103
// CHECK-NEXT:    br i1 %92, label %106, label %109
// CHECK-EMPTY:
// CHECK-NEXT:  106:                                              ; preds = %105
// CHECK-NEXT:    br i1 %93, label %107, label %108
// CHECK-EMPTY:
// CHECK-NEXT:  107:                                              ; preds = %106
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %54)
// CHECK-NEXT:    br label %111
// CHECK-EMPTY:
// CHECK-NEXT:  108:                                              ; preds = %106
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %54)
// CHECK-NEXT:    br label %111
// CHECK-EMPTY:
// CHECK-NEXT:  109:                                              ; preds = %105
// CHECK-NEXT:    br i1 %93, label %110, label %111
// CHECK-EMPTY:
// CHECK-NEXT:  110:                                              ; preds = %109
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %54)
// CHECK-NEXT:    br label %111
// CHECK-EMPTY:
// CHECK-NEXT:  111:                                              ; preds = %107, %108, %110, %109, %47
// CHECK-NEXT:    %112 = phi ptr [ %48, %110 ], [ %48, %109 ], [ %48, %108 ], [ %48, %107 ], [ %48, %47 ]
// CHECK-NEXT:    %113 = phi ptr [ %49, %110 ], [ %49, %109 ], [ %49, %108 ], [ %49, %107 ], [ %49, %47 ]
// CHECK-NEXT:    %114 = phi ptr [ %50, %110 ], [ %50, %109 ], [ %50, %108 ], [ %50, %107 ], [ %50, %47 ]
// CHECK-NEXT:    %115 = phi ptr [ %51, %110 ], [ %51, %109 ], [ %51, %108 ], [ %51, %107 ], [ %51, %47 ]
// CHECK-NEXT:    %116 = phi ptr [ %52, %110 ], [ %52, %109 ], [ %52, %108 ], [ %52, %107 ], [ %52, %47 ]
// CHECK-NEXT:    %117 = phi ptr [ %53, %110 ], [ %53, %109 ], [ %53, %108 ], [ %53, %107 ], [ %53, %47 ]
// CHECK-NEXT:    %118 = phi ptr [ %54, %110 ], [ %54, %109 ], [ %54, %108 ], [ %54, %107 ], [ %54, %47 ]
// CHECK-NEXT:    %119 = phi i1 [ %65, %110 ], [ %65, %109 ], [ %65, %108 ], [ %65, %107 ], [ %7, %47 ]
// CHECK-NEXT:    %120 = phi i1 [ %70, %110 ], [ %70, %109 ], [ %70, %108 ], [ %70, %107 ], [ %8, %47 ]
// CHECK-NEXT:    %121 = phi i1 [ %75, %110 ], [ %75, %109 ], [ %75, %108 ], [ %75, %107 ], [ %9, %47 ]
// CHECK-NEXT:    %122 = phi i1 [ %80, %110 ], [ %80, %109 ], [ %80, %108 ], [ %80, %107 ], [ %10, %47 ]
// CHECK-NEXT:    %123 = phi i1 [ %85, %110 ], [ %85, %109 ], [ %85, %108 ], [ %85, %107 ], [ %11, %47 ]
// CHECK-NEXT:    %124 = phi i1 [ %90, %110 ], [ %90, %109 ], [ %90, %108 ], [ %90, %107 ], [ %12, %47 ]
// CHECK-NEXT:    %125 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } undef, ptr %112, 0
// CHECK-NEXT:    %126 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %125, ptr %113, 1
// CHECK-NEXT:    %127 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %126, ptr %114, 2
// CHECK-NEXT:    %128 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %127, ptr %115, 3
// CHECK-NEXT:    %129 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %128, ptr %116, 4
// CHECK-NEXT:    %130 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %129, ptr %117, 5
// CHECK-NEXT:    %131 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %130, ptr %118, 6
// CHECK-NEXT:    %132 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %131, i1 %119, 7
// CHECK-NEXT:    %133 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %132, i1 %120, 8
// CHECK-NEXT:    %134 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %133, i1 %121, 9
// CHECK-NEXT:    %135 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %134, i1 %122, 10
// CHECK-NEXT:    %136 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %135, i1 %123, 11
// CHECK-NEXT:    %137 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %136, i1 %124, 12
// CHECK-NEXT:    ret { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %137
// CHECK-NEXT:  }
// CHECK-EMPTY:


func.func @decode(%fd1 : i1, %fd2 : i1, %fd3 : i1, %sd1 : i1, %sd2 : i1, %sd3 : i1) -> i1 {
  %cTrue = arith.constant true
  %not_sd1 = arith.xori %sd1, %cTrue : i1
  %not_sd2 = arith.xori %sd2, %cTrue : i1
  %not_sd3 = arith.xori %sd3, %cTrue : i1
  %not_fd1 = arith.xori %fd1, %cTrue : i1

  %0 = arith.andi %sd2, %not_sd3 : i1
  %1 = arith.ori %not_fd1, %fd2 : i1
  %2 = arith.ori %1, %fd3 : i1
  %3 = arith.andi %0, %2 : i1

  %4 = arith.andi %sd2, %sd3 : i1

  %5 = arith.andi %not_sd2, %sd3 : i1
  %6 = arith.xori %not_fd1, %fd2 : i1
  %7 = arith.xori %not_fd1, %fd3 : i1
  %8 = arith.ori %6, %7 : i1
  %9 = arith.andi %5, %8 : i1

  %10 = arith.ori %3, %4 : i1
  %11 = arith.ori %10, %9 : i1

  func.return %11 : i1
}
// CHECK-LABEL: define i1 @decode
// CHECK-NEXT:    %7 = xor i1 %4, true
// CHECK-NEXT:    %8 = xor i1 %5, true
// CHECK-NEXT:    %9 = xor i1 %0, true
// CHECK-NEXT:    %10 = and i1 %4, %8
// CHECK-NEXT:    %11 = or i1 %9, %1
// CHECK-NEXT:    %12 = or i1 %11, %2
// CHECK-NEXT:    %13 = and i1 %10, %12
// CHECK-NEXT:    %14 = and i1 %4, %5
// CHECK-NEXT:    %15 = and i1 %7, %5
// CHECK-NEXT:    %16 = xor i1 %9, %1
// CHECK-NEXT:    %17 = xor i1 %9, %2
// CHECK-NEXT:    %18 = or i1 %16, %17
// CHECK-NEXT:    %19 = and i1 %15, %18
// CHECK-NEXT:    %20 = or i1 %13, %14
// CHECK-NEXT:    %21 = or i1 %20, %19
// CHECK-NEXT:    ret i1 %21
// CHECK-NEXT:  }
