// RUN: TO_QIR

// Adaptive QEC cycle from https://arxiv.org/pdf/2107.07505
// We perform the corrections instead of updating a pauli frame.

func.func @adaptive_qec_cycle(
  %q1: !qu.bit,
  %q2: !qu.bit,
  %q3: !qu.bit,
  %q4: !qu.bit,
  %q5: !qu.bit,
  %q6: !qu.bit,
  %q7: !qu.bit,
  %prev_s1: i1,
  %prev_s2: i1,
  %prev_s3: i1,
  %prev_s4: i1,
  %prev_s5: i1,
  %prev_s6: i1
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
// CHECK-NEXT:    %18 = call i1 @__quantum__rt__read_result__body(ptr %17)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %14)
// CHECK-NEXT:    %19 = xor i1 %18, %7
// CHECK-NEXT:    %20 = call ptr @__quantum__qis__m__body(ptr %15)
// CHECK-NEXT:    %21 = call i1 @__quantum__rt__read_result__body(ptr %20)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %15)
// CHECK-NEXT:    %22 = xor i1 %21, %11
// CHECK-NEXT:    %23 = call ptr @__quantum__qis__m__body(ptr %16)
// CHECK-NEXT:    %24 = call i1 @__quantum__rt__read_result__body(ptr %23)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %16)
// CHECK-NEXT:    %25 = xor i1 %24, %12
// CHECK-NEXT:    %26 = or i1 %19, %22
// CHECK-NEXT:    %27 = or i1 %26, %25
// CHECK-NEXT:    br i1 %27, label %28, label %41
// CHECK-EMPTY:
// CHECK-NEXT:  28:                                               ; preds = %13
// CHECK-NEXT:    %29 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %30 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    %31 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %30)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %31)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %3, ptr %29)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %31, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %30, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %30, ptr %29)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %0, ptr %29)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %31, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %30, ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %1, ptr %29)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %31, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %30, ptr %2)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %31, ptr %29)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %2, ptr %29)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %31, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %30, ptr %1)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %30)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %31)
// CHECK-NEXT:    %32 = call ptr @__quantum__qis__m__body(ptr %29)
// CHECK-NEXT:    %33 = call i1 @__quantum__rt__read_result__body(ptr %32)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %29)
// CHECK-NEXT:    %34 = xor i1 %33, %8
// CHECK-NEXT:    %35 = call ptr @__quantum__qis__m__body(ptr %30)
// CHECK-NEXT:    %36 = call i1 @__quantum__rt__read_result__body(ptr %35)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %30)
// CHECK-NEXT:    %37 = xor i1 %36, %9
// CHECK-NEXT:    %38 = call ptr @__quantum__qis__m__body(ptr %31)
// CHECK-NEXT:    %39 = call i1 @__quantum__rt__read_result__body(ptr %38)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %31)
// CHECK-NEXT:    %40 = xor i1 %39, %10
// CHECK-NEXT:    br label %41
// CHECK-EMPTY:
// CHECK-NEXT:  41:                                               ; preds = %28, %13
// CHECK-NEXT:    %42 = phi ptr [ %0, %28 ], [ %0, %13 ]
// CHECK-NEXT:    %43 = phi ptr [ %1, %28 ], [ %1, %13 ]
// CHECK-NEXT:    %44 = phi ptr [ %2, %28 ], [ %2, %13 ]
// CHECK-NEXT:    %45 = phi ptr [ %3, %28 ], [ %3, %13 ]
// CHECK-NEXT:    %46 = phi ptr [ %4, %28 ], [ %4, %13 ]
// CHECK-NEXT:    %47 = phi ptr [ %5, %28 ], [ %5, %13 ]
// CHECK-NEXT:    %48 = phi ptr [ %6, %28 ], [ %6, %13 ]
// CHECK-NEXT:    %49 = phi i1 [ %34, %28 ], [ false, %13 ]
// CHECK-NEXT:    %50 = phi i1 [ %37, %28 ], [ false, %13 ]
// CHECK-NEXT:    %51 = phi i1 [ %40, %28 ], [ false, %13 ]
// CHECK-NEXT:    %52 = or i1 %49, %50
// CHECK-NEXT:    %53 = or i1 %52, %51
// CHECK-NEXT:    %54 = or i1 %27, %53
// CHECK-NEXT:    br i1 %54, label %55, label %99
// CHECK-EMPTY:
// CHECK-NEXT:  55:                                               ; preds = %41
// CHECK-NEXT:    %56 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %42, ptr %56)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %43, ptr %56)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %44, ptr %56)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %45, ptr %56)
// CHECK-NEXT:    %57 = call ptr @__quantum__qis__m__body(ptr %56)
// CHECK-NEXT:    %58 = call i1 @__quantum__rt__read_result__body(ptr %57)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %56)
// CHECK-NEXT:    %59 = xor i1 %7, %58
// CHECK-NEXT:    %60 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %43, ptr %60)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %44, ptr %60)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %46, ptr %60)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %47, ptr %60)
// CHECK-NEXT:    %61 = call ptr @__quantum__qis__m__body(ptr %60)
// CHECK-NEXT:    %62 = call i1 @__quantum__rt__read_result__body(ptr %61)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %60)
// CHECK-NEXT:    %63 = xor i1 %8, %62
// CHECK-NEXT:    %64 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %44, ptr %64)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %45, ptr %64)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %47, ptr %64)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %48, ptr %64)
// CHECK-NEXT:    %65 = call ptr @__quantum__qis__m__body(ptr %64)
// CHECK-NEXT:    %66 = call i1 @__quantum__rt__read_result__body(ptr %65)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %64)
// CHECK-NEXT:    %67 = xor i1 %9, %66
// CHECK-NEXT:    %68 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %68)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %68, ptr %42)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %68, ptr %43)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %68, ptr %44)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %68, ptr %45)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %68)
// CHECK-NEXT:    %69 = call ptr @__quantum__qis__m__body(ptr %68)
// CHECK-NEXT:    %70 = call i1 @__quantum__rt__read_result__body(ptr %69)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %68)
// CHECK-NEXT:    %71 = xor i1 %10, %70
// CHECK-NEXT:    %72 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %72)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %72, ptr %43)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %72, ptr %44)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %72, ptr %46)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %72, ptr %47)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %72)
// CHECK-NEXT:    %73 = call ptr @__quantum__qis__m__body(ptr %72)
// CHECK-NEXT:    %74 = call i1 @__quantum__rt__read_result__body(ptr %73)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %72)
// CHECK-NEXT:    %75 = xor i1 %11, %74
// CHECK-NEXT:    %76 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %76)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %76, ptr %44)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %76, ptr %45)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %76, ptr %47)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %76, ptr %48)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %76)
// CHECK-NEXT:    %77 = call ptr @__quantum__qis__m__body(ptr %76)
// CHECK-NEXT:    %78 = call i1 @__quantum__rt__read_result__body(ptr %77)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %76)
// CHECK-NEXT:    %79 = xor i1 %12, %78
// CHECK-NEXT:    %80 = call i1 @decode(i1 %19, i1 %49, i1 %50, i1 %59, i1 %63, i1 %67)
// CHECK-NEXT:    %81 = call i1 @decode(i1 %51, i1 %22, i1 %25, i1 %71, i1 %75, i1 %79)
// CHECK-NEXT:    br i1 %80, label %82, label %85
// CHECK-EMPTY:
// CHECK-NEXT:  82:                                               ; preds = %55
// CHECK-NEXT:    br i1 %81, label %83, label %84
// CHECK-EMPTY:
// CHECK-NEXT:  83:                                               ; preds = %82
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %46)
// CHECK-NEXT:    br label %87
// CHECK-EMPTY:
// CHECK-NEXT:  84:                                               ; preds = %82
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %46)
// CHECK-NEXT:    br label %87
// CHECK-EMPTY:
// CHECK-NEXT:  85:                                               ; preds = %55
// CHECK-NEXT:    br i1 %81, label %86, label %87
// CHECK-EMPTY:
// CHECK-NEXT:  86:                                               ; preds = %85
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %46)
// CHECK-NEXT:    br label %87
// CHECK-EMPTY:
// CHECK-NEXT:  87:                                               ; preds = %83, %84, %86, %85
// CHECK-NEXT:    br i1 %80, label %88, label %91
// CHECK-EMPTY:
// CHECK-NEXT:  88:                                               ; preds = %87
// CHECK-NEXT:    br i1 %81, label %89, label %90
// CHECK-EMPTY:
// CHECK-NEXT:  89:                                               ; preds = %88
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %47)
// CHECK-NEXT:    br label %93
// CHECK-EMPTY:
// CHECK-NEXT:  90:                                               ; preds = %88
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %47)
// CHECK-NEXT:    br label %93
// CHECK-EMPTY:
// CHECK-NEXT:  91:                                               ; preds = %87
// CHECK-NEXT:    br i1 %81, label %92, label %93
// CHECK-EMPTY:
// CHECK-NEXT:  92:                                               ; preds = %91
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %47)
// CHECK-NEXT:    br label %93
// CHECK-EMPTY:
// CHECK-NEXT:  93:                                               ; preds = %89, %90, %92, %91
// CHECK-NEXT:    br i1 %80, label %94, label %97
// CHECK-EMPTY:
// CHECK-NEXT:  94:                                               ; preds = %93
// CHECK-NEXT:    br i1 %81, label %95, label %96
// CHECK-EMPTY:
// CHECK-NEXT:  95:                                               ; preds = %94
// CHECK-NEXT:    call void @__quantum__qis__y__body(ptr %48)
// CHECK-NEXT:    br label %99
// CHECK-EMPTY:
// CHECK-NEXT:  96:                                               ; preds = %94
// CHECK-NEXT:    call void @__quantum__qis__x__body(ptr %48)
// CHECK-NEXT:    br label %99
// CHECK-EMPTY:
// CHECK-NEXT:  97:                                               ; preds = %93
// CHECK-NEXT:    br i1 %81, label %98, label %99
// CHECK-EMPTY:
// CHECK-NEXT:  98:                                               ; preds = %97
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %48)
// CHECK-NEXT:    br label %99
// CHECK-EMPTY:
// CHECK-NEXT:  99:                                               ; preds = %95, %96, %98, %97, %41
// CHECK-NEXT:    %100 = phi ptr [ %42, %98 ], [ %42, %97 ], [ %42, %96 ], [ %42, %95 ], [ %42, %41 ]
// CHECK-NEXT:    %101 = phi ptr [ %43, %98 ], [ %43, %97 ], [ %43, %96 ], [ %43, %95 ], [ %43, %41 ]
// CHECK-NEXT:    %102 = phi ptr [ %44, %98 ], [ %44, %97 ], [ %44, %96 ], [ %44, %95 ], [ %44, %41 ]
// CHECK-NEXT:    %103 = phi ptr [ %45, %98 ], [ %45, %97 ], [ %45, %96 ], [ %45, %95 ], [ %45, %41 ]
// CHECK-NEXT:    %104 = phi ptr [ %46, %98 ], [ %46, %97 ], [ %46, %96 ], [ %46, %95 ], [ %46, %41 ]
// CHECK-NEXT:    %105 = phi ptr [ %47, %98 ], [ %47, %97 ], [ %47, %96 ], [ %47, %95 ], [ %47, %41 ]
// CHECK-NEXT:    %106 = phi ptr [ %48, %98 ], [ %48, %97 ], [ %48, %96 ], [ %48, %95 ], [ %48, %41 ]
// CHECK-NEXT:    %107 = phi i1 [ %58, %98 ], [ %58, %97 ], [ %58, %96 ], [ %58, %95 ], [ %7, %41 ]
// CHECK-NEXT:    %108 = phi i1 [ %62, %98 ], [ %62, %97 ], [ %62, %96 ], [ %62, %95 ], [ %8, %41 ]
// CHECK-NEXT:    %109 = phi i1 [ %66, %98 ], [ %66, %97 ], [ %66, %96 ], [ %66, %95 ], [ %9, %41 ]
// CHECK-NEXT:    %110 = phi i1 [ %70, %98 ], [ %70, %97 ], [ %70, %96 ], [ %70, %95 ], [ %10, %41 ]
// CHECK-NEXT:    %111 = phi i1 [ %74, %98 ], [ %74, %97 ], [ %74, %96 ], [ %74, %95 ], [ %11, %41 ]
// CHECK-NEXT:    %112 = phi i1 [ %78, %98 ], [ %78, %97 ], [ %78, %96 ], [ %78, %95 ], [ %12, %41 ]
// CHECK-NEXT:    %113 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } poison, ptr %100, 0
// CHECK-NEXT:    %114 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %113, ptr %101, 1
// CHECK-NEXT:    %115 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %114, ptr %102, 2
// CHECK-NEXT:    %116 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %115, ptr %103, 3
// CHECK-NEXT:    %117 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %116, ptr %104, 4
// CHECK-NEXT:    %118 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %117, ptr %105, 5
// CHECK-NEXT:    %119 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %118, ptr %106, 6
// CHECK-NEXT:    %120 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %119, i1 %107, 7
// CHECK-NEXT:    %121 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %120, i1 %108, 8
// CHECK-NEXT:    %122 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %121, i1 %109, 9
// CHECK-NEXT:    %123 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %122, i1 %110, 10
// CHECK-NEXT:    %124 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %123, i1 %111, 11
// CHECK-NEXT:    %125 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %124, i1 %112, 12
// CHECK-NEXT:    ret { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i1, i1, i1, i1, i1, i1 } %125
// CHECK-NEXT:  }


func.func @decode(%fd1: i1, %fd2: i1, %fd3: i1, %sd1: i1, %sd2: i1, %sd3: i1) -> i1 {
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
