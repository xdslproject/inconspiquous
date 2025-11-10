// RUN: TO_QIR

// Random walk phase estimation with "unknown" unitary family U(t) = RZ(theta*t)
// From https://arxiv.org/pdf/2208.04526

func.func @rwpe(%mu0 : !angle.type, %sigma0 : f64, %theta : !angle.type, %iter: i64) -> !angle.type {
  %a_pi = angle.constant<pi>
  %minus_pi_by_2 = arith.constant -1.57079632679 : f64
  %root_e = arith.constant 1.6487212707 : f64
  %root_e-1_by_e = arith.constant 0.79506009762 : f64

  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c1_float = arith.constant 1.0 : f64

  %nexp = arith.subi %iter, %c1 : i64

  %q = qu.alloc

  %mu_final, %sigma_final, %q_final = scf.for %iv = %c0 to %nexp step %c1 iter_args(%mu = %mu0, %sigma = %sigma0, %q_1 = %q) -> (!angle.type, f64, !qu.bit) : i64 {
    %0 = arith.mulf %minus_pi_by_2, %sigma : f64
    %1 = angle.scale %a_pi, %0
    %winv = angle.add %mu, %1

    %t = arith.divf %c1_float, %sigma : f64

    %a = qu.alloc
    %a_1 = qssa.gate<#gate.h> %a

    %mt = arith.negf %t : f64
    %angle1 = angle.scale %winv, %mt
    %rz1 = gate.dyn_rz<%angle1>
    %a_2 = qssa.dyn_gate<%rz1> %a_1

    %angle2 = angle.scale %theta, %t
    %rz2 = gate.dyn_rz<%angle2>
    %crz = gate.control %rz2 : !gate.type<1>
    %a_3, %q_2 = qssa.dyn_gate<%crz> %a_2, %q_1

    %a_4 = qssa.gate<#gate.h> %a_3
    %m = qssa.measure %a_4

    %sigma_over_root_e = arith.divf %sigma, %root_e : f64
    %adjust = angle.scale %a_pi, %sigma_over_root_e

    %negated = angle.cond_negate %m, %adjust

    %mu_new = angle.add %mu, %negated

    %sigma_new = arith.mulf %sigma, %root_e-1_by_e : f64

    scf.yield %mu_new, %sigma_new, %q_2 : !angle.type, f64, !qu.bit
  }
  func.return %mu_final : !angle.type
}
// CHECK-LABEL: @rwpe
// CHECK-NEXT:    %5 = sub i64 %3, 1
// CHECK-NEXT:    %6 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    br label %7
// CHECK-EMPTY:
// CHECK-NEXT:  7:                                                ; preds = %13, %4
// CHECK-NEXT:    %8 = phi i64 [ %31, %13 ], [ 0, %4 ]
// CHECK-NEXT:    %9 = phi double [ %29, %13 ], [ %0, %4 ]
// CHECK-NEXT:    %10 = phi double [ %30, %13 ], [ %1, %4 ]
// CHECK-NEXT:    %11 = phi ptr [ %11, %13 ], [ %6, %4 ]
// CHECK-NEXT:    %12 = icmp slt i64 %8, %5
// CHECK-NEXT:    br i1 %12, label %13, label %32
// CHECK-EMPTY:
// CHECK-NEXT:  13:                                               ; preds = %7
// CHECK-NEXT:    %14 = fmul double %10, 0xBFF921FB5443D6F4
// CHECK-NEXT:    %15 = fmul double %14, 0x400921FB54442D18
// CHECK-NEXT:    %16 = fadd double %9, %15
// CHECK-NEXT:    %17 = fdiv double 1.000000e+00, %10
// CHECK-NEXT:    %18 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %18)
// CHECK-NEXT:    %19 = fneg double %17
// CHECK-NEXT:    %20 = fmul double %16, %19
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %20, ptr %18)
// CHECK-NEXT:    %21 = fmul double %2, %17
// CHECK-NEXT:    call void @__quantum__qis__rz__ctl(double %21, ptr %18, ptr %11)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %18)
// CHECK-NEXT:    %22 = call ptr @__quantum__qis__m__body(ptr %18)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %18)
// CHECK-NEXT:    %23 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %24 = call i1 @__quantum__rt__result_equal(ptr %22, ptr %23)
// CHECK-NEXT:    %25 = fdiv double %10, 0x3FFA61298E1E045B
// CHECK-NEXT:    %26 = fmul double %25, 0x400921FB54442D18
// CHECK-NEXT:    %27 = fneg double %26
// CHECK-NEXT:    %28 = select i1 %24, double %27, double %26
// CHECK-NEXT:    %29 = fadd double %9, %28
// CHECK-NEXT:    %30 = fmul double %10, 0x3FE97121DFB43D2C
// CHECK-NEXT:    %31 = add i64 %8, 1
// CHECK-NEXT:    br label %7
// CHECK-EMPTY:
// CHECK-NEXT:  32:                                               ; preds = %7
// CHECK-NEXT:    ret double %9
// CHECK-NEXT:  }
