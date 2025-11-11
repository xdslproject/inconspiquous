// RUN: TO_QIR

// Iterative phase estimation on "unknown" unitary RX(theta) and eigenstate |+>
// From https://arxiv.org/pdf/2102.01682

func.func @ipe(%theta: !angle.type, %m: i64) -> !angle.type {
  %q = qu.alloc<#qu.plus>

  %c1 = arith.constant 1 : i64

  %a0 = angle.constant<0>

  %a, %q_final = scf.for %iv = %c1 to %m step %c1 iter_args(%phi = %a0, %q_1 = %q) -> (!angle.type, !qu.bit) : i64 {
    %i = arith.subi %m, %iv : i64
    %float = arith.uitofp %i : i64 to f64
    %factor = math.exp2 %float : f64
    %mfloat = arith.negf %float : f64
    %mfactor = math.exp2 %mfloat : f64

    %a = qu.alloc
    %a_1 = qssa.gate<#gate.h> %a

    %angle = angle.scale %theta, %factor
    %rx = gate.dyn_rx<%angle>
    %crx = gate.control %rx : !gate.type<1>
    %a_2, %q_2 = qssa.dyn_gate<%crx> %a_1, %q_1

    %z = gate.dyn_rz<%phi>
    %a_3 = qssa.dyn_gate<%z> %a_2
    %a_4 = qssa.gate<#gate.h> %a_3
    %measured = qssa.measure %a_4

    // Update phi
    %unscaled = angle.constant<pi>
    %scaled = angle.scale %unscaled, %mfactor
    %to_add = arith.select %measured, %scaled, %a0 : !angle.type
    %phi_1 = angle.add %phi, %to_add

    scf.yield %phi_1, %q_2 : !angle.type, !qu.bit
  }

  func.return %a : !angle.type
}
// CHECK-LABEL: @ipe
// CHECK-NEXT:    %3 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %3)
// CHECK-NEXT:    br label %4
// CHECK-EMPTY:
// CHECK-NEXT:  4:                                                ; preds = %9, %2
// CHECK-NEXT:    %5 = phi i64 [ %23, %9 ], [ 1, %2 ]
// CHECK-NEXT:    %6 = phi double [ %22, %9 ], [ 0.000000e+00, %2 ]
// CHECK-NEXT:    %7 = phi ptr [ %7, %9 ], [ %3, %2 ]
// CHECK-NEXT:    %8 = icmp slt i64 %5, %1
// CHECK-NEXT:    br i1 %8, label %9, label %24
// CHECK-EMPTY:
// CHECK-NEXT:  9:                                                ; preds = %4
// CHECK-NEXT:    %10 = sub i64 %1, %5
// CHECK-NEXT:    %11 = uitofp i64 %10 to double
// CHECK-NEXT:    %12 = call double @llvm.exp2.f64(double %11)
// CHECK-NEXT:    %13 = fneg double %11
// CHECK-NEXT:    %14 = call double @llvm.exp2.f64(double %13)
// CHECK-NEXT:    %15 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %15)
// CHECK-NEXT:    %16 = fmul double %0, %12
// CHECK-NEXT:    call void @__quantum__qis__rx__ctl(double %16, ptr %15, ptr %7)
// CHECK-NEXT:    call void @__quantum__qis__rz__body(double %6, ptr %15)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %15)
// CHECK-NEXT:    %17 = call ptr @__quantum__qis__m__body(ptr %15)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %15)
// CHECK-NEXT:    %18 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %19 = call i1 @__quantum__rt__result_equal(ptr %17, ptr %18)
// CHECK-NEXT:    %20 = fmul double %14, 0x400921FB54442D18
// CHECK-NEXT:    %21 = select i1 %19, double %20, double 0.000000e+00
// CHECK-NEXT:    %22 = fadd double %6, %21
// CHECK-NEXT:    %23 = add i64 %5, 1
// CHECK-NEXT:    br label %4
// CHECK-EMPTY:
// CHECK-NEXT:  24:                                               ; preds = %4
// CHECK-NEXT:    ret double %6
// CHECK-NEXT:  }
