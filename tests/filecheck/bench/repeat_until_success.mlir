// RUN: quopt %s -p convert-qssa-to-qref,lower-xzs-to-select,cse,canonicalize,lower-dyn-gate-to-scf,canonicalize,convert-qref-to-qir,convert-qir-to-llvm | mlir-opt -p 'builtin.module(convert-scf-to-cf,canonicalize,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm)' | mlir-translate --mlir-to-llvmir | filecheck %s

// From https://arxiv.org/pdf/1311.1074 figure 1

func.func @a(%q : !qu.bit) -> !qu.bit {
  %res = scf.while (%q_1 = %q) : (!qu.bit) -> !qu.bit {
    %a1 = qu.alloc<#qu.plus>
    %a2 = qu.alloc<#qu.plus>

    %a1_1, %a2_1, %q_2 = qssa.gate<#gate.toffoli> %a1, %a2, %q_1
    %q_3 = qssa.gate<#gate.s> %q_2
    %a1_2, %a2_2, %q_4 = qssa.gate<#gate.toffoli> %a1_1, %a2_1, %q_3

    %q_5 = qssa.gate<#gate.z> %q_4

    %m1 = qssa.measure<#measurement.x_basis> %a1_2
    %m2 = qssa.measure<#measurement.x_basis> %a2_2

    %m = arith.ori %m1, %m2 : i1

    scf.condition(%m) %q_5 : !qu.bit
  } do {
    ^bb0(%q_1 : !qu.bit):
    scf.yield %q_1 : !qu.bit
  }
  func.return %res : !qu.bit
}
// CHECK-LABEL: @a
// CHECK-NEXT:    br label %2
// CHECK-EMPTY:
// CHECK-NEXT:  2:                                                ; preds = %2, %1
// CHECK-NEXT:    %3 = phi ptr [ %3, %2 ], [ %0, %1 ]
// CHECK-NEXT:    %4 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    %5 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__ccx__body(ptr %4, ptr %5, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__s__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__ccx__body(ptr %4, ptr %5, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    %6 = call ptr @__quantum__qis__m__body(ptr %4)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %4)
// CHECK-NEXT:    %7 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %8 = call i1 @__quantum__rt__result_equal(ptr %6, ptr %7)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %9 = call ptr @__quantum__qis__m__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %5)
// CHECK-NEXT:    %10 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %11 = call i1 @__quantum__rt__result_equal(ptr %9, ptr %10)
// CHECK-NEXT:    %12 = or i1 %8, %11
// CHECK-NEXT:    br i1 %12, label %2, label %13
// CHECK-EMPTY:
// CHECK-NEXT:  13:                                               ; preds = %2
// CHECK-NEXT:    ret ptr %3
// CHECK-NEXT:  }

func.func @b(%q : !qu.bit) -> !qu.bit {
  %res = scf.while (%q_1 = %q) : (!qu.bit) -> !qu.bit {
    %a1 = qu.alloc<#qu.plus>
    %a2 = qu.alloc<#qu.plus>
    %a3 = qu.alloc

    %a1_1, %a2_1, %a3_1 = qssa.gate<#gate.toffoli> %a1, %a2, %a3
    %a3_2, %q_2 = qssa.gate<#gate.cx> %a3_1, %q_1
    %q_3 = qssa.gate<#gate.s> %q_2
    %a3_3, %q_4 = qssa.gate<#gate.cx> %a3_2, %q_3
    %q_5 = qssa.gate<#gate.z> %q_4

    %m3 = qssa.measure<#measurement.x_basis> %a3_3

    %cz = gate.constant #gate.cz
    %id = gate.constant #gate.id<2>
    %g = arith.select %m3, %cz, %id : !gate.type<2>

    %a1_2, %a2_2 = qssa.dyn_gate<%g> %a1_1, %a2_1

    %m1 = qssa.measure<#measurement.x_basis> %a1_2
    %m2 = qssa.measure<#measurement.x_basis> %a2_2

    %m = arith.ori %m1, %m2 : i1

    scf.condition(%m) %q_4 : !qu.bit
  } do {
    ^bb0(%q_1 : !qu.bit):
    scf.yield %q_1 : !qu.bit
  }
  func.return %res : !qu.bit
}
// CHECK-LABEL: @b
// CHECK-NEXT:    br label %2
// CHECK-EMPTY:
// CHECK-NEXT:  2:                                                ; preds = %11, %1
// CHECK-NEXT:    %3 = phi ptr [ %3, %11 ], [ %0, %1 ]
// CHECK-NEXT:    %4 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    %5 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %6 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__ccx__body(ptr %4, ptr %5, ptr %6)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %6, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__s__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %6, ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %6)
// CHECK-NEXT:    %7 = call ptr @__quantum__qis__m__body(ptr %6)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %6)
// CHECK-NEXT:    %8 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %9 = call i1 @__quantum__rt__result_equal(ptr %7, ptr %8)
// CHECK-NEXT:    br i1 %9, label %10, label %11
// CHECK-EMPTY:
// CHECK-NEXT:  10:                                               ; preds = %2
// CHECK-NEXT:    call void @__quantum__qis__cz__body(ptr %4, ptr %5)
// CHECK-NEXT:    br label %11
// CHECK-EMPTY:
// CHECK-NEXT:  11:                                               ; preds = %10, %2
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    %12 = call ptr @__quantum__qis__m__body(ptr %4)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %4)
// CHECK-NEXT:    %13 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %14 = call i1 @__quantum__rt__result_equal(ptr %12, ptr %13)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %15 = call ptr @__quantum__qis__m__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %5)
// CHECK-NEXT:    %16 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %17 = call i1 @__quantum__rt__result_equal(ptr %15, ptr %16)
// CHECK-NEXT:    %18 = or i1 %14, %17
// CHECK-NEXT:    br i1 %18, label %2, label %19
// CHECK-EMPTY:
// CHECK-NEXT:  19:                                               ; preds = %11
// CHECK-NEXT:    ret ptr %3
// CHECK-NEXT:  }

func.func @c(%q : !qu.bit) -> !qu.bit {
  %res = scf.while (%q_1 = %q) : (!qu.bit) -> !qu.bit {
    %q_2 = qssa.gate<#gate.t> %q_1
    %q_3 = qssa.gate<#gate.z> %q_2

    %a1 = qu.alloc<#qu.plus>
    %a2 = qu.alloc<#qu.plus>

    %a1_1 = qssa.gate<#gate.t_dagger> %a1
    %a2_1, %a1_2 = qssa.gate<#gate.cx> %a2, %a1_1
    %q_4, %a2_2 = qssa.gate<#gate.cx> %q_3, %a2_1
    %a1_3 = qssa.gate<#gate.t> %a1_2
    %a2_3 = qssa.gate<#gate.t> %a2_2

    %m1 = qssa.measure<#measurement.x_basis> %a1_3
    %m2 = qssa.measure<#measurement.x_basis> %a2_3

    %m = arith.ori %m1, %m2 : i1

    scf.condition(%m) %q_4 : !qu.bit
  } do {
    ^bb0(%q_1 : !qu.bit):
    scf.yield %q_1 : !qu.bit
  }
  func.return %res : !qu.bit
}
// CHECK-LABEL: @c
// CHECK-NEXT:    br label %2
// CHECK-EMPTY:
// CHECK-NEXT:  2:                                                ; preds = %2, %1
// CHECK-NEXT:    %3 = phi ptr [ %3, %2 ], [ %0, %1 ]
// CHECK-NEXT:    call void @__quantum__qis__t__body(ptr %3)
// CHECK-NEXT:    call void @__quantum__qis__z__body(ptr %3)
// CHECK-NEXT:    %4 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    %5 = call ptr @__quantum__rt__qubit_allocate()
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__t__adj(ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %5, ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__cnot__body(ptr %3, ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__t__body(ptr %4)
// CHECK-NEXT:    call void @__quantum__qis__t__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %4)
// CHECK-NEXT:    %6 = call ptr @__quantum__qis__m__body(ptr %4)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %4)
// CHECK-NEXT:    %7 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %8 = call i1 @__quantum__rt__result_equal(ptr %6, ptr %7)
// CHECK-NEXT:    call void @__quantum__qis__h__body(ptr %5)
// CHECK-NEXT:    %9 = call ptr @__quantum__qis__m__body(ptr %5)
// CHECK-NEXT:    call void @__quantum__rt__qubit_release(ptr %5)
// CHECK-NEXT:    %10 = call ptr @__quantum__rt__result_get_one()
// CHECK-NEXT:    %11 = call i1 @__quantum__rt__result_equal(ptr %9, ptr %10)
// CHECK-NEXT:    %12 = or i1 %8, %11
// CHECK-NEXT:    br i1 %12, label %2, label %13
// CHECK-EMPTY:
// CHECK-NEXT:  13:                                               ; preds = %2
// CHECK-NEXT:    ret ptr %3
// CHECK-NEXT:  }
