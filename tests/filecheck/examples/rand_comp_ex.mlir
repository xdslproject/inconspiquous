// RUN: quopt -p randomized-comp %s | filecheck %s

// CHECK:      %q0 = qubit.alloc
// CHECK-NEXT: %q1 = qubit.alloc
// CHECK-NEXT: %q2 = qubit.alloc
// CHECK-NEXT: %0 = prob.uniform : i1
// CHECK-NEXT: %1 = prob.uniform : i1
// CHECK-NEXT: %2 = gate.constant #gate.id
// CHECK-NEXT: %3 = gate.constant #gate.x
// CHECK-NEXT: %4 = gate.constant #gate.z
// CHECK-NEXT: %5 = arith.select %0, %3, %2 : !gate.type<1>
// CHECK-NEXT: %6 = qssa.dyn_gate<%5> %q0 : !qubit.bit
// CHECK-NEXT: %7 = arith.select %1, %4, %2 : !gate.type<1>
// CHECK-NEXT: %8 = qssa.dyn_gate<%7> %6 : !qubit.bit
// CHECK-NEXT: %9 = qssa.gate<#gate.h> %8 : !qubit.bit
// CHECK-NEXT: %10 = arith.select %1, %3, %2 : !gate.type<1>
// CHECK-NEXT: %11 = qssa.dyn_gate<%10> %9 : !qubit.bit
// CHECK-NEXT: %12 = arith.select %0, %4, %2 : !gate.type<1>
// CHECK-NEXT: %q0_1 = qssa.dyn_gate<%12> %11 : !qubit.bit
// CHECK-NEXT: %13 = prob.uniform : i1
// CHECK-NEXT: %14 = prob.uniform : i1
// CHECK-NEXT: %15 = prob.uniform : i1
// CHECK-NEXT: %16 = prob.uniform : i1
// CHECK-NEXT: %17 = gate.constant #gate.id
// CHECK-NEXT: %18 = gate.constant #gate.x
// CHECK-NEXT: %19 = gate.constant #gate.z
// CHECK-NEXT: %20 = arith.select %13, %18, %17 : !gate.type<1>
// CHECK-NEXT: %21 = arith.select %14, %18, %17 : !gate.type<1>
// CHECK-NEXT: %22 = arith.select %15, %19, %17 : !gate.type<1>
// CHECK-NEXT: %23 = arith.select %16, %19, %17 : !gate.type<1>
// CHECK-NEXT: %24 = qssa.dyn_gate<%20> %q0_1 : !qubit.bit
// CHECK-NEXT: %25 = qssa.dyn_gate<%22> %24 : !qubit.bit
// CHECK-NEXT: %26 = qssa.dyn_gate<%21> %q1 : !qubit.bit
// CHECK-NEXT: %27 = qssa.dyn_gate<%23> %26 : !qubit.bit
// CHECK-NEXT: %28, %29 = qssa.gate<#gate.cnot> %25, %27 : !qubit.bit, !qubit.bit
// CHECK-NEXT: %30 = qssa.dyn_gate<%22> %28 : !qubit.bit
// CHECK-NEXT: %31 = qssa.dyn_gate<%23> %30 : !qubit.bit
// CHECK-NEXT: %32 = qssa.dyn_gate<%23> %29 : !qubit.bit
// CHECK-NEXT: %33 = qssa.dyn_gate<%20> %32 : !qubit.bit
// CHECK-NEXT: %q0_2 = qssa.dyn_gate<%20> %31 : !qubit.bit
// CHECK-NEXT: %q1_1 = qssa.dyn_gate<%21> %33 : !qubit.bit
// CHECK-NEXT: %34 = prob.uniform : i1
// CHECK-NEXT: %35 = prob.uniform : i1
// CHECK-NEXT: %36 = prob.uniform : i1
// CHECK-NEXT: %37 = prob.uniform : i1
// CHECK-NEXT: %38 = gate.constant #gate.id
// CHECK-NEXT: %39 = gate.constant #gate.x
// CHECK-NEXT: %40 = gate.constant #gate.z
// CHECK-NEXT: %41 = arith.select %34, %39, %38 : !gate.type<1>
// CHECK-NEXT: %42 = arith.select %35, %39, %38 : !gate.type<1>
// CHECK-NEXT: %43 = arith.select %36, %40, %38 : !gate.type<1>
// CHECK-NEXT: %44 = arith.select %37, %40, %38 : !gate.type<1>
// CHECK-NEXT: %45 = qssa.dyn_gate<%41> %q0_2 : !qubit.bit
// CHECK-NEXT: %46 = qssa.dyn_gate<%43> %45 : !qubit.bit
// CHECK-NEXT: %47 = qssa.dyn_gate<%42> %q2 : !qubit.bit
// CHECK-NEXT: %48 = qssa.dyn_gate<%44> %47 : !qubit.bit
// CHECK-NEXT: %49, %50 = qssa.gate<#gate.cnot> %46, %48 : !qubit.bit, !qubit.bit
// CHECK-NEXT: %51 = qssa.dyn_gate<%43> %49 : !qubit.bit
// CHECK-NEXT: %52 = qssa.dyn_gate<%44> %51 : !qubit.bit
// CHECK-NEXT: %53 = qssa.dyn_gate<%44> %50 : !qubit.bit
// CHECK-NEXT: %54 = qssa.dyn_gate<%41> %53 : !qubit.bit
// CHECK-NEXT: %q0_3 = qssa.dyn_gate<%41> %52 : !qubit.bit
// CHECK-NEXT: %q2_1 = qssa.dyn_gate<%42> %54 : !qubit.bit

%q0 = qubit.alloc
%q1 = qubit.alloc
%q2 = qubit.alloc

%q0_1 = qssa.gate<#gate.h> %q0 : !qubit.bit
%q0_2, %q1_2 = qssa.gate<#gate.cnot> %q0_1, %q1 : !qubit.bit, !qubit.bit
%q0_3, %q2_3 = qssa.gate<#gate.cnot> %q0_2, %q2 : !qubit.bit, !qubit.bit
