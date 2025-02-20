// RUN: quopt %s -p convert-qref-to-qssa,qec-inline | filecheck %s

// CHECK:      func.func @perfect_code(%q1 : !qubit.bit, %q2 : !qubit.bit, %q3 : !qubit.bit, %q4 : !qubit.bit, %q5 : !qubit.bit) {
// CHECK-NEXT:   %a1 = qubit.alloc
// CHECK-NEXT:   %a1_1 = qssa.gate<#gate.h> %a1
// CHECK-NEXT:   %a1_2, %q1_1 = qssa.gate<#gate.cx> %a1_1, %q1
// CHECK-NEXT:   %a1_3, %q2_1 = qssa.gate<#gate.cz> %a1_2, %q2
// CHECK-NEXT:   %a1_4, %q3_1 = qssa.gate<#gate.cz> %a1_3, %q3
// CHECK-NEXT:   %a1_5, %q4_1 = qssa.gate<#gate.cx> %a1_4, %q4
// CHECK-NEXT:   %a1_6 = qssa.gate<#gate.h> %a1_5
// CHECK-NEXT:   %s1 = qssa.measure %a1_6
// CHECK-NEXT:   %a2 = qubit.alloc
// CHECK-NEXT:   %a2_1 = qssa.gate<#gate.h> %a2
// CHECK-NEXT:   %a2_2, %q2_2 = qssa.gate<#gate.cx> %a2_1, %q2_1
// CHECK-NEXT:   %a2_3, %q3_2 = qssa.gate<#gate.cz> %a2_2, %q3_1
// CHECK-NEXT:   %a2_4, %q4_2 = qssa.gate<#gate.cz> %a2_3, %q4_1
// CHECK-NEXT:   %a2_5, %q5_1 = qssa.gate<#gate.cx> %a2_4, %q5
// CHECK-NEXT:   %a2_6 = qssa.gate<#gate.h> %a2_5
// CHECK-NEXT:   %s2 = qssa.measure %a2_6
// CHECK-NEXT:   %a3 = qubit.alloc
// CHECK-NEXT:   %a3_1 = qssa.gate<#gate.h> %a3
// CHECK-NEXT:   %a3_2, %q1_2 = qssa.gate<#gate.cx> %a3_1, %q1_1
// CHECK-NEXT:   %a3_3, %q3_3 = qssa.gate<#gate.cx> %a3_2, %q3_2
// CHECK-NEXT:   %a3_4, %q4_3 = qssa.gate<#gate.cz> %a3_3, %q4_2
// CHECK-NEXT:   %a3_5, %q5_2 = qssa.gate<#gate.cz> %a3_4, %q5_1
// CHECK-NEXT:   %a3_6 = qssa.gate<#gate.h> %a3_5
// CHECK-NEXT:   %s3 = qssa.measure %a3_6
// CHECK-NEXT:   %a4 = qubit.alloc
// CHECK-NEXT:   %a4_1 = qssa.gate<#gate.h> %a4
// CHECK-NEXT:   %a4_2, %q1_3 = qssa.gate<#gate.cz> %a4_1, %q1_2
// CHECK-NEXT:   %a4_3, %q2_3 = qssa.gate<#gate.cx> %a4_2, %q2_2
// CHECK-NEXT:   %a4_4, %q4_4 = qssa.gate<#gate.cx> %a4_3, %q4_3
// CHECK-NEXT:   %a4_5, %q5_3 = qssa.gate<#gate.cz> %a4_4, %q5_2
// CHECK-NEXT:   %a4_6 = qssa.gate<#gate.h> %a4_5
// CHECK-NEXT:   %s4 = qssa.measure %a4_6
// CHECK-NEXT:   %x = gate.constant #gate.x
// CHECK-NEXT:   %z = gate.constant #gate.z
// CHECK-NEXT:   %id = gate.constant #gate.id
// CHECK-NEXT:   %0 = arith.constant true
// CHECK-NEXT:   %1 = arith.xori %s1, %s3 : i1
// CHECK-NEXT:   %2 = arith.ori %1, %s2 : i1
// CHECK-NEXT:   %3 = arith.xori %2, %0 : i1
// CHECK-NEXT:   %4 = arith.andi %3, %s4 : i1
// CHECK-NEXT:   %5 = arith.select %4, %x, %id : !gate.type<1>
// CHECK-NEXT:   %q1_4 = qssa.dyn_gate<%5> %q1_3
// CHECK-NEXT:   %6 = arith.andi %3, %s1 : i1
// CHECK-NEXT:   %7 = arith.select %6, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q1_5 = qssa.dyn_gate<%7> %q1_4
// CHECK-NEXT:   %8 = arith.xori %s2, %s4 : i1
// CHECK-NEXT:   %9 = arith.ori %8, %s3 : i1
// CHECK-NEXT:   %10 = arith.xori %9, %0 : i1
// CHECK-NEXT:   %11 = arith.andi %10, %s1 : i1
// CHECK-NEXT:   %12 = arith.select %11, %x, %id : !gate.type<1>
// CHECK-NEXT:   %q2_4 = qssa.dyn_gate<%12> %q2_3
// CHECK-NEXT:   %13 = arith.andi %10, %s2 : i1
// CHECK-NEXT:   %14 = arith.select %13, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q2_5 = qssa.dyn_gate<%14> %q2_4
// CHECK-NEXT:   %15 = arith.xori %s1, %s2 : i1
// CHECK-NEXT:   %16 = arith.ori %15, %s4 : i1
// CHECK-NEXT:   %17 = arith.xori %16, %0 : i1
// CHECK-NEXT:   %18 = arith.andi %17, %s1 : i1
// CHECK-NEXT:   %19 = arith.select %18, %x, %id : !gate.type<1>
// CHECK-NEXT:   %q3_4 = qssa.dyn_gate<%19> %q3_3
// CHECK-NEXT:   %20 = arith.andi %17, %s3 : i1
// CHECK-NEXT:   %21 = arith.select %20, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q3_5 = qssa.dyn_gate<%21> %q3_4
// CHECK-NEXT:   %22 = arith.xori %s1, %s4 : i1
// CHECK-NEXT:   %23 = arith.xori %s2, %s3 : i1
// CHECK-NEXT:   %24 = arith.ori %22, %23 : i1
// CHECK-NEXT:   %25 = arith.xori %24, %0 : i1
// CHECK-NEXT:   %26 = arith.andi %25, %s2 : i1
// CHECK-NEXT:   %27 = arith.select %26, %x, %id : !gate.type<1>
// CHECK-NEXT:   %q4_5 = qssa.dyn_gate<%27> %q4_4
// CHECK-NEXT:   %28 = arith.andi %25, %s1 : i1
// CHECK-NEXT:   %29 = arith.select %28, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q4_6 = qssa.dyn_gate<%29> %q4_5
// CHECK-NEXT:   %30 = arith.xori %s3, %s4 : i1
// CHECK-NEXT:   %31 = arith.ori %30, %s1 : i1
// CHECK-NEXT:   %32 = arith.xori %31, %0 : i1
// CHECK-NEXT:   %33 = arith.andi %32, %s3 : i1
// CHECK-NEXT:   %34 = arith.select %33, %x, %id : !gate.type<1>
// CHECK-NEXT:   %q5_4 = qssa.dyn_gate<%34> %q5_3
// CHECK-NEXT:   %35 = arith.andi %32, %s2 : i1
// CHECK-NEXT:   %36 = arith.select %35, %z, %id : !gate.type<1>
// CHECK-NEXT:   %q5_5 = qssa.dyn_gate<%36> %q5_4
// CHECK-NEXT:   %a1_7 = qubit.alloc
// CHECK-NEXT:   %a1_8 = qssa.gate<#gate.h> %a1_7
// CHECK-NEXT:   %a1_9, %q1_6 = qssa.gate<#gate.cx> %a1_8, %q1_5
// CHECK-NEXT:   %a1_10, %q2_6 = qssa.gate<#gate.cz> %a1_9, %q2_5
// CHECK-NEXT:   %a1_11, %q3_6 = qssa.gate<#gate.cz> %a1_10, %q3_5
// CHECK-NEXT:   %a1_12, %q4_7 = qssa.gate<#gate.cx> %a1_11, %q4_6
// CHECK-NEXT:   %a1_13 = qssa.gate<#gate.h> %a1_12
// CHECK-NEXT:   %s1_1 = qssa.measure %a1_13
// CHECK-NEXT:   %a2_7 = qubit.alloc
// CHECK-NEXT:   %a2_8 = qssa.gate<#gate.h> %a2_7
// CHECK-NEXT:   %a2_9, %q2_7 = qssa.gate<#gate.cx> %a2_8, %q2_6
// CHECK-NEXT:   %a2_10, %q3_7 = qssa.gate<#gate.cz> %a2_9, %q3_6
// CHECK-NEXT:   %a2_11, %q4_8 = qssa.gate<#gate.cz> %a2_10, %q4_7
// CHECK-NEXT:   %a2_12, %q5_6 = qssa.gate<#gate.cx> %a2_11, %q5_5
// CHECK-NEXT:   %a2_13 = qssa.gate<#gate.h> %a2_12
// CHECK-NEXT:   %s2_1 = qssa.measure %a2_13
// CHECK-NEXT:   %a3_7 = qubit.alloc
// CHECK-NEXT:   %a3_8 = qssa.gate<#gate.h> %a3_7
// CHECK-NEXT:   %a3_9, %q1_7 = qssa.gate<#gate.cx> %a3_8, %q1_6
// CHECK-NEXT:   %a3_10, %q3_8 = qssa.gate<#gate.cx> %a3_9, %q3_7
// CHECK-NEXT:   %a3_11, %q4_9 = qssa.gate<#gate.cz> %a3_10, %q4_8
// CHECK-NEXT:   %a3_12, %q5_7 = qssa.gate<#gate.cz> %a3_11, %q5_6
// CHECK-NEXT:   %a3_13 = qssa.gate<#gate.h> %a3_12
// CHECK-NEXT:   %s3_1 = qssa.measure %a3_13
// CHECK-NEXT:   %a4_7 = qubit.alloc
// CHECK-NEXT:   %a4_8 = qssa.gate<#gate.h> %a4_7
// CHECK-NEXT:   %a4_9, %q1_8 = qssa.gate<#gate.cz> %a4_8, %q1_7
// CHECK-NEXT:   %a4_10, %q2_8 = qssa.gate<#gate.cx> %a4_9, %q2_7
// CHECK-NEXT:   %a4_11, %q4_10 = qssa.gate<#gate.cx> %a4_10, %q4_9
// CHECK-NEXT:   %a4_12, %q5_8 = qssa.gate<#gate.cz> %a4_11, %q5_7
// CHECK-NEXT:   %a4_13 = qssa.gate<#gate.h> %a4_12
// CHECK-NEXT:   %s4_1 = qssa.measure %a4_13
// CHECK-NEXT:   %x_1 = gate.constant #gate.x
// CHECK-NEXT:   %z_1 = gate.constant #gate.z
// CHECK-NEXT:   %id_1 = gate.constant #gate.id
// CHECK-NEXT:   %37 = arith.constant true
// CHECK-NEXT:   %38 = arith.xori %s1_1, %s3_1 : i1
// CHECK-NEXT:   %39 = arith.ori %38, %s2_1 : i1
// CHECK-NEXT:   %40 = arith.xori %39, %37 : i1
// CHECK-NEXT:   %41 = arith.andi %40, %s4_1 : i1
// CHECK-NEXT:   %42 = arith.select %41, %x_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q1_9 = qssa.dyn_gate<%42> %q1_8
// CHECK-NEXT:   %43 = arith.andi %40, %s1_1 : i1
// CHECK-NEXT:   %44 = arith.select %43, %z_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q1_10 = qssa.dyn_gate<%44> %q1_9
// CHECK-NEXT:   %45 = arith.xori %s2_1, %s4_1 : i1
// CHECK-NEXT:   %46 = arith.ori %45, %s3_1 : i1
// CHECK-NEXT:   %47 = arith.xori %46, %37 : i1
// CHECK-NEXT:   %48 = arith.andi %47, %s1_1 : i1
// CHECK-NEXT:   %49 = arith.select %48, %x_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q2_9 = qssa.dyn_gate<%49> %q2_8
// CHECK-NEXT:   %50 = arith.andi %47, %s2_1 : i1
// CHECK-NEXT:   %51 = arith.select %50, %z_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q2_10 = qssa.dyn_gate<%51> %q2_9
// CHECK-NEXT:   %52 = arith.xori %s1_1, %s2_1 : i1
// CHECK-NEXT:   %53 = arith.ori %52, %s4_1 : i1
// CHECK-NEXT:   %54 = arith.xori %53, %37 : i1
// CHECK-NEXT:   %55 = arith.andi %54, %s1_1 : i1
// CHECK-NEXT:   %56 = arith.select %55, %x_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q3_9 = qssa.dyn_gate<%56> %q3_8
// CHECK-NEXT:   %57 = arith.andi %54, %s3_1 : i1
// CHECK-NEXT:   %58 = arith.select %57, %z_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q3_10 = qssa.dyn_gate<%58> %q3_9
// CHECK-NEXT:   %59 = arith.xori %s1_1, %s4_1 : i1
// CHECK-NEXT:   %60 = arith.xori %s2_1, %s3_1 : i1
// CHECK-NEXT:   %61 = arith.ori %59, %60 : i1
// CHECK-NEXT:   %62 = arith.xori %61, %37 : i1
// CHECK-NEXT:   %63 = arith.andi %62, %s2_1 : i1
// CHECK-NEXT:   %64 = arith.select %63, %x_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q4_11 = qssa.dyn_gate<%64> %q4_10
// CHECK-NEXT:   %65 = arith.andi %62, %s1_1 : i1
// CHECK-NEXT:   %66 = arith.select %65, %z_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q4_12 = qssa.dyn_gate<%66> %q4_11
// CHECK-NEXT:   %67 = arith.xori %s3_1, %s4_1 : i1
// CHECK-NEXT:   %68 = arith.ori %67, %s1_1 : i1
// CHECK-NEXT:   %69 = arith.xori %68, %37 : i1
// CHECK-NEXT:   %70 = arith.andi %69, %s3_1 : i1
// CHECK-NEXT:   %71 = arith.select %70, %x_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q5_9 = qssa.dyn_gate<%71> %q5_8
// CHECK-NEXT:   %72 = arith.andi %69, %s2_1 : i1
// CHECK-NEXT:   %73 = arith.select %72, %z_1, %id_1 : !gate.type<1>
// CHECK-NEXT:   %q5_10 = qssa.dyn_gate<%73> %q5_9
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @perfect_code(%q1: !qubit.bit, %q2: !qubit.bit, %q3: !qubit.bit, %q4: !qubit.bit, %q5: !qubit.bit) {
  qref.gate<#qec.perfect> %q1, %q2, %q3, %q4, %q5
  qref.gate<#qec.perfect> %q1, %q2, %q3, %q4, %q5
  func.return
}
