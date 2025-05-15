// RUN: QUOPT_ROUNDTRIP

// CHECK:      func.func @perfect_code(%q1 : !qu.bit, %q2 : !qu.bit, %q3 : !qu.bit, %q4 : !qu.bit, %q5 : !qu.bit) {
// CHECK-NEXT:   %a1 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a1
// CHECK-NEXT:   qref.gate<#gate.cx> %a1, %q1
// CHECK-NEXT:   qref.gate<#gate.cz> %a1, %q2
// CHECK-NEXT:   qref.gate<#gate.cz> %a1, %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %a1, %q4
// CHECK-NEXT:   qref.gate<#gate.h> %a1
// CHECK-NEXT:   %s1 = qref.measure %a1
// CHECK-NEXT:   %a2 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a2
// CHECK-NEXT:   qref.gate<#gate.cx> %a2, %q2
// CHECK-NEXT:   qref.gate<#gate.cz> %a2, %q3
// CHECK-NEXT:   qref.gate<#gate.cz> %a2, %q4
// CHECK-NEXT:   qref.gate<#gate.cx> %a2, %q5
// CHECK-NEXT:   qref.gate<#gate.h> %a2
// CHECK-NEXT:   %s2 = qref.measure %a2
// CHECK-NEXT:   %a3 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a3
// CHECK-NEXT:   qref.gate<#gate.cx> %a3, %q1
// CHECK-NEXT:   qref.gate<#gate.cx> %a3, %q3
// CHECK-NEXT:   qref.gate<#gate.cz> %a3, %q4
// CHECK-NEXT:   qref.gate<#gate.cz> %a3, %q5
// CHECK-NEXT:   qref.gate<#gate.h> %a3
// CHECK-NEXT:   %s3 = qref.measure %a3
// CHECK-NEXT:   %a4 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a4
// CHECK-NEXT:   qref.gate<#gate.cz> %a4, %q1
// CHECK-NEXT:   qref.gate<#gate.cx> %a4, %q2
// CHECK-NEXT:   qref.gate<#gate.cx> %a4, %q4
// CHECK-NEXT:   qref.gate<#gate.cz> %a4, %q5
// CHECK-NEXT:   qref.gate<#gate.h> %a4
// CHECK-NEXT:   %s4 = qref.measure %a4
// CHECK-NEXT:   %x = gate.constant #gate.x
// CHECK-NEXT:   %z = gate.constant #gate.z
// CHECK-NEXT:   %id = gate.constant #gate.id
// CHECK-NEXT:   %c1 = arith.constant true
// CHECK-NEXT:   %0 = arith.addi %s1, %s3 : i1
// CHECK-NEXT:   %1 = arith.ori %0, %s2 : i1
// CHECK-NEXT:   %cor1 = arith.addi %1, %c1 : i1
// CHECK-NEXT:   %cor1x = arith.andi %cor1, %s4 : i1
// CHECK-NEXT:   %cor1x_sel = arith.select %cor1x, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor1x_sel> %q1
// CHECK-NEXT:   %cor1z = arith.andi %cor1, %s1 : i1
// CHECK-NEXT:   %cor1z_sel = arith.select %cor1z, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor1z_sel> %q1
// CHECK-NEXT:   %2 = arith.addi %s2, %s4 : i1
// CHECK-NEXT:   %3 = arith.ori %2, %s3 : i1
// CHECK-NEXT:   %cor2 = arith.addi %3, %c1 : i1
// CHECK-NEXT:   %cor2x = arith.andi %cor2, %s1 : i1
// CHECK-NEXT:   %cor2x_sel = arith.select %cor2x, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor2x_sel> %q2
// CHECK-NEXT:   %cor2z = arith.andi %cor2, %s2 : i1
// CHECK-NEXT:   %cor2z_sel = arith.select %cor2z, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor2z_sel> %q2
// CHECK-NEXT:   %4 = arith.addi %s1, %s2 : i1
// CHECK-NEXT:   %5 = arith.ori %4, %s4 : i1
// CHECK-NEXT:   %cor3 = arith.addi %5, %c1 : i1
// CHECK-NEXT:   %cor3x = arith.andi %cor3, %s1 : i1
// CHECK-NEXT:   %cor3x_sel = arith.select %cor3x, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor3x_sel> %q3
// CHECK-NEXT:   %cor3z = arith.andi %cor3, %s3 : i1
// CHECK-NEXT:   %cor3z_sel = arith.select %cor3z, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor3z_sel> %q3
// CHECK-NEXT:   %6 = arith.addi %s1, %s4 : i1
// CHECK-NEXT:   %7 = arith.addi %s2, %s3 : i1
// CHECK-NEXT:   %8 = arith.ori %6, %7 : i1
// CHECK-NEXT:   %cor4 = arith.addi %8, %c1 : i1
// CHECK-NEXT:   %cor4x = arith.andi %cor4, %s2 : i1
// CHECK-NEXT:   %cor4x_sel = arith.select %cor4x, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor4x_sel> %q4
// CHECK-NEXT:   %cor4z = arith.andi %cor4, %s1 : i1
// CHECK-NEXT:   %cor4z_sel = arith.select %cor4z, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor4z_sel> %q4
// CHECK-NEXT:   %9 = arith.addi %s3, %s4 : i1
// CHECK-NEXT:   %10 = arith.ori %9, %s1 : i1
// CHECK-NEXT:   %cor5 = arith.addi %10, %c1 : i1
// CHECK-NEXT:   %cor5x = arith.andi %cor5, %s3 : i1
// CHECK-NEXT:   %cor5x_sel = arith.select %cor5x, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor5x_sel> %q5
// CHECK-NEXT:   %cor5z = arith.andi %cor5, %s2 : i1
// CHECK-NEXT:   %cor5z_sel = arith.select %cor5z, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor5z_sel> %q5
// CHECK-NEXT:   %a1_1 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a1_1
// CHECK-NEXT:   qref.gate<#gate.cx> %a1_1, %q1
// CHECK-NEXT:   qref.gate<#gate.cz> %a1_1, %q2
// CHECK-NEXT:   qref.gate<#gate.cz> %a1_1, %q3
// CHECK-NEXT:   qref.gate<#gate.cx> %a1_1, %q4
// CHECK-NEXT:   qref.gate<#gate.h> %a1_1
// CHECK-NEXT:   %s1_1 = qref.measure %a1_1
// CHECK-NEXT:   %a2_1 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a2_1
// CHECK-NEXT:   qref.gate<#gate.cx> %a2_1, %q2
// CHECK-NEXT:   qref.gate<#gate.cz> %a2_1, %q3
// CHECK-NEXT:   qref.gate<#gate.cz> %a2_1, %q4
// CHECK-NEXT:   qref.gate<#gate.cx> %a2_1, %q5
// CHECK-NEXT:   qref.gate<#gate.h> %a2_1
// CHECK-NEXT:   %s2_1 = qref.measure %a2_1
// CHECK-NEXT:   %a3_1 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a3_1
// CHECK-NEXT:   qref.gate<#gate.cx> %a3_1, %q1
// CHECK-NEXT:   qref.gate<#gate.cx> %a3_1, %q3
// CHECK-NEXT:   qref.gate<#gate.cz> %a3_1, %q4
// CHECK-NEXT:   qref.gate<#gate.cz> %a3_1, %q5
// CHECK-NEXT:   qref.gate<#gate.h> %a3_1
// CHECK-NEXT:   %s3_1 = qref.measure %a3_1
// CHECK-NEXT:   %a4_1 = qu.alloc
// CHECK-NEXT:   qref.gate<#gate.h> %a4_1
// CHECK-NEXT:   qref.gate<#gate.cz> %a4_1, %q1
// CHECK-NEXT:   qref.gate<#gate.cx> %a4_1, %q2
// CHECK-NEXT:   qref.gate<#gate.cx> %a4_1, %q4
// CHECK-NEXT:   qref.gate<#gate.cz> %a4_1, %q5
// CHECK-NEXT:   qref.gate<#gate.h> %a4_1
// CHECK-NEXT:   %s4_1 = qref.measure %a4_1
// CHECK-NEXT:   %11 = arith.addi %s1_1, %s3_1 : i1
// CHECK-NEXT:   %12 = arith.ori %11, %s2_1 : i1
// CHECK-NEXT:   %cor1_1 = arith.addi %12, %c1 : i1
// CHECK-NEXT:   %cor1x_1 = arith.andi %cor1_1, %s4_1 : i1
// CHECK-NEXT:   %cor1x_sel_1 = arith.select %cor1x_1, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor1x_sel_1> %q1
// CHECK-NEXT:   %cor1z_1 = arith.andi %cor1_1, %s1_1 : i1
// CHECK-NEXT:   %cor1z_sel_1 = arith.select %cor1z_1, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor1z_sel_1> %q1
// CHECK-NEXT:   %13 = arith.addi %s2_1, %s4_1 : i1
// CHECK-NEXT:   %14 = arith.ori %13, %s3_1 : i1
// CHECK-NEXT:   %cor2_1 = arith.addi %14, %c1 : i1
// CHECK-NEXT:   %cor2x_1 = arith.andi %cor2_1, %s1_1 : i1
// CHECK-NEXT:   %cor2x_sel_1 = arith.select %cor2x_1, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor2x_sel_1> %q2
// CHECK-NEXT:   %cor2z_1 = arith.andi %cor2_1, %s2_1 : i1
// CHECK-NEXT:   %cor2z_sel_1 = arith.select %cor2z_1, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor2z_sel_1> %q2
// CHECK-NEXT:   %15 = arith.addi %s1_1, %s2_1 : i1
// CHECK-NEXT:   %16 = arith.ori %15, %s4_1 : i1
// CHECK-NEXT:   %cor3_1 = arith.addi %16, %c1 : i1
// CHECK-NEXT:   %cor3x_1 = arith.andi %cor3_1, %s1_1 : i1
// CHECK-NEXT:   %cor3x_sel_1 = arith.select %cor3x_1, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor3x_sel_1> %q3
// CHECK-NEXT:   %cor3z_1 = arith.andi %cor3_1, %s3_1 : i1
// CHECK-NEXT:   %cor3z_sel_1 = arith.select %cor3z_1, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor3z_sel_1> %q3
// CHECK-NEXT:   %17 = arith.addi %s1_1, %s4_1 : i1
// CHECK-NEXT:   %18 = arith.addi %s2_1, %s3_1 : i1
// CHECK-NEXT:   %19 = arith.ori %17, %18 : i1
// CHECK-NEXT:   %cor4_1 = arith.addi %19, %c1 : i1
// CHECK-NEXT:   %cor4x_1 = arith.andi %cor4_1, %s2_1 : i1
// CHECK-NEXT:   %cor4x_sel_1 = arith.select %cor4x_1, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor4x_sel_1> %q4
// CHECK-NEXT:   %cor4z_1 = arith.andi %cor4_1, %s1_1 : i1
// CHECK-NEXT:   %cor4z_sel_1 = arith.select %cor4z_1, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor4z_sel_1> %q4
// CHECK-NEXT:   %20 = arith.addi %s3_1, %s4_1 : i1
// CHECK-NEXT:   %21 = arith.ori %20, %s1_1 : i1
// CHECK-NEXT:   %cor5_1 = arith.addi %21, %c1 : i1
// CHECK-NEXT:   %cor5x_1 = arith.andi %cor5_1, %s3_1 : i1
// CHECK-NEXT:   %cor5x_sel_1 = arith.select %cor5x_1, %x, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor5x_sel_1> %q5
// CHECK-NEXT:   %cor5z_1 = arith.andi %cor5_1, %s2_1 : i1
// CHECK-NEXT:   %cor5z_sel_1 = arith.select %cor5z_1, %z, %id : !gate.type<1>
// CHECK-NEXT:   qref.dyn_gate<%cor5z_sel_1> %q5
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @perfect_code(%q1: !qu.bit, %q2: !qu.bit, %q3: !qu.bit, %q4: !qu.bit, %q5: !qu.bit) {
  %a1 = qu.alloc
  qref.gate<#gate.h> %a1
  qref.gate<#gate.cx> %a1, %q1
  qref.gate<#gate.cz> %a1, %q2
  qref.gate<#gate.cz> %a1, %q3
  qref.gate<#gate.cx> %a1, %q4
  qref.gate<#gate.h> %a1
  %s1 = qref.measure %a1

  %a2 = qu.alloc
  qref.gate<#gate.h> %a2
  qref.gate<#gate.cx> %a2, %q2
  qref.gate<#gate.cz> %a2, %q3
  qref.gate<#gate.cz> %a2, %q4
  qref.gate<#gate.cx> %a2, %q5
  qref.gate<#gate.h> %a2
  %s2 = qref.measure %a2

  %a3 = qu.alloc
  qref.gate<#gate.h> %a3
  qref.gate<#gate.cx> %a3, %q1
  qref.gate<#gate.cx> %a3, %q3
  qref.gate<#gate.cz> %a3, %q4
  qref.gate<#gate.cz> %a3, %q5
  qref.gate<#gate.h> %a3
  %s3 = qref.measure %a3

  %a4 = qu.alloc
  qref.gate<#gate.h> %a4
  qref.gate<#gate.cz> %a4, %q1
  qref.gate<#gate.cx> %a4, %q2
  qref.gate<#gate.cx> %a4, %q4
  qref.gate<#gate.cz> %a4, %q5
  qref.gate<#gate.h> %a4
  %s4 = qref.measure %a4

  %x = gate.constant #gate.x
  %z = gate.constant #gate.z
  %id = gate.constant #gate.id
  %c1 = arith.constant true

  %0 = arith.addi %s1, %s3 : i1
  %1 = arith.ori %0, %s2 : i1
  %cor1 = arith.addi %1, %c1 : i1

  %cor1x = arith.andi %cor1, %s4 : i1
  %cor1x_sel = arith.select %cor1x, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor1x_sel> %q1

  %cor1z = arith.andi %cor1, %s1 : i1
  %cor1z_sel = arith.select %cor1z, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor1z_sel> %q1


  %2 = arith.addi %s2, %s4 : i1
  %3 = arith.ori %2, %s3 : i1
  %cor2 = arith.addi %3, %c1 : i1

  %cor2x = arith.andi %cor2, %s1 : i1
  %cor2x_sel = arith.select %cor2x, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor2x_sel> %q2

  %cor2z = arith.andi %cor2, %s2 : i1
  %cor2z_sel = arith.select %cor2z, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor2z_sel> %q2


  %4 = arith.addi %s1, %s2 : i1
  %5 = arith.ori %4, %s4 : i1
  %cor3 = arith.addi %5, %c1 : i1

  %cor3x = arith.andi %cor3, %s1 : i1
  %cor3x_sel = arith.select %cor3x, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor3x_sel> %q3

  %cor3z = arith.andi %cor3, %s3 : i1
  %cor3z_sel = arith.select %cor3z, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor3z_sel> %q3


  %6 = arith.addi %s1, %s4 : i1
  %7 = arith.addi %s2, %s3 : i1
  %8 = arith.ori %6, %7 : i1
  %cor4 = arith.addi %8, %c1 : i1

  %cor4x = arith.andi %cor4, %s2 : i1
  %cor4x_sel = arith.select %cor4x, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor4x_sel> %q4

  %cor4z = arith.andi %cor4, %s1 : i1
  %cor4z_sel = arith.select %cor4z, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor4z_sel> %q4


  %9 = arith.addi %s3, %s4 : i1
  %10 = arith.ori %9, %s1 : i1
  %cor5 = arith.addi %10, %c1 : i1

  %cor5x = arith.andi %cor5, %s3 : i1
  %cor5x_sel = arith.select %cor5x, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor5x_sel> %q5

  %cor5z = arith.andi %cor5, %s2 : i1
  %cor5z_sel = arith.select %cor5z, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor5z_sel> %q5



  %a1_1 = qu.alloc
  qref.gate<#gate.h> %a1_1
  qref.gate<#gate.cx> %a1_1, %q1
  qref.gate<#gate.cz> %a1_1, %q2
  qref.gate<#gate.cz> %a1_1, %q3
  qref.gate<#gate.cx> %a1_1, %q4
  qref.gate<#gate.h> %a1_1
  %s1_1 = qref.measure %a1_1

  %a2_1 = qu.alloc
  qref.gate<#gate.h> %a2_1
  qref.gate<#gate.cx> %a2_1, %q2
  qref.gate<#gate.cz> %a2_1, %q3
  qref.gate<#gate.cz> %a2_1, %q4
  qref.gate<#gate.cx> %a2_1, %q5
  qref.gate<#gate.h> %a2_1
  %s2_1 = qref.measure %a2_1

  %a3_1 = qu.alloc
  qref.gate<#gate.h> %a3_1
  qref.gate<#gate.cx> %a3_1, %q1
  qref.gate<#gate.cx> %a3_1, %q3
  qref.gate<#gate.cz> %a3_1, %q4
  qref.gate<#gate.cz> %a3_1, %q5
  qref.gate<#gate.h> %a3_1
  %s3_1 = qref.measure %a3_1

  %a4_1 = qu.alloc
  qref.gate<#gate.h> %a4_1
  qref.gate<#gate.cz> %a4_1, %q1
  qref.gate<#gate.cx> %a4_1, %q2
  qref.gate<#gate.cx> %a4_1, %q4
  qref.gate<#gate.cz> %a4_1, %q5
  qref.gate<#gate.h> %a4_1
  %s4_1 = qref.measure %a4_1

  %11 = arith.addi %s1_1, %s3_1 : i1
  %12 = arith.ori %11, %s2_1 : i1
  %cor1_1 = arith.addi %12, %c1 : i1

  %cor1x_1 = arith.andi %cor1_1, %s4_1 : i1
  %cor1x_sel_1 = arith.select %cor1x_1, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor1x_sel_1> %q1

  %cor1z_1 = arith.andi %cor1_1, %s1_1 : i1
  %cor1z_sel_1 = arith.select %cor1z_1, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor1z_sel_1> %q1


  %13 = arith.addi %s2_1, %s4_1 : i1
  %14 = arith.ori %13, %s3_1 : i1
  %cor2_1 = arith.addi %14, %c1 : i1

  %cor2x_1 = arith.andi %cor2_1, %s1_1 : i1
  %cor2x_sel_1 = arith.select %cor2x_1, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor2x_sel_1> %q2

  %cor2z_1 = arith.andi %cor2_1, %s2_1 : i1
  %cor2z_sel_1 = arith.select %cor2z_1, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor2z_sel_1> %q2


  %15 = arith.addi %s1_1, %s2_1 : i1
  %16 = arith.ori %15, %s4_1 : i1
  %cor3_1 = arith.addi %16, %c1 : i1

  %cor3x_1 = arith.andi %cor3_1, %s1_1 : i1
  %cor3x_sel_1 = arith.select %cor3x_1, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor3x_sel_1> %q3

  %cor3z_1 = arith.andi %cor3_1, %s3_1 : i1
  %cor3z_sel_1 = arith.select %cor3z_1, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor3z_sel_1> %q3


  %17 = arith.addi %s1_1, %s4_1 : i1
  %18 = arith.addi %s2_1, %s3_1 : i1
  %19 = arith.ori %17, %18 : i1
  %cor4_1 = arith.addi %19, %c1 : i1

  %cor4x_1 = arith.andi %cor4_1, %s2_1 : i1
  %cor4x_sel_1 = arith.select %cor4x_1, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor4x_sel_1> %q4

  %cor4z_1 = arith.andi %cor4_1, %s1_1 : i1
  %cor4z_sel_1 = arith.select %cor4z_1, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor4z_sel_1> %q4


  %20 = arith.addi %s3_1, %s4_1 : i1
  %21 = arith.ori %20, %s1_1 : i1
  %cor5_1 = arith.addi %21, %c1 : i1

  %cor5x_1 = arith.andi %cor5_1, %s3_1 : i1
  %cor5x_sel_1 = arith.select %cor5x_1, %x, %id : !gate.type<1>
  qref.dyn_gate<%cor5x_sel_1> %q5

  %cor5z_1 = arith.andi %cor5_1, %s2_1 : i1
  %cor5z_sel_1 = arith.select %cor5z_1, %z, %id : !gate.type<1>
  qref.dyn_gate<%cor5z_sel_1> %q5


  func.return
}
