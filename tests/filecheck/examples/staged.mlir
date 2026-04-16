// RUN: QUOPT_ROUNDTRIP
// RUN: quopt %s -p func-inline,canonicalize | filecheck %s --check-prefix CHECK-INLINE

// CHECK:      func.func @rec() -> i32 {
// CHECK-NEXT:   %c0 = arith.constant 0 : i32
// CHECK-NEXT:   %c1 = arith.constant 1 : i32
// CHECK-NEXT:   cf.br ^bb0(%c0 : i32)
// CHECK-NEXT: ^bb0(%i: i32):
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   staged.gate<#gate.h> %q0
// CHECK-NEXT:   %0 = staged.measure %q0
// CHECK-NEXT:   staged.step ^bb1(%0 : !staged.later<i1>)
// CHECK-NEXT: ^bb1(%m: i1):
// CHECK-NEXT:   %j = arith.addi %i, %c1 : i32
// CHECK-NEXT:   cf.cond_br %m, ^bb0(%j : i32), ^bb2
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   func.return %i : i32
// CHECK-NEXT: }
func.func @rec() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  cf.br ^bb0(%c0 : i32)
^bb0(%i: i32):
  %q0 = qu.alloc
  staged.gate<#gate.h> %q0
  %1 = staged.measure %q0
  staged.step ^bb1(%1 : !staged.later<i1>)
^bb1(%m: i1):
  %j = arith.addi %i, %c1 : i32
  cf.cond_br %m, ^bb0(%j : i32), ^bb2
^bb2:
  func.return %i : i32
}

// CHECK:      func.func @vqe_ansatz(%a1: !angle.type, %a2: !angle.type, %a3: !angle.type, %basis: i1) -> (i1, i1) {
// CHECK-NEXT:   %q1 = qu.alloc
// CHECK-NEXT:   %q2 = qu.alloc
// CHECK-NEXT:   %g1 = gate.dyn_rx<%a1>
// CHECK-NEXT:   %g2 = gate.dyn_rx<%a2>
// CHECK-NEXT:   %g3 = gate.dyn_rz<%a3>
// CHECK-NEXT:   %h = gate.constant #gate.h
// CHECK-NEXT:   %id = gate.constant #gate.id<1>
// CHECK-NEXT:   %g4 = arith.select %basis, %h, %id : !gate.type<1>
// CHECK-NEXT:   staged.dyn_gate<%g1> %q1
// CHECK-NEXT:   staged.dyn_gate<%g2> %q2
// CHECK-NEXT:   staged.gate<#gate.cx> %q1, %q2
// CHECK-NEXT:   staged.dyn_gate<%g2> %q1
// CHECK-NEXT:   staged.dyn_gate<%g4> %q1
// CHECK-NEXT:   staged.dyn_gate<%g4> %q2
// CHECK-NEXT:   %m1 = staged.measure %q1
// CHECK-NEXT:   %m2 = staged.measure %q2
// CHECK-NEXT:   staged.step ^bb0(%m1, %m2 : !staged.later<i1>, !staged.later<i1>)
// CHECK-NEXT: ^bb0(%0: i1, %1: i1):
// CHECK-NEXT:   func.return %0, %1 : i1, i1
// CHECK-NEXT: }
func.func @vqe_ansatz(
  %a1: !angle.type,
  %a2: !angle.type,
  %a3: !angle.type,
  %basis: i1
) -> (i1, i1) {
  %q1 = qu.alloc
  %q2 = qu.alloc
  %g1 = gate.dyn_rx<%a1>
  %g2 = gate.dyn_rx<%a2>
  %g3 = gate.dyn_rz<%a3>
  %h = gate.constant #gate.h
  %id = gate.constant #gate.id<1>
  %g4 = arith.select %basis, %h, %id : !gate.type<1>

  staged.dyn_gate<%g1> %q1
  staged.dyn_gate<%g2> %q2
  staged.gate<#gate.cx> %q1, %q2
  staged.dyn_gate<%g2> %q1

  staged.dyn_gate<%g4> %q1
  staged.dyn_gate<%g4> %q2

  %m1 = staged.measure %q1
  %m2 = staged.measure %q2
  staged.step ^bb0(%m1, %m2 : !staged.later<i1>, !staged.later<i1>)
^bb0(%0: i1, %1: i1):
  func.return %0, %1 : i1, i1
}

// CHECK:      func.func @vqe_expectation(%a1: !angle.type, %a2: !angle.type, %a3: !angle.type, %shots: i32) -> f32 {
// CHECK-NEXT:   %c0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %ci0 = arith.constant 0 : i32
// CHECK-NEXT:   %cFalse = arith.constant false
// CHECK-NEXT:   %cTrue = arith.constant true
// CHECK-NEXT:   %c1 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %ci1 = arith.constant 1 : i32
// CHECK-NEXT:   %cm1 = arith.constant -1.000000e+00 : f32
// CHECK-NEXT:   cf.br ^bb0(%c0, %c0, %shots : f32, f32, i32)
// CHECK-NEXT: ^bb0(%z: f32, %x: f32, %s: i32):
// CHECK-NEXT:   %m1, %m2 = func.call @vqe_ansatz(%a1, %a2, %a3, %cFalse) : (!angle.type, !angle.type, !angle.type, i1) -> (i1, i1)
// CHECK-NEXT:   %m3, %m4 = func.call @vqe_ansatz(%a1, %a2, %a3, %cTrue) : (!angle.type, !angle.type, !angle.type, i1) -> (i1, i1)
// CHECK-NEXT:   %z_1 = arith.xori %m1, %m2 : i1
// CHECK-NEXT:   %z_2 = arith.select %z_1, %cm1, %c1 : f32
// CHECK-NEXT:   %z_3 = arith.addf %z, %z_2 : f32
// CHECK-NEXT:   %x_1 = arith.xori %m3, %m4 : i1
// CHECK-NEXT:   %x_2 = arith.select %x_1, %cm1, %c1 : f32
// CHECK-NEXT:   %x_3 = arith.addf %x, %x_2 : f32
// CHECK-NEXT:   %s_1 = arith.addi %s, %ci1 : i32
// CHECK-NEXT:   %b = arith.cmpi eq, %shots, %s_1 : i32
// CHECK-NEXT:   cf.cond_br %b, ^bb1(%z_3, %x_3 : f32, f32), ^bb0(%z_3, %x_3, %s_1 : f32, f32, i32)
// CHECK-NEXT: ^bb1(%z_exp: f32, %x_exp: f32):
// CHECK-NEXT:   %sf = arith.uitofp %shots : i32 to f32
// CHECK-NEXT:   %z_exp_1 = arith.divf %z_exp, %sf : f32
// CHECK-NEXT:   %x_exp_1 = arith.divf %x_exp, %sf : f32
// CHECK-NEXT:   %exp = arith.addf %z_exp_1, %x_exp_1 : f32
// CHECK-NEXT:   func.return %exp : f32
// CHECK-NEXT: }
// CHECK-INLINE:      func.func @vqe_expectation(%a1: !angle.type, %a2: !angle.type, %a3: !angle.type, %shots: i32) -> f32 {
// CHECK-INLINE-NEXT:   %c0 = arith.constant 0.000000e+00 : f32
// CHECK-INLINE-NEXT:   %c1 = arith.constant 1.000000e+00 : f32
// CHECK-INLINE-NEXT:   %ci1 = arith.constant 1 : i32
// CHECK-INLINE-NEXT:   %cm1 = arith.constant -1.000000e+00 : f32
// CHECK-INLINE-NEXT:   cf.br ^bb0(%c0, %c0, %shots : f32, f32, i32)
// CHECK-INLINE-NEXT: ^bb0(%z: f32, %x: f32, %s: i32):
// CHECK-INLINE-NEXT:   %q1 = qu.alloc
// CHECK-INLINE-NEXT:   %q2 = qu.alloc
// CHECK-INLINE-NEXT:   %g1 = gate.dyn_rx<%a1>
// CHECK-INLINE-NEXT:   %g2 = gate.dyn_rx<%a2>
// CHECK-INLINE-NEXT:   staged.dyn_gate<%g1> %q1
// CHECK-INLINE-NEXT:   staged.dyn_gate<%g2> %q2
// CHECK-INLINE-NEXT:   staged.gate<#gate.cx> %q1, %q2
// CHECK-INLINE-NEXT:   staged.dyn_gate<%g2> %q1
// CHECK-INLINE-NEXT:   %m1 = staged.measure %q1
// CHECK-INLINE-NEXT:   %m2 = staged.measure %q2
// CHECK-INLINE-NEXT:   staged.step ^bb1(%m1, %m2 : !staged.later<i1>, !staged.later<i1>)
// CHECK-INLINE-NEXT: ^bb1(%m1_1: i1, %m2_1: i1):
// CHECK-INLINE-NEXT:   %q1_1 = qu.alloc
// CHECK-INLINE-NEXT:   %q2_1 = qu.alloc
// CHECK-INLINE-NEXT:   %g1_1 = gate.dyn_rx<%a1>
// CHECK-INLINE-NEXT:   %g2_1 = gate.dyn_rx<%a2>
// CHECK-INLINE-NEXT:   staged.dyn_gate<%g1_1> %q1_1
// CHECK-INLINE-NEXT:   staged.dyn_gate<%g2_1> %q2_1
// CHECK-INLINE-NEXT:   staged.gate<#gate.cx> %q1_1, %q2_1
// CHECK-INLINE-NEXT:   staged.dyn_gate<%g2_1> %q1_1
// CHECK-INLINE-NEXT:   staged.gate<#gate.h> %q1_1
// CHECK-INLINE-NEXT:   staged.gate<#gate.h> %q2_1
// CHECK-INLINE-NEXT:   %m1_2 = staged.measure %q1_1
// CHECK-INLINE-NEXT:   %m2_2 = staged.measure %q2_1
// CHECK-INLINE-NEXT:   staged.step ^bb2(%m1_2, %m2_2 : !staged.later<i1>, !staged.later<i1>)
// CHECK-INLINE-NEXT: ^bb2(%m3: i1, %m4: i1):
// CHECK-INLINE-NEXT:   %z_1 = arith.xori %m1_1, %m2_1 : i1
// CHECK-INLINE-NEXT:   %z_2 = arith.select %z_1, %cm1, %c1 : f32
// CHECK-INLINE-NEXT:   %z_3 = arith.addf %z, %z_2 : f32
// CHECK-INLINE-NEXT:   %x_1 = arith.xori %m3, %m4 : i1
// CHECK-INLINE-NEXT:   %x_2 = arith.select %x_1, %cm1, %c1 : f32
// CHECK-INLINE-NEXT:   %x_3 = arith.addf %x, %x_2 : f32
// CHECK-INLINE-NEXT:   %s_1 = arith.addi %s, %ci1 : i32
// CHECK-INLINE-NEXT:   %b = arith.cmpi eq, %shots, %s_1 : i32
// CHECK-INLINE-NEXT:   cf.cond_br %b, ^bb3(%z_3, %x_3 : f32, f32), ^bb0(%z_3, %x_3, %s_1 : f32, f32, i32)
// CHECK-INLINE-NEXT: ^bb3(%z_exp: f32, %x_exp: f32):
// CHECK-INLINE-NEXT:   %sf = arith.uitofp %shots : i32 to f32
// CHECK-INLINE-NEXT:   %z_exp_1 = arith.divf %z_exp, %sf : f32
// CHECK-INLINE-NEXT:   %x_exp_1 = arith.divf %x_exp, %sf : f32
// CHECK-INLINE-NEXT:   %exp = arith.addf %z_exp_1, %x_exp_1 : f32
// CHECK-INLINE-NEXT:   func.return %exp : f32
// CHECK-INLINE-NEXT: }
func.func @vqe_expectation(%a1: !angle.type, %a2: !angle.type, %a3: !angle.type, %shots: i32) -> f32 {
  %c0 = arith.constant 0.0 : f32
  %ci0 = arith.constant 0 : i32
  %cFalse = arith.constant false
  %cTrue = arith.constant true
  %c1 = arith.constant 1.0 : f32
  %ci1 = arith.constant 1 : i32
  %cm1 = arith.constant -1.0 : f32
  cf.br ^bb0(%c0, %c0, %shots : f32, f32, i32)
^bb0(%z: f32, %x: f32, %s: i32):
  %m1, %m2 = func.call @vqe_ansatz(%a1, %a2, %a3, %cFalse) : (!angle.type, !angle.type, !angle.type, i1) -> (i1, i1)
  %m3, %m4 = func.call @vqe_ansatz(%a1, %a2, %a3, %cTrue) : (!angle.type, !angle.type, !angle.type, i1) -> (i1, i1)
  %z_1 = arith.xori %m1, %m2 : i1
  %z_2 = arith.select %z_1, %cm1, %c1 : f32
  %z_3 = arith.addf %z, %z_2 : f32
  %x_1 = arith.xori %m3, %m4 : i1
  %x_2 = arith.select %x_1, %cm1, %c1 : f32
  %x_3 = arith.addf %x, %x_2 : f32
  %s_1 = arith.addi %s, %ci1 : i32
  %b = arith.cmpi eq, %shots, %s_1 : i32
  cf.cond_br %b, ^bb1(%z_3, %x_3 : f32, f32), ^bb0(%z_3, %x_3, %s_1 : f32, f32, i32)
^bb1(%z_exp: f32, %x_exp: f32):
  %sf = arith.uitofp %shots : i32 to f32
  %z_exp_1 = arith.divf %z_exp, %sf : f32
  %x_exp_1 = arith.divf %x_exp, %sf : f32
  %exp = arith.addf %z_exp_1, %x_exp_1 : f32
  func.return %exp : f32
}
