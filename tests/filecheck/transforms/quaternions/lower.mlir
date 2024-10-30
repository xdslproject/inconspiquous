// RUN: quopt %s -p lower-to-quaternion | filecheck %s

// CHECK:      %c0 = arith.constant 0 : i64
// CHECK-NEXT: %c1 = arith.constant 1 : i64
// CHECK-NEXT: %id = gate.quaternion<i64> %c1 + %c0 i + %c0 j + %c0 k
// CHECK-NEXT: %x = gate.quaternion<i64> %c0 + %c1 i + %c0 j + %c0 k
// CHECK-NEXT: %y = gate.quaternion<i64> %c0 + %c0 i + %c1 j + %c0 k
// CHECK-NEXT: %z = gate.quaternion<i64> %c0 + %c0 i + %c0 j + %c1 k
// CHECK-NEXT: %s = gate.quaternion<i64> %c1 + %c0 i + %c0 j + %c1 k
// CHECK-NEXT: %q0 = qubit.alloc
// CHECK-NEXT: %q1 = qssa.dyn_gate<%id> %q0 : !qubit.bit
// CHECK-NEXT: %q2 = qssa.dyn_gate<%x> %q1 : !qubit.bit
// CHECK-NEXT: %q3 = qssa.dyn_gate<%y> %q2 : !qubit.bit
// CHECK-NEXT: %q4 = qssa.dyn_gate<%z> %q3 : !qubit.bit
// CHECK-NEXT: %q5 = qssa.dyn_gate<%s> %q4 : !qubit.bit
// CHECK-NEXT: %q6 = qssa.dyn_gate<%id> %q5 : !qubit.bit
// CHECK-NEXT: %q7 = qssa.dyn_gate<%x> %q6 : !qubit.bit
// CHECK-NEXT: %q8 = qssa.dyn_gate<%y> %q7 : !qubit.bit
// CHECK-NEXT: %q9 = qssa.dyn_gate<%z> %q8 : !qubit.bit
// CHECK-NEXT: %q10 = qssa.dyn_gate<%s> %q9 : !qubit.bit

%id = gate.constant #gate.id
%x = gate.constant #gate.x
%y = gate.constant #gate.y
%z = gate.constant #gate.z
%s = gate.constant #gate.s

%q0 = qubit.alloc

%q1 = qssa.gate<#gate.id> %q0 : !qubit.bit
%q2 = qssa.gate<#gate.x> %q1 : !qubit.bit
%q3 = qssa.gate<#gate.y> %q2 : !qubit.bit
%q4 = qssa.gate<#gate.z> %q3 : !qubit.bit
%q5 = qssa.gate<#gate.s> %q4 : !qubit.bit

%q6 = qssa.dyn_gate<%id> %q5 : !qubit.bit
%q7 = qssa.dyn_gate<%x> %q6 : !qubit.bit
%q8 = qssa.dyn_gate<%y> %q7 : !qubit.bit
%q9 = qssa.dyn_gate<%z> %q8 : !qubit.bit
%q10 = qssa.dyn_gate<%s> %q9 : !qubit.bit
