// RUN: QUOPT_ROUNDTRIP

// CHECK:      %q0 = qu.alloc
// CHECK-NEXT: %q1 = qu.alloc
// CHECK-NEXT: %q2 = qu.alloc
// CHECK-NEXT: %q3 = qu.alloc
// CHECK-NEXT: %q4 = qu.alloc
// CHECK-NEXT: %q5 = qu.alloc
// CHECK-NEXT: %q6 = qu.alloc
// CHECK-NEXT: %q7 = qu.alloc
// CHECK-NEXT: %q8 = qu.alloc
// CHECK-NEXT: %q9 = qu.alloc
// CHECK-NEXT: qref.gate<#gate.h> %q0
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q1
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q2
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q3
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q4
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q5
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q6
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q7
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q8
// CHECK-NEXT: qref.gate<#gate.cx> %q0, %q9

%q0 = qu.alloc
%q1 = qu.alloc
%q2 = qu.alloc
%q3 = qu.alloc
%q4 = qu.alloc
%q5 = qu.alloc
%q6 = qu.alloc
%q7 = qu.alloc
%q8 = qu.alloc
%q9 = qu.alloc

qref.gate<#gate.h> %q0
qref.gate<#gate.cx> %q0, %q1
qref.gate<#gate.cx> %q0, %q2
qref.gate<#gate.cx> %q0, %q3
qref.gate<#gate.cx> %q0, %q4
qref.gate<#gate.cx> %q0, %q5
qref.gate<#gate.cx> %q0, %q6
qref.gate<#gate.cx> %q0, %q7
qref.gate<#gate.cx> %q0, %q8
qref.gate<#gate.cx> %q0, %q9
