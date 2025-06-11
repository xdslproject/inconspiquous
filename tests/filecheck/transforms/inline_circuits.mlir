// RUN: quopt %s -p inline-circuits | filecheck %s

// Test basic circuit inlining with single gate
builtin.module {
  %q0 = qu.alloc

  // Circuit that applies X gate
  %circuit1 = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %0 = qssa.gate<#gate.x> %arg0
    qssa.return %0
  }) : () -> !gate.type<1>

  // Should be inlined
  // CHECK:      %q0 = qu.alloc
  // CHECK-NEXT: %circuit1 = qssa.circuit() ({
  // CHECK-NEXT: ^{{.*}}(%arg0 : !qu.bit):
  // CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %arg0
  // CHECK-NEXT:   qssa.return %{{.*}}
  // CHECK-NEXT: }) : () -> !gate.type<1>
  // CHECK-NEXT: %{{.*}} = qssa.gate<#gate.x> %q0
  // CHECK-NOT:  qssa.dyn_gate
  %q1 = qssa.dyn_gate<%circuit1> %q0
}

// Test complex circuit inlining with multiple gates
builtin.module {
  %q0 = qu.alloc

  // Circuit that applies X then Z gates
  %circuit2 = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %0 = qssa.gate<#gate.x> %arg0
    %1 = qssa.gate<#gate.z> %0
    qssa.return %1
  }) : () -> !gate.type<1>

  // Should be inlined
  // CHECK:      %q0 = qu.alloc
  // CHECK-NEXT: %circuit2 = qssa.circuit() ({
  // CHECK-NEXT: ^{{.*}}(%arg0 : !qu.bit):
  // CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %arg0
  // CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.z> %{{.*}}
  // CHECK-NEXT:   qssa.return %{{.*}}
  // CHECK-NEXT: }) : () -> !gate.type<1>
  // CHECK-NEXT: %{{.*}} = qssa.gate<#gate.x> %q0
  // CHECK-NEXT: %{{.*}} = qssa.gate<#gate.z> %{{.*}}
  // CHECK-NOT:  qssa.dyn_gate
  %q1 = qssa.dyn_gate<%circuit2> %q0
}

// Test two-qubit circuit inlining
builtin.module {
  %q0 = qu.alloc
  %q1 = qu.alloc

  // Circuit that applies X to first qubit, Y to second qubit
  %circuit3 = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit, %arg1 : !qu.bit):
    %0 = qssa.gate<#gate.x> %arg0
    %1 = qssa.gate<#gate.y> %arg1
    qssa.return %0, %1
  }) : () -> !gate.type<2>

  // Should be inlined
  // CHECK:      %q0 = qu.alloc
  // CHECK-NEXT: %q1 = qu.alloc
  // CHECK-NEXT: %circuit3 = qssa.circuit() ({
  // CHECK-NEXT: ^{{.*}}(%arg0 : !qu.bit, %arg1 : !qu.bit):
  // CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %arg0
  // CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.y> %arg1
  // CHECK-NEXT:   qssa.return %{{.*}}, %{{.*}}
  // CHECK-NEXT: }) : () -> !gate.type<2>
  // CHECK-NEXT: %{{.*}} = qssa.gate<#gate.x> %q0
  // CHECK-NEXT: %{{.*}} = qssa.gate<#gate.y> %q1
  // CHECK-NOT:  qssa.dyn_gate
  %q2, %q3 = qssa.dyn_gate<%circuit3> %q0, %q1
}

// Test empty circuit (identity)
builtin.module {
  %q0 = qu.alloc

  // Circuit that just returns its input
  %circuit4 = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    qssa.return %arg0
  }) : () -> !gate.type<1>

  // Should be inlined (no operations added)
  // CHECK:      %q0 = qu.alloc
  // CHECK-NEXT: %circuit4 = qssa.circuit() ({
  // CHECK-NEXT: ^{{.*}}(%arg0 : !qu.bit):
  // CHECK-NEXT:   qssa.return %arg0
  // CHECK-NEXT: }) : () -> !gate.type<1>
  // CHECK-NOT:  qssa.dyn_gate
  // CHECK-NOT:  qssa.gate
  %q1 = qssa.dyn_gate<%circuit4> %q0
}

// Test circuit with entangling gate
builtin.module {
  %q0 = qu.alloc
  %q1 = qu.alloc

  // Circuit that applies CNOT gate
  %circuit5 = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit, %arg1 : !qu.bit):
    %0, %1 = qssa.gate<#gate.cx> %arg0, %arg1
    qssa.return %0, %1
  }) : () -> !gate.type<2>

  // Should be inlined
  // CHECK:      %q0 = qu.alloc
  // CHECK-NEXT: %q1 = qu.alloc
  // CHECK-NEXT: %circuit5 = qssa.circuit() ({
  // CHECK-NEXT: ^{{.*}}(%arg0 : !qu.bit, %arg1 : !qu.bit):
  // CHECK-NEXT:   %{{.*}}, %{{.*}} = qssa.gate<#gate.cx> %arg0, %arg1
  // CHECK-NEXT:   qssa.return %{{.*}}, %{{.*}}
  // CHECK-NEXT: }) : () -> !gate.type<2>
  // CHECK-NEXT: %{{.*}}, %{{.*}} = qssa.gate<#gate.cx> %q0, %q1
  // CHECK-NOT:  qssa.dyn_gate
  %q2, %q3 = qssa.dyn_gate<%circuit5> %q0, %q1
}
