// RUN: quopt %s -p inline-circuits | filecheck %s

// Test basic single-qubit circuit inlining
// CHECK:      func.func @basic_inline() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %circuit = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.x> %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<1>
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %q0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @basic_inline() {
  %q0 = qu.alloc
  %circuit = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %q1 = qssa.gate<#gate.x> %arg0
    qssa.return %q1
  }) : () -> !gate.type<1>
  %result = qssa.dyn_gate<%circuit> %q0
  func.return
}

// Test identity circuit (no operations, just return)
// CHECK:      func.func @identity_inline() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %circuit = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:     qssa.return %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<1>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @identity_inline() {
  %q0 = qu.alloc
  %circuit = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    qssa.return %arg0
  }) : () -> !gate.type<1>
  %result = qssa.dyn_gate<%circuit> %q0
  func.return
}

// Test complex single-qubit circuit with multiple gates
// CHECK:      func.func @complex_single_qubit() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %circuit = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.x> %{{.*}}
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.z> %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<1>
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %q0
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.z> %{{.*}}
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @complex_single_qubit() {
  %q0 = qu.alloc
  %circuit = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %q1 = qssa.gate<#gate.x> %arg0
    %q2 = qssa.gate<#gate.z> %q1
    qssa.return %q2
  }) : () -> !gate.type<1>
  %result = qssa.dyn_gate<%circuit> %q0
  func.return
}

// Test two-qubit circuit inlining
// CHECK:      func.func @two_qubit_inline() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %q1 = qu.alloc
// CHECK-NEXT:   %circuit = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit, %{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.x> %{{.*}}
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.y> %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}, %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<2>
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %q0
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.y> %q1
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @two_qubit_inline() {
  %q0 = qu.alloc
  %q1 = qu.alloc
  %circuit = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit, %arg1 : !qu.bit):
    %q2 = qssa.gate<#gate.x> %arg0
    %q3 = qssa.gate<#gate.y> %arg1
    qssa.return %q2, %q3
  }) : () -> !gate.type<2>
  %result0, %result1 = qssa.dyn_gate<%circuit> %q0, %q1
  func.return
}

// Test circuit with entangling gates
// CHECK:      func.func @entangling_circuit() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %q1 = qu.alloc
// CHECK-NEXT:   %circuit = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit, %{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.h> %{{.*}}
// CHECK-NEXT:     %{{.*}}, %{{.*}} = qssa.gate<#gate.cx> %{{.*}}, %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}, %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<2>
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.h> %q0
// CHECK-NEXT:   %{{.*}}, %{{.*}} = qssa.gate<#gate.cx> %{{.*}}, %q1
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @entangling_circuit() {
  %q0 = qu.alloc
  %q1 = qu.alloc
  %circuit = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit, %arg1 : !qu.bit):
    %q2 = qssa.gate<#gate.h> %arg0
    %q3, %q4 = qssa.gate<#gate.cx> %q2, %arg1
    qssa.return %q3, %q4
  }) : () -> !gate.type<2>
  %result0, %result1 = qssa.dyn_gate<%circuit> %q0, %q1
  func.return
}

// Test that results are properly used after inlining
// CHECK:      func.func @use_results() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %circuit = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.x> %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<1>
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %q0
// CHECK-NEXT:   %final = qssa.gate<#gate.z> %{{.*}}
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @use_results() {
  %q0 = qu.alloc
  %circuit = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %q1 = qssa.gate<#gate.x> %arg0
    qssa.return %q1
  }) : () -> !gate.type<1>
  %result = qssa.dyn_gate<%circuit> %q0
  %final = qssa.gate<#gate.z> %result
  func.return
}

// Test multiple circuit inlining in same function
// CHECK:      func.func @multiple_circuits() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %q1 = qu.alloc
// CHECK-NEXT:   %circuit1 = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.x> %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<1>
// CHECK-NEXT:   %circuit2 = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.y> %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<1>
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.x> %q0
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.y> %q1
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @multiple_circuits() {
  %q0 = qu.alloc
  %q1 = qu.alloc

  %circuit1 = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %q2 = qssa.gate<#gate.x> %arg0
    qssa.return %q2
  }) : () -> !gate.type<1>

  %circuit2 = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %q3 = qssa.gate<#gate.y> %arg0
    qssa.return %q3
  }) : () -> !gate.type<1>

  %result1 = qssa.dyn_gate<%circuit1> %q0
  %result2 = qssa.dyn_gate<%circuit2> %q1
  func.return
}

// Test that non-circuit dyn_gate operations are not affected
// CHECK:      func.func @non_circuit_dyn_gate() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %g = gate.constant #gate.h
// CHECK-NEXT:   %result = qssa.dyn_gate<%g> %q0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @non_circuit_dyn_gate() {
  %q0 = qu.alloc
  %g = gate.constant #gate.h
  %result = qssa.dyn_gate<%g> %q0
  func.return
}

// Test that the same circuit can be reused in multiple places
// CHECK:      func.func @circuit_reuse() {
// CHECK-NEXT:   %q0 = qu.alloc
// CHECK-NEXT:   %q1 = qu.alloc
// CHECK-NEXT:   %reusable_circuit = qssa.circuit() ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : !qu.bit):
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.h> %{{.*}}
// CHECK-NEXT:     %{{.*}} = qssa.gate<#gate.z> %{{.*}}
// CHECK-NEXT:     qssa.return %{{.*}}
// CHECK-NEXT:   }) : () -> !gate.type<1>
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.h> %q0
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.z> %{{.*}}
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.h> %q1
// CHECK-NEXT:   %{{.*}} = qssa.gate<#gate.z> %{{.*}}
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @circuit_reuse() {
  %q0 = qu.alloc
  %q1 = qu.alloc

  %reusable_circuit = qssa.circuit() ({
  ^bb0(%arg0 : !qu.bit):
    %q2 = qssa.gate<#gate.h> %arg0
    %q3 = qssa.gate<#gate.z> %q2
    qssa.return %q3
  }) : () -> !gate.type<1>

  // Use the same circuit on two different qubits
  %result1 = qssa.dyn_gate<%reusable_circuit> %q0
  %result2 = qssa.dyn_gate<%reusable_circuit> %q1
  func.return
}
