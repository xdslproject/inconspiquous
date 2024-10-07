// RUN: quopt %s --verify-diagnostics --split-input-file | filecheck %s

%q0 = qssa.alloc

// CHECK: expected 2 qubit inputs, but got 1 instead.
%q1, %q2 = "qssa.gate"(%q0) <{"gate" = #quantum.cnot}> : (!qssa.qubit) -> (!qssa.qubit, !qssa.qubit)

// -----

%q0 = qssa.alloc
%q1 = qssa.alloc

// CHECK: expected 2 qubit outputs, but got 1 instead.
%q2 = "qssa.gate"(%q0, %q1) <{"gate" = #quantum.cnot}> : (!qssa.qubit, !qssa.qubit) -> !qssa.qubit
