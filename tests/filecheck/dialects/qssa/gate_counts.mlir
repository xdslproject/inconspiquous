// RUN: quopt %s --verify-diagnostics --split-input-file | filecheck %s

%q0 = qubit.alloc

// CHECK: integer 1 expected from int variable 'I', but got 2
%q1, %q2 = "qssa.gate"(%q0) <{"gate" = #gate.cnot}> : (!qubit.bit) -> (!qubit.bit, !qubit.bit)

// -----

%q0 = qubit.alloc
%q1 = qubit.alloc

// CHECK: integer 2 expected from int variable 'I', but got 1
%q2 = "qssa.gate"(%q0, %q1) <{"gate" = #gate.cnot}> : (!qubit.bit, !qubit.bit) -> !qubit.bit

// -----

%q0 = qubit.alloc

// CHECK: integer 1 expected from int variable 'I', but got 2
%q1 = "qssa.gate"(%q0) <{"gate" = #gate.cnot}> : (!qubit.bit) -> !qubit.bit

// -----

%g = "test.op"() : () -> !gate.type<2>
%q0 = qubit.alloc

// CHECK: integer 1 expected from int variable 'I', but got 2
%q1 = "qssa.dyn_gate"(%q0, %g) : (!qubit.bit, !gate.type<2>) -> !qubit.bit
