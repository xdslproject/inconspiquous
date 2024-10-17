// RUN: quopt %s --verify-diagnostics --split-input-file | filecheck %s

%q0 = qubit.alloc

// CHECK: attributes ('!qubit.bit',) expected from range variable 'T', but got ('!qubit.bit', '!qubit.bit')
%q1, %q2 = "qssa.gate"(%q0) <{"gate" = #gate.cnot}> : (!qubit.bit) -> (!qubit.bit, !qubit.bit)

// -----

%q0 = qubit.alloc
%q1 = qubit.alloc

// CHECK: attributes ('!qubit.bit', '!qubit.bit') expected from range variable 'T', but got ('!qubit.bit',)
%q2 = "qssa.gate"(%q0, %q1) <{"gate" = #gate.cnot}> : (!qubit.bit, !qubit.bit) -> !qubit.bit

// -----

%q0 = qubit.alloc

// CHECK: Gate #gate.cnot expected 2 input qubits but got 1
%q1 = "qssa.gate"(%q0) <{"gate" = #gate.cnot}> : (!qubit.bit) -> !qubit.bit