// RUN: quopt %s --verify-diagnostics --split-input-file | filecheck %s

%q0 = qu.alloc

// CHECK: integer 1 expected from int variable 'I', but got 2
%q1, %q2 = "qssa.gate"(%q0) <{"gate" = #gate.cx}> : (!qu.bit) -> (!qu.bit, !qu.bit)

// -----

%q0 = qu.alloc
%q1 = qu.alloc

// CHECK: integer 2 expected from int variable 'I', but got 1
%q2 = "qssa.gate"(%q0, %q1) <{"gate" = #gate.cx}> : (!qu.bit, !qu.bit) -> !qu.bit

// -----

%q0 = qu.alloc

// CHECK: integer 1 expected from int variable 'I', but got 2
%q1 = "qssa.gate"(%q0) <{"gate" = #gate.cx}> : (!qu.bit) -> !qu.bit

// -----

%g = "test.op"() : () -> !gate.type<2>
%q0 = qu.alloc

// CHECK: integer 1 expected from int variable 'I', but got 2
%q1 = "qssa.dyn_gate"(%q0, %g) : (!qu.bit, !gate.type<2>) -> !qu.bit
