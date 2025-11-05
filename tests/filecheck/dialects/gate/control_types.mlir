// RUN: quopt %s --verify-diagnostics | filecheck %s

%g = gate.constant #gate.h

// CHECK: result 'out' at position 0 does not verify
// CHECK-NEXT: integer 1 expected from int variable 'I', but got 0
"gate.control"(%g) : (!gate.type<1>) -> !gate.type<1>
