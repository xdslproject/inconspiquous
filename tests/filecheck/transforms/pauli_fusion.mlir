// RUN: quopt %s -p pauli-fusion | filecheck %s


// CHECK:      func.func @XX(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   func.return %q : !qubit.bit
func.func @XX(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.x> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @XY(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.z> %q
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
func.func @XY(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @XZ(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.y> %q
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
func.func @XZ(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.z> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @YX(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.z> %q
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
func.func @YX(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.y> %q
  %q_2 = qssa.gate<#gate.x> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @YY(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   func.return %q : !qubit.bit
func.func @YY(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.y> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @YZ(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
func.func @YZ(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.y> %q
  %q_2 = qssa.gate<#gate.z> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @ZX(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.y> %q
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
func.func @ZX(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.z> %q
  %q_2 = qssa.gate<#gate.x> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @ZY(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:   func.return %q_1 : !qubit.bit
func.func @ZY(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.z> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @ZZ(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   func.return %q : !qubit.bit
func.func @ZZ(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.z> %q
  %q_2 = qssa.gate<#gate.z> %q_1
  func.return %q_2 : !qubit.bit
}

// CHECK:      func.func @XYZ(%q : !qubit.bit) -> !qubit.bit {
// CHECK-NEXT:   func.return %q : !qubit.bit
func.func @XYZ(%q: !qubit.bit) -> !qubit.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  %q_3 = qssa.gate<#gate.z> %q_2
  func.return %q_3 : !qubit.bit
}
