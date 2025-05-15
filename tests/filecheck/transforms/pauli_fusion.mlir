// RUN: quopt %s -p pauli-fusion | filecheck %s


// CHECK:      func.func @XX(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   func.return %q : !qu.bit
func.func @XX(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.x> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @XY(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.z> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
func.func @XY(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @XZ(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.y> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
func.func @XZ(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.z> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @YX(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.z> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
func.func @YX(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.y> %q
  %q_2 = qssa.gate<#gate.x> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @YY(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   func.return %q : !qu.bit
func.func @YY(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.y> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @YZ(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
func.func @YZ(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.y> %q
  %q_2 = qssa.gate<#gate.z> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @ZX(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.y> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
func.func @ZX(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.z> %q
  %q_2 = qssa.gate<#gate.x> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @ZY(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   %q_1 = qssa.gate<#gate.x> %q
// CHECK-NEXT:   func.return %q_1 : !qu.bit
func.func @ZY(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.z> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @ZZ(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   func.return %q : !qu.bit
func.func @ZZ(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.z> %q
  %q_2 = qssa.gate<#gate.z> %q_1
  func.return %q_2 : !qu.bit
}

// CHECK:      func.func @XYZ(%q : !qu.bit) -> !qu.bit {
// CHECK-NEXT:   func.return %q : !qu.bit
func.func @XYZ(%q: !qu.bit) -> !qu.bit {
  %q_1 = qssa.gate<#gate.x> %q
  %q_2 = qssa.gate<#gate.y> %q_1
  %q_3 = qssa.gate<#gate.z> %q_2
  func.return %q_3 : !qu.bit
}
