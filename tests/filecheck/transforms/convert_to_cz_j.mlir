// RUN: quopt %s -p convert-to-cz-j

// CHECK:      func.func @cx(%q1 : !qubit.bit, %q2 : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<0>> %q2
// CHECK-NEXT:   %1, %2 = qssa.gate<#gate.cz> %q1, %0
// CHECK-NEXT:   %3 = qssa.gate<#gate.j<0>> %2
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cx(%q1: !qubit.bit, %q2: !qubit.bit) {
    qssa.gate<#gate.cx> %q1, %q2
    func.return
}

// CHECK:      func.func @z(%q : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.gate<#gate.j<pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @z(%q: !qubit.bit) {
    qssa.gate<#gate.z> %q
    func.return
}

// CHECK:      func.func @x(%q : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<pi>> %q
// CHECK-NEXT:   %1 = qssa.gate<#gate.j<0>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @x(%q: !qubit.bit) {
    qssa.gate<#gate.x> %q
    func.return
}

// CHECK:      func.func @y(%q : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<pi>> %q
// CHECK-NEXT:   %1 = qssa.gate<#gate.j<pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @y(%q: !qubit.bit) {
    qssa.gate<#gate.y> %q
    func.return
}

// CHECK:      func.func @h(%q : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<0>> %q
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @h(%q: !qubit.bit) {
    qssa.gate<#gate.h> %q
    func.return
}

// CHECK:      func.func @s(%q : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.gate<#gate.j<0.5pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @s(%q: !qubit.bit) {
    qssa.gate<#gate.s> %q
    func.return
}

// CHECK:      func.func @t(%q : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.gate<#gate.j<0.25pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @t(%q: !qubit.bit) {
    qssa.gate<#gate.t> %q
    func.return
}

// CHECK:      func.func @rz(%q : !qubit.bit) {
// CHECK-NEXT:   %0 = qssa.gate<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.gate<#gate.j<1.5pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @rz(%q: !qubit.bit) {
    qssa.gate<#gate.rz<1.5pi>> %q
    func.return
}
