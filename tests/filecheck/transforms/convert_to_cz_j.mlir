// RUN: quopt %s -p convert-to-cz-j

// CHECK:      func.func @cx(%q1 : !qu.bit, %q2 : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<0>> %q2
// CHECK-NEXT:   %1, %2 = qssa.apply<#gate.cz> %q1, %0
// CHECK-NEXT:   %3 = qssa.apply<#gate.j<0>> %2
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cx(%q1: !qu.bit, %q2: !qu.bit) {
    qssa.apply<#gate.cx> %q1, %q2
    func.return
}

// CHECK:      func.func @z(%q : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.apply<#gate.j<pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @z(%q: !qu.bit) {
    qssa.apply<#gate.z> %q
    func.return
}

// CHECK:      func.func @x(%q : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<pi>> %q
// CHECK-NEXT:   %1 = qssa.apply<#gate.j<0>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @x(%q: !qu.bit) {
    qssa.apply<#gate.x> %q
    func.return
}

// CHECK:      func.func @y(%q : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<pi>> %q
// CHECK-NEXT:   %1 = qssa.apply<#gate.j<pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @y(%q: !qu.bit) {
    qssa.apply<#gate.y> %q
    func.return
}

// CHECK:      func.func @h(%q : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<0>> %q
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @h(%q: !qu.bit) {
    qssa.apply<#gate.h> %q
    func.return
}

// CHECK:      func.func @s(%q : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.apply<#gate.j<0.5pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @s(%q: !qu.bit) {
    qssa.apply<#gate.s> %q
    func.return
}

// CHECK:      func.func @t(%q : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.apply<#gate.j<0.25pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @t(%q: !qu.bit) {
    qssa.apply<#gate.t> %q
    func.return
}

// CHECK:      func.func @rz(%q : !qu.bit) {
// CHECK-NEXT:   %0 = qssa.apply<#gate.j<0>> %q
// CHECK-NEXT:   %1 = qssa.apply<#gate.j<1.5pi>> %0
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @rz(%q: !qu.bit) {
    qssa.apply<#gate.rz<1.5pi>> %q
    func.return
}
