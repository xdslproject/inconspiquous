// RUN: inconspiquous-opt %s | FileCheck %s
// CHECK: quir.alloc_qubit : !quir.qubit
// CHECK: quir.measure %0 : !quir.qubit -> !quir.bit

module {
  quir.alloc_qubit : !quir.qubit
  quir.measure %0 : !quir.qubit -> !quir.bit
}
