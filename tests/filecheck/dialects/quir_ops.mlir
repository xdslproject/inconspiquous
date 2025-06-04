// RUN: inconspiquous-opt %s | FileCheck %s

module {
  quir.alloc_qubit : !quir.qubit
  quir.measure %0 : !quir.qubit -> !quir.bit
}
