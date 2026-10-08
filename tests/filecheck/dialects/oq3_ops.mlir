// RUN: inconspiquous-opt %s | FileCheck %s
// CHECK: oq3.gate %{{.*}} : !oq3.qubit -> !oq3.qubit
// CHECK: oq3.measure %{{.*}} : !oq3.qubit -> !oq3.bit

module {
  oq3.gate %0 : !oq3.qubit -> !oq3.qubit
  oq3.measure %0 : !oq3.qubit -> !oq3.bit
}
