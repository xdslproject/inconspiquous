// RUN: inconspiquous-opt %s | FileCheck %s

module {
  oq3.reset %0 : !oq3.qubit -> !oq3.qubit
  oq3.barrier %0 : !oq3.qubit -> !oq3.qubit
  oq3.cond_gate %0, %1 : !oq3.bit, !oq3.qubit -> !oq3.qubit
}
