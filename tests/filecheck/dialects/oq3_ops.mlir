// RUN: inconspiquous-opt %s | FileCheck %s

module {
  oq3.gate %0 : !oq3.qubit -> !oq3.qubit
  oq3.measure %0 : !oq3.qubit -> !oq3.bit
}
