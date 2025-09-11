// RUN: quopt -p compute-frame-times %s | filecheck %s

builtin.module {
  %0 = "pulse.alloc_frame"() : () -> !pulse.frame<"f1">
  %1 = "pulse.alloc_frame"() : () -> !pulse.frame<"f2">
  %2 = "pulse.const_duration"() {value = #builtin.int<1>} : () -> !pulse.duration
  %3 = "pulse.delay"(%2, %0) : (!pulse.duration, !pulse.frame<"f1">) -> !pulse.frame<"f1">
  %4, %5 = "pulse.barrier"(%3, %1) : (!pulse.frame<"f1">, !pulse.frame<"f2">) -> (!pulse.frame<"f1">, !pulse.frame<"f2">)
  %6 = "pulse.delay"(%2, %5) : (!pulse.duration, !pulse.frame<"f2">) -> !pulse.frame<"f2">
}

// CHECK:       %{{.*}} = "pulse.delay"(%2, %0) {start_time = #builtin.int<0>, end_time = #builtin.int<1>} : (!pulse.duration, !pulse.frame<"f1">) -> !pulse.frame<"f1">
// CHECK-NEXT:  %{{.*}}, %{{.*}} = "pulse.barrier"(%3, %1) {start_time = #builtin.int<1>, end_time = #builtin.int<1>} : (!pulse.frame<"f1">, !pulse.frame<"f2">) -> (!pulse.frame<"f1">, !pulse.frame<"f2">)
// CHECK-NEXT:  %{{.*}} = "pulse.delay"(%2, %5) {start_time = #builtin.int<1>, end_time = #builtin.int<2>} : (!pulse.duration, !pulse.frame<"f2">) -> !pulse.frame<"f2">
