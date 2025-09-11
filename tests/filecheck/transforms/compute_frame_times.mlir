// RUN: quopt -p compute-frame-times %s | filecheck %s

builtin.module {
  %0 = "pulse.alloc_frame"() : () -> !pulse.frame<"f">
  %1 = "pulse.const_duration"() {value = #builtin.int<0>} : () -> !pulse.duration
  %2 = "pulse.const_duration"() {value = #builtin.int<1>} : () -> !pulse.duration
  %3 = "pulse.delay"(%2, %0) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
  %4 = "pulse.delay"(%2, %3) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
  %5 = "pulse.delay"(%1, %4) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
}

// CHECK:       %{{.*}} = "pulse.delay"(%2, %0) {start_time = #builtin.int<0>, end_time = #builtin.int<1>} : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
// CHECK-NEXT:  %{{.*}}= "pulse.delay"(%2, %3) {start_time = #builtin.int<1>, end_time = #builtin.int<2>} : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
// CHECK-NEXT:  %{{.*}} = "pulse.delay"(%1, %4) {start_time = #builtin.int<2>, end_time = #builtin.int<2>} : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
