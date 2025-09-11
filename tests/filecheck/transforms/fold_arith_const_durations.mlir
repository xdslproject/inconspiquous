// RUN: quopt -p fold-arith-const-durations,dce %s | filecheck %s

builtin.module {
  %0 = "pulse.alloc_frame"() : () -> !pulse.frame<"f">
  %1 = arith.constant 0 : i32
  %2 = "pulse.duration_from_int"(%1) : (i32) -> !pulse.duration
  %3 = "pulse.const_duration"() {value = #builtin.int<1>} : () -> !pulse.duration
  %4 = "pulse.delay"(%3, %0) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
  %5 = "pulse.delay"(%3, %4) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
  %6 = "pulse.delay"(%2, %5) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
}

// CHECK:       %0 = "pulse.alloc_frame"() : () -> !pulse.frame<"f">
// CHECK-NEXT:  %1 = "pulse.const_duration"() {value = #builtin.int<0>} : () -> !pulse.duration
// CHECK-NEXT:  %2 = "pulse.const_duration"() {value = #builtin.int<1>} : () -> !pulse.duration
// CHECK-NEXT:  %3 = "pulse.delay"(%2, %0) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
// CHECK-NEXT:  %4 = "pulse.delay"(%2, %3) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
// CHECK-NEXT:  %5 = "pulse.delay"(%1, %4) : (!pulse.duration, !pulse.frame<"f">) -> !pulse.frame<"f">
