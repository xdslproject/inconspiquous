// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: "test.op"() {angle = #angle.attr<0>} : () -> ()
"test.op"() {angle = #angle.attr<0>} : () -> ()

// CHECK-NEXT: "test.op"() {angle = #angle.attr<pi>} : () -> ()
"test.op"() {angle = #angle.attr<pi>} : () -> ()

// CHECK-NEXT: "test.op"() {angle = #angle.attr<0>} : () -> ()
"test.op"() {angle = #angle.attr<2pi>} : () -> ()

// CHECK-NEXT: "test.op"() {angle = #angle.attr<0.5pi>} : () -> ()
"test.op"() {angle = #angle.attr<0.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {angle = #angle.attr<1.5pi>} : () -> ()
"test.op"() {angle = #angle.attr<1.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {angle = #angle.attr<0.5pi>} : () -> ()
"test.op"() {angle = #angle.attr<2.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {angle = #angle.attr<1.5pi>} : () -> ()
"test.op"() {angle = #angle.attr<-0.5pi>} : () -> ()

// CHECK-NEXT: "test.op"() {angle = #angle.attr<pi>} : () -> ()
"test.op"() {angle = #angle.attr<-pi>} : () -> ()

// CHECK-NEXT: %a = angle.constant<pi>
// CHECK-GENERIC: %a = "angle.constant"() <{angle = #angle.attr<pi>}> : () -> !angle.type
%a = angle.constant<pi>

%0 = "test.op"() : () -> i1

// CHECK: %a2 = angle.cond_negate %0, %a
// CHECK-GENERIC: %a2 = "angle.cond_negate"(%0, %a) : (i1, !angle.type) -> !angle.type
%a2 = angle.cond_negate %0, %a
