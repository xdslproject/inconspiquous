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

// CHECK: %a1 = angle.negate %a
// CHECK-GENERIC: %a1 = "angle.negate"(%a) : (!angle.type) -> !angle.type
%a1 = angle.negate %a

// CHECK: %a2 = angle.cond_negate %0, %a1
// CHECK-GENERIC: %a2 = "angle.cond_negate"(%0, %a1) : (i1, !angle.type) -> !angle.type
%a2 = angle.cond_negate %0, %a1
