// RUN: quopt %s -p canonicalize | filecheck %s

%c0 = arith.constant false

%a = "test.op"() : () -> !angle.type

%b = angle.cond_negate %c0, %a

// CHECK: "test.op"(%a) {cond_false} : (!angle.type) -> ()
"test.op"(%b) {cond_false} : (!angle.type) -> ()

%c1 = arith.constant true

%c = angle.constant<0.5pi>

%d = angle.cond_negate %c1, %c
// CHECK: [[const:%.*]] = angle.constant<1.5pi>
// CHECK: "test.op"([[const]]) : (!angle.type) -> ()
"test.op"(%d) : (!angle.type) -> ()

%x, %y = "test.op"() : () -> (i1, i1)

%e = angle.cond_negate %x, %a
%f = angle.cond_negate %y, %e
// CHECK: [[xor:%.*]] = arith.xori %y, %x
// CHECK: [[assoc:%.*]] = angle.cond_negate [[xor]], %a
// CHECK: "test.op"([[assoc]]) : (!angle.type) -> ()
"test.op"(%f) : (!angle.type) -> ()

%g = angle.constant<pi>
%h = angle.cond_negate %x, %g

%i = angle.constant<0>
%j = angle.cond_negate %y, %i

// CHECK: "test.op"(%g, %i) {pi_or_zero} : (!angle.type, !angle.type) -> ()
"test.op"(%h, %j) {pi_or_zero} : (!angle.type, !angle.type) -> ()
