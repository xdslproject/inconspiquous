// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q = qu.alloc
// CHECK-GENERIC: %q = "qu.alloc"() <{alloc = #qu.zero}> : () -> !qu.bit
%q = qu.alloc

// CHECK: %q2 = qu.alloc<#qu.plus>
// CHECK-GENERIC: %q2 = "qu.alloc"() <{alloc = #qu.plus}> : () -> !qu.bit
%q2 = qu.alloc<#qu.plus>
