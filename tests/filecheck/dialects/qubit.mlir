// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: %q = qubit.alloc
// CHECK-GENERIC: %q = "qubit.alloc"() <{alloc = #qubit.zero}> : () -> !qubit.bit
%q = qubit.alloc

// CHECK: %q2 = qubit.alloc<#qubit.plus>
// CHECK-GENERIC: %q2 = "qubit.alloc"() <{alloc = #qubit.plus}> : () -> !qubit.bit
%q2 = qubit.alloc<#qubit.plus>
