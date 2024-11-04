// RUN: QUOPT_ROUNDTRIP

%q0 = qubit.alloc
%q1 = qubit.alloc
%q2 = qubit.alloc

%q0_1 = qssa.gate<#gate.h> %q0 : !qubit.bit
%q0_2, %q1_2 = qssa.gate<#gate.cnot> %q0_1, %q1 : !qubit.bit, !qubit.bit
%q0_3, %q2_3 = qssa.gate<#gate.cnot> %q0_2, %q2 : !qubit.bit, !qubit.bit
