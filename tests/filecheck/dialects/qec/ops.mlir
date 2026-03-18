// RUN: QUOPT_ROUNDTRIP
// RUN: QUOPT_GENERIC_ROUNDTRIP

// CHECK: instrument.constant #qec.perfect
// CHECK-GENERIC: "instrument.constant"() <{instrument = #qec.perfect}> : () -> !instrument.type<5>
instrument.constant #qec.perfect

// CHECK: instrument.constant #qec.stabilizer<IIXXI>
// CHECK-GENERIC: "instrument.constant"() <{instrument = #qec.stabilizer<IIXXI>}> : () -> !instrument.type<5, i1>
instrument.constant #qec.stabilizer<IIXXI>
