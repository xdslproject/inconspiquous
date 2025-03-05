// RUN: quopt %s -p convert-qref-to-qssa,qft-std-inline | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @qft_std(%q1 : !qubit.bit, %q2 : !qubit.bit, %q3 : !qubit.bit, %q4 : !qubit.bit, %q5 : !qubit.bit) {
// CHECK-NEXT:      %q1_1 = qssa.gate<#qft.h> %q1
// CHECK-NEXT:      %q1_2, %q2_1 = qssa.gate<#qft.pair<1>> %q1_1, %q2
// CHECK-NEXT:      %q1_3, %q3_1 = qssa.gate<#qft.pair<2>> %q1_2, %q3
// CHECK-NEXT:      %q1_4, %q4_1 = qssa.gate<#qft.pair<3>> %q1_3, %q4
// CHECK-NEXT:      %q1_5, %q5_1 = qssa.gate<#qft.pair<4>> %q1_4, %q5
// CHECK-NEXT:      %q2_2 = qssa.gate<#qft.h> %q2_1
// CHECK-NEXT:      %q2_3, %q3_2 = qssa.gate<#qft.pair<1>> %q2_2, %q3_1
// CHECK-NEXT:      %q2_4, %q4_2 = qssa.gate<#qft.pair<2>> %q2_3, %q4_1
// CHECK-NEXT:      %q2_5, %q5_2 = qssa.gate<#qft.pair<3>> %q2_4, %q5_1
// CHECK-NEXT:      %q3_3 = qssa.gate<#qft.h> %q3_2
// CHECK-NEXT:      %q3_4, %q4_3 = qssa.gate<#qft.pair<1>> %q3_3, %q4_2
// CHECK-NEXT:      %q3_5, %q5_3 = qssa.gate<#qft.pair<2>> %q3_4, %q5_2
// CHECK-NEXT:      %q4_4 = qssa.gate<#qft.h> %q4_3
// CHECK-NEXT:      %q4_5, %q5_4 = qssa.gate<#qft.pair<1>> %q4_4, %q5_3
// CHECK-NEXT:      %q5_5 = qssa.gate<#qft.h> %q5_4
// CHECK-NEXT:      %q1_6 = qssa.gate<#qft.h> %q1_5
// CHECK-NEXT:      %q1_7, %q2_6 = qssa.gate<#qft.pair<1>> %q1_6, %q2_5
// CHECK-NEXT:      %q1_8, %q3_6 = qssa.gate<#qft.pair<2>> %q1_7, %q3_5
// CHECK-NEXT:      %q1_9, %q4_6 = qssa.gate<#qft.pair<3>> %q1_8, %q4_5
// CHECK-NEXT:      %q1_10, %q5_6 = qssa.gate<#qft.pair<4>> %q1_9, %q5_5
// CHECK-NEXT:      %q2_7 = qssa.gate<#qft.h> %q2_6
// CHECK-NEXT:      %q2_8, %q3_7 = qssa.gate<#qft.pair<1>> %q2_7, %q3_6
// CHECK-NEXT:      %q2_9, %q4_7 = qssa.gate<#qft.pair<2>> %q2_8, %q4_6
// CHECK-NEXT:      %q2_10, %q5_7 = qssa.gate<#qft.pair<3>> %q2_9, %q5_6
// CHECK-NEXT:      %q3_8 = qssa.gate<#qft.h> %q3_7
// CHECK-NEXT:      %q3_9, %q4_8 = qssa.gate<#qft.pair<1>> %q3_8, %q4_7
// CHECK-NEXT:      %q3_10, %q5_8 = qssa.gate<#qft.pair<2>> %q3_9, %q5_7
// CHECK-NEXT:      %q4_9 = qssa.gate<#qft.h> %q4_8
// CHECK-NEXT:      %q4_10, %q5_9 = qssa.gate<#qft.pair<1>> %q4_9, %q5_8
// CHECK-NEXT:      %q5_10 = qssa.gate<#qft.h> %q5_9
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
func.func @qft_std(%q1: !qubit.bit, %q2: !qubit.bit, %q3: !qubit.bit, %q4: !qubit.bit, %q5: !qubit.bit) {
  qref.gate<#qft.n<5>> %q1, %q2, %q3, %q4, %q5
  qref.gate<#qft.n<5>> %q1, %q2, %q3, %q4, %q5
  func.return
}
