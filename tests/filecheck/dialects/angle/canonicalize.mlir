// RUN: quopt %s -p canonicalize --split-input-file | filecheck %s

// CHECK-LABEL: negate_fold
func.func @negate_fold() -> !angle.type {
  %a = angle.constant<0.5pi>

  // CHECK: [[angle:%.*]] = angle.constant<1.5pi>
  %b = angle.negate %a

  // CHECK: func.return [[angle]]
  func.return %b : !angle.type
}

// -----

// CHECK-LABEL: negate_negate
func.func @negate_negate(%a : !angle.type) -> !angle.type {
  %b = angle.negate %a
  %c = angle.negate %b

  // CHECK: func.return %a
  func.return %c : !angle.type
}

// -----

func.func @negate_cond_negate(%a : !angle.type, %x: i1) -> !angle.type {
  // CHECK: [[true:%.*]] = arith.constant true
  // CHECK-NEXT: [[xor:%.*]] = arith.xori %x, [[true]]
  // CHECK-NEXT: [[res:%.*]] = angle.cond_negate [[xor]], %a
  %b = angle.cond_negate %x, %a
  %c = angle.negate %b

  // CHECK: func.return [[res]]
  func.return %c : !angle.type
}

// -----

// CHECK-LABEL: cond_negate_pi
func.func @cond_negate_pi(%x : i1) -> !angle.type {
  // CHECK: %a = angle.constant<pi>
  %a = angle.constant<pi>
  // CHECK-NOT: angle.cond_negate
  %b = angle.cond_negate %x, %a

  // CHECK: func.return %a
  func.return %b : !angle.type
}

// -----

// CHECK-LABEL: cond_negate_zero
func.func @cond_negate_zero(%x : i1) -> !angle.type {
  // CHECK: %a = angle.constant<0>
  %a = angle.constant<0>
  // CHECK-NOT: angle.cond_negate
  %b = angle.cond_negate %x, %a

  // CHECK: func.return %a
  func.return %b : !angle.type
}

// -----

// CHECK-LABEL: cond_false
func.func @cond_false(%a : !angle.type) -> !angle.type {
  %cFalse = arith.constant false

  // CHECK-NOT: angle.cond_negate
  %b = angle.cond_negate %cFalse, %a

  // CHECK: func.return %a
  func.return %b : !angle.type
}

// -----

// CHECK-LABEL: cond_true
func.func @cond_true(%a : !angle.type) -> !angle.type {
  %cTrue = arith.constant true

  // CHECK-NOT: angle.cond_negate
  // CHECK: [[res:%.*]] = angle.negate %a
  %b = angle.cond_negate %cTrue, %a
  // CHECK-NOT: angle.cond_negate

  // CHECK: func.return [[res]]
  func.return %b : !angle.type
}

// -----

// CHECK-LABEL: cond_negate_negate
func.func @cond_negate_negate(%a : !angle.type, %x: i1) -> !angle.type {
  // CHECK: [[true:%.*]] = arith.constant true
  // CHECK-NEXT: [[xor:%.*]] = arith.xori %x, [[true]]
  // CHECK-NEXT: [[res:%.*]] = angle.cond_negate [[xor]], %a
  %b = angle.negate %a
  %c = angle.cond_negate %x, %b

  // CHECK: func.return [[res]]
  func.return %c : !angle.type
}

// -----

// CHECK-LABEL: cond_negate_cond_negate
func.func @cond_negate_cond_negate(%a : !angle.type, %x: i1, %y: i1) -> !angle.type {
  // CHECK: [[xor:%.*]] = arith.xori %y, %x
  // CHECK: [[res:%.*]] = angle.cond_negate [[xor]], %a
  %b = angle.cond_negate %x, %a
  %c = angle.cond_negate %y, %b

  // CHECK: func.return [[res]]
  func.return %c : !angle.type
}

// -----

// CHECK-LABEL: scale_fold
func.func @scale_fold() -> !angle.type {
  %a = angle.constant<0.5pi>

  %c2 = arith.constant 2.0 : f64
  // angle.constant<pi>
  %a1 = angle.scale %a, %c2

  func.return %a1 : !angle.type
}

// -----

// CHECK-LABEL: add_fold
func.func @add_fold() -> !angle.type {
  %a = angle.constant<0.5pi>
  %b = angle.constant<pi>
  // CHECK: angle.constant<1.5pi>
  %c = angle.add %a, %b

  func.return %c : !angle.type
}
