// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated
// Out-of-range equality is an empty mask; never build a negative or
// out-of-bounds slice. Keep the existing lowering for live and dynamic masks.

// -----

// CHECK-LABEL: func.func @below(
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @below(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<0> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @at_end(
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @at_end(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<49> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @above(
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @above(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<57> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @negative_below(
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @negative_below(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<-17> : tensor<32xi32>
  %bound = arith.constant dense<-18> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @negative_above(
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @negative_above(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<-17> : tensor<32xi32>
  %bound = arith.constant dense<15> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @first(
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: return
func.func @first(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<17> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @middle(
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: return
func.func @middle(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<24> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @last(
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: return
func.func @last(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<48> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @negative_middle(
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: return
func.func @negative_middle(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<-17> : tensor<32xi32>
  %bound = arith.constant dense<0> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @row_below(
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @row_below(%t: tensor<32x8xf32>, %f: tensor<32x8xf32>) -> tensor<32x8xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<16> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %e = tt.expand_dims %eq {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
  %mask = tt.broadcast %e : tensor<32x1xi1> -> tensor<32x8xi1>
  %v = arith.select %mask, %t, %f : tensor<32x8xi1>, tensor<32x8xf32>
  return %v : tensor<32x8xf32>
}

// -----

// CHECK-LABEL: func.func @column_above(
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: arith.select
// CHECK: return %arg1
func.func @column_above(%t: tensor<4x32xf32>, %f: tensor<4x32xf32>) -> tensor<4x32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %start = arith.constant dense<17> : tensor<32xi32>
  %bound = arith.constant dense<49> : tensor<32xi32>
  %i = arith.addi %r, %start : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %e = tt.expand_dims %eq {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mask = tt.broadcast %e : tensor<1x32xi1> -> tensor<4x32xi1>
  %v = arith.select %mask, %t, %f : tensor<4x32xi1>, tensor<4x32xf32>
  return %v : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_bound(
// CHECK: scf.if
// CHECK: tensor.extract_slice
// CHECK: return
func.func @dynamic_bound(%t: tensor<32xf32>, %f: tensor<32xf32>, %scalar: i32) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %splat = tt.splat %scalar : i32 -> tensor<32xi32>
  %eq = arith.cmpi eq, %r, %splat : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_start(
// CHECK: scf.if
// CHECK: tensor.extract_slice
// CHECK: return
func.func @dynamic_start(%t: tensor<32xf32>, %f: tensor<32xf32>, %scalar: i32) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %splat = tt.splat %scalar : i32 -> tensor<32xi32>
  %bound = arith.constant dense<0> : tensor<32xi32>
  %i = arith.addi %r, %splat : tensor<32xi32>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @wrapped_i32(
// CHECK-NOT: tensor.extract_slice
// CHECK: arith.select
// CHECK: return
func.func @wrapped_i32(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %shift = arith.constant dense<2147483647> : tensor<32xi32>
  %wrapped = arith.addi %r, %shift : tensor<32xi32>
  %bound = arith.constant dense<-2147483648> : tensor<32xi32>
  %eq = arith.cmpi eq, %wrapped, %bound : tensor<32xi32>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @wrapped_then_extended(
// CHECK-NOT: tensor.extract_slice
// CHECK: arith.select
// CHECK: return
func.func @wrapped_then_extended(%t: tensor<32xf32>, %f: tensor<32xf32>) -> tensor<32xf32> {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %shift = arith.constant dense<2147483647> : tensor<32xi32>
  %wrapped = arith.addi %r, %shift : tensor<32xi32>
  %i = arith.extsi %wrapped : tensor<32xi32> to tensor<32xi64>
  %bound = arith.constant dense<-2147483648> : tensor<32xi64>
  %eq = arith.cmpi eq, %i, %bound : tensor<32xi64>
  %v = arith.select %eq, %t, %f : tensor<32xi1>, tensor<32xf32>
  return %v : tensor<32xf32>
}
