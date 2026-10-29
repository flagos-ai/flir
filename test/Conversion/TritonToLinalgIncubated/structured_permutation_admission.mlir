// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// A surviving split consumer must not prevent the exact inverse from folding.
// CHECK-LABEL: func.func @inverse_extra_user(
// CHECK: symbol = "custom_flagtree_split_float"
// CHECK: return %arg0,
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @inverse_extra_user(%x: tensor<32x2xf32>) -> (tensor<32x2xf32>, tensor<32xf32>) {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    %y = tt.join %a, %b : tensor<32xf32> -> tensor<32x2xf32>
    return %y, %a : tensor<32x2xf32>, tensor<32xf32>
  }
}

// -----

// CHECK-LABEL: func.func @swapped_channels(
// CHECK: symbol = "custom_flagtree_split_float"
// CHECK: tensor.concat
// CHECK: symbol = "custom_flagtree_stride_gather_int32"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @swapped_channels(%x: tensor<32x2xf32>) -> tensor<32x2xf32> {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    %y = tt.join %b, %a : tensor<32xf32> -> tensor<32x2xf32>
    return %y : tensor<32x2xf32>
  }
}

// -----

// CHECK-LABEL: func.func @duplicated_channel(
// CHECK: symbol = "custom_flagtree_split_float"
// CHECK: tensor.concat
// CHECK: symbol = "custom_flagtree_stride_gather_int32"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @duplicated_channel(%x: tensor<32x2xf32>) -> tensor<32x2xf32> {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    %y = tt.join %a, %a : tensor<32xf32> -> tensor<32x2xf32>
    return %y : tensor<32x2xf32>
  }
}

// -----

// CHECK-LABEL: func.func @different_split_sources(
// CHECK-COUNT-2: symbol = "custom_flagtree_split_float"
// CHECK: tensor.concat
// CHECK: symbol = "custom_flagtree_stride_gather_int32"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @different_split_sources(%x: tensor<32x2xf32>, %z: tensor<32x2xf32>) -> tensor<32x2xf32> {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    %c, %d = tt.split %z : tensor<32x2xf32> -> tensor<32xf32>
    %y = tt.join %a, %d : tensor<32xf32> -> tensor<32x2xf32>
    return %y : tensor<32x2xf32>
  }
}

// -----

// Legal factoring can commute the channel selection before transpose.
// CHECK-LABEL: func.func @commutable_split(
// CHECK-NOT: custom_flagtree_split_
// CHECK: tensor.extract_slice
// CHECK-NOT: custom_flagtree_split_
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @commutable_split(%x: tensor<4x32xf32>) -> (tensor<32x2xf32>, tensor<32x2xf32>) {
    %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x32xf32> -> tensor<32x4xf32>
    %r = tt.reshape %t : tensor<32x4xf32> -> tensor<32x2x2xf32>
    %a, %b = tt.split %r : tensor<32x2x2xf32> -> tensor<32x2xf32>
    return %a, %b : tensor<32x2xf32>, tensor<32x2xf32>
  }
}

// -----

// CHECK-LABEL: func.func @allow_reorder_cannot_commute(
// CHECK: symbol = "custom_flagtree_split_float"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @allow_reorder_cannot_commute(%x: tensor<4x32xf32>) -> (tensor<32x2xf32>, tensor<32x2xf32>) {
    %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x32xf32> -> tensor<32x4xf32>
    %r = tt.reshape %t allow_reorder : tensor<32x4xf32> -> tensor<32x2x2xf32>
    %a, %b = tt.split %r : tensor<32x2x2xf32> -> tensor<32x2xf32>
    return %a, %b : tensor<32x2xf32>, tensor<32x2xf32>
  }
}

// -----

// CHECK-LABEL: func.func @reshape_changes_outer_axis(
// CHECK: symbol = "custom_flagtree_split_float"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @reshape_changes_outer_axis(%x: tensor<4x32xf32>) -> (tensor<16x4xf32>, tensor<16x4xf32>) {
    %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x32xf32> -> tensor<32x4xf32>
    %r = tt.reshape %t : tensor<32x4xf32> -> tensor<16x4x2xf32>
    %a, %b = tt.split %r : tensor<16x4x2xf32> -> tensor<16x4xf32>
    return %a, %b : tensor<16x4xf32>, tensor<16x4xf32>
  }
}

// -----

// CHECK-LABEL: func.func @index_budget_boundary(
// CHECK: symbol = "custom_flagtree_stride_gather_half"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @index_budget_boundary(%x: tensor<4x1024xf16>) -> tensor<1024x4xf16> {
    %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x1024xf16> -> tensor<1024x4xf16>
    return %t : tensor<1024x4xf16>
  }
}

// -----

// Reject the packed route before introducing any widening/scratch operations.
// CHECK-LABEL: func.func @large_byte_transpose(
// CHECK-NOT: custom_flagtree_
// CHECK: linalg.transpose
// CHECK-NOT: custom_flagtree_
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @large_byte_transpose(%x: tensor<4x2048xi8>) -> tensor<2048x4xi8> {
    %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x2048xi8> -> tensor<2048x4xi8>
    return %t : tensor<2048x4xi8>
  }
}

// -----

// CHECK-LABEL: func.func @join_budget_boundary(
// CHECK: symbol = "custom_flagtree_stride_gather_int32"
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @join_budget_boundary(%a: tensor<2048xf32>, %b: tensor<2048xf32>) -> tensor<2048x2xf32> {
    %y = tt.join %a, %b : tensor<2048xf32> -> tensor<2048x2xf32>
    return %y : tensor<2048x2xf32>
  }
}

// -----

// CHECK-LABEL: func.func @unsupported_word_width(
// CHECK-NOT: custom_flagtree_
// CHECK: linalg.transpose
// CHECK-NOT: custom_flagtree_
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unsupported_word_width(%x: tensor<4x32xf64>) -> tensor<32x4xf64> {
    %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x32xf64> -> tensor<32x4xf64>
    return %t : tensor<32x4xf64>
  }
}

// -----

// CHECK-LABEL: func.func @unit_copy_dynamic_offset(
// CHECK-NOT: memref.reinterpret_cast
// CHECK: memref.copy
// CHECK-NOT: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unit_copy_dynamic_offset(%a: memref<1xi8, strided<[3], offset: ?>>, %b: memref<1xi8>) {
    memref.copy %a, %b : memref<1xi8, strided<[3], offset: ?>> to memref<1xi8>
    return
  }
}
