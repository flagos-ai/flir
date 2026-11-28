// RUN: %flir-ascend-opt --split-input-file --triton-to-structured-incubated --triton-to-structured-incubated --triton-to-linalg-incubated %s | FileCheck %s
// REQUIRES: flir-incubated

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK-LABEL: func.func @transpose_into_narrow_rows(
  // CHECK-NOT: linalg.transpose
  // CHECK: call @triton_gather{{.*}} : (tensor<64xf32>, tensor<64xi32>, i32) -> tensor<64xf32>
  // CHECK-NOT: linalg.transpose
  // CHECK: return
  func.func @transpose_into_narrow_rows(%input: tensor<4x16xf32>) -> tensor<16x4xf32> {
    %result = tt.trans %input {order = array<i32: 1, 0>} : tensor<4x16xf32> -> tensor<16x4xf32>
    return %result : tensor<16x4xf32>
  }

  // CHECK-LABEL: func.func @transpose_out_of_narrow_rows(
  // CHECK-NOT: linalg.transpose
  // CHECK: call @triton_gather{{.*}} : (tensor<1024xf32>, tensor<1024xi32>, i32) -> tensor<1024xf32>
  // CHECK-NOT: linalg.transpose
  // CHECK: return
  func.func @transpose_out_of_narrow_rows(%input: tensor<256x4xf32>) -> tensor<4x256xf32> {
    %result = tt.trans %input {order = array<i32: 1, 0>} : tensor<256x4xf32> -> tensor<4x256xf32>
    return %result : tensor<4x256xf32>
  }

  // CHECK-LABEL: func.func @swap_outer_axes_with_narrow_rows(
  // CHECK-NOT: linalg.transpose
  // CHECK: call @triton_gather{{.*}} : (tensor<1024xf32>, tensor<1024xi32>, i32) -> tensor<1024xf32>
  // CHECK-NOT: linalg.transpose
  // CHECK: return
  func.func @swap_outer_axes_with_narrow_rows(%input: tensor<16x16x4xf32>) -> tensor<16x16x4xf32> {
    %result = tt.trans %input {order = array<i32: 1, 0, 2>} : tensor<16x16x4xf32> -> tensor<16x16x4xf32>
    return %result : tensor<16x16x4xf32>
  }

  // CHECK-LABEL: func.func @transpose_aligned_rows(
  // CHECK: linalg.transpose {{.*}} permutation = [1, 0]
  // CHECK: return
  func.func @transpose_aligned_rows(%input: tensor<8x16xf32>) -> tensor<16x8xf32> {
    %result = tt.trans %input {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
    return %result : tensor<16x8xf32>
  }

  // A strided load introduces a transpose of the same type as the explicit one.
  // CHECK-LABEL: func.func @layout_and_value_transpose(
  // CHECK-DAG: linalg.transpose ins({{.*}}tensor<32x4xf32>) outs({{.*}}tensor<4x32xf32>) permutation = [1, 0]
  // CHECK-DAG: call @triton_gather{{.*}} : (tensor<128xf32>, tensor<128xi32>, i32) -> tensor<128xf32>
  // CHECK: return
  tt.func public @layout_and_value_transpose(%input: !tt.ptr<f32>, %data: !tt.ptr<f32>,
                                             %copy: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %four = arith.constant dense<4> : tensor<1x32xi32>
    %major = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %minor = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %major_col = tt.expand_dims %major {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %minor_row = tt.expand_dims %minor {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %strided = arith.muli %minor_row, %four : tensor<1x32xi32>
    %offset0 = tt.broadcast %major_col : tensor<4x1xi32> -> tensor<4x32xi32>
    %offset1 = tt.broadcast %strided : tensor<1x32xi32> -> tensor<4x32xi32>
    %offset = arith.addi %offset0, %offset1 : tensor<4x32xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<4x32x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %input_base, %offset : tensor<4x32x!tt.ptr<f32>>, tensor<4x32xi32>
    %src = tt.load %src_ptrs : tensor<4x32x!tt.ptr<f32>>
    %linear = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %copy_base = tt.splat %copy : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %copy_ptrs = tt.addptr %copy_base, %linear : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %flat_copy = tt.reshape %src : tensor<4x32xf32> -> tensor<128xf32>
    tt.store %copy_ptrs, %flat_copy : tensor<128x!tt.ptr<f32>>
    %data_base = tt.splat %data : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %data_ptrs = tt.addptr %data_base, %linear : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %flat = tt.load %data_ptrs : tensor<128x!tt.ptr<f32>>
    %tile = tt.reshape %flat : tensor<128xf32> -> tensor<32x4xf32>
    %result = tt.trans %tile {order = array<i32: 1, 0>} : tensor<32x4xf32> -> tensor<4x32xf32>
    %flat_out = tt.reshape %result : tensor<4x32xf32> -> tensor<128xf32>
    %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %linear : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %out_ptrs, %flat_out : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Model the input to a later pass after an implicit transpose was introduced.
// Folding the inner transpose changes the order recorded on the outer one.
module attributes {hacc.target = #hacc.target<"Ascend910B4">, tt.original_transpose_orders_recorded} {
  // CHECK-LABEL: func.func @transpose_folded_with_layout(
  // CHECK-NOT: call @triton_gather
  // CHECK: linalg.transpose {{.*}} permutation = [1, 0, 2]
  // CHECK-NOT: call @triton_gather
  // CHECK: return
  func.func @transpose_folded_with_layout(%input: tensor<16x16x4xf32>) -> tensor<16x16x4xf32> {
    %layout = tt.trans %input {order = array<i32: 1, 2, 0>} : tensor<16x16x4xf32> -> tensor<16x4x16xf32>
    %result = tt.trans %layout {order = array<i32: 0, 2, 1>, tt.original_transpose_order = array<i32: 0, 2, 1>} : tensor<16x4x16xf32> -> tensor<16x16x4xf32>
    return %result : tensor<16x16x4xf32>
  }
}
