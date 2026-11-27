// RUN: triton-shared-opt --triton-to-linalg-incubated %s | FileCheck %s
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
}
