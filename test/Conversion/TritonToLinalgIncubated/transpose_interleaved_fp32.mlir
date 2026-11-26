// RUN: triton-shared-opt --triton-to-linalg-incubated %s | FileCheck %s
// REQUIRES: flir-incubated

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK-LABEL: func.func @swap_outer_axes(
  // CHECK: %[[FLAT:.*]] = tensor.collapse_shape
  // CHECK: hfusion.deinterleave %[[FLAT]] channel<0>
  // CHECK: linalg.transpose {{.*}} permutation = [1, 0]
  // CHECK: hfusion.deinterleave %[[FLAT]] channel<1>
  // CHECK: linalg.transpose {{.*}} permutation = [1, 0]
  // CHECK: hfusion.interleave
  // CHECK: tensor.expand_shape {{.*}} into tensor<32x16x2xf32>
  func.func @swap_outer_axes(%input: tensor<16x32x2xf32>) -> tensor<32x16x2xf32> {
    %result = tt.trans %input {order = array<i32: 1, 0, 2>} : tensor<16x32x2xf32> -> tensor<32x16x2xf32>
    return %result : tensor<32x16x2xf32>
  }

  // CHECK-LABEL: func.func @swap_inner_axes(
  // CHECK-NOT: hfusion.deinterleave
  // CHECK: linalg.transpose {{.*}} permutation = [0, 2, 1]
  // CHECK-NOT: hfusion.interleave
  // CHECK: return
  func.func @swap_inner_axes(%input: tensor<16x32x2xf32>) -> tensor<16x2x32xf32> {
    %result = tt.trans %input {order = array<i32: 0, 2, 1>} : tensor<16x32x2xf32> -> tensor<16x2x32xf32>
    return %result : tensor<16x2x32xf32>
  }
}
