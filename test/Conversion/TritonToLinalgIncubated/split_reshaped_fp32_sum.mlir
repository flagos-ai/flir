// RUN: triton-shared-opt --triton-to-linalg-incubated=named-ops=true %s | FileCheck %s
// REQUIRES: flir-incubated

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK-LABEL: func.func @split_sum(
  // CHECK: %[[FLAT:.*]] = tensor.collapse_shape {{.*}} into tensor<8192xf32>
  // CHECK: %[[EVEN:.*]] = hfusion.deinterleave %[[FLAT]] channel<0>
  // CHECK: %[[ODD:.*]] = hfusion.deinterleave %[[FLAT]] channel<1>
  // CHECK: return %[[EVEN]], %[[ODD]]
  func.func @split_sum(%lhs: tensor<64x128xf32>, %rhs: tensor<64x128xf32>) -> (tensor<4096xf32>, tensor<4096xf32>) {
    %sum = arith.addf %lhs, %rhs : tensor<64x128xf32>
    %pairs = tt.reshape %sum : tensor<64x128xf32> -> tensor<4096x2xf32>
    %even, %odd = tt.split %pairs : tensor<4096x2xf32> -> tensor<4096xf32>
    return %even, %odd : tensor<4096xf32>, tensor<4096xf32>
  }

  // CHECK-LABEL: func.func @split_input(
  // CHECK-NOT: hfusion.deinterleave
  // CHECK: tensor.extract_slice %arg0[0, 0]
  // CHECK: tensor.extract_slice %arg0[0, 1]
  // CHECK: return
  func.func @split_input(%input: tensor<4096x2xf32>) -> (tensor<4096xf32>, tensor<4096xf32>) {
    %even, %odd = tt.split %input : tensor<4096x2xf32> -> tensor<4096xf32>
    return %even, %odd : tensor<4096xf32>, tensor<4096xf32>
  }
}
