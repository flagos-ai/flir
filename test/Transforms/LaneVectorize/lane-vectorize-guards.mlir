// RUN: triton-shared-opt --triton-lane-vectorize %s | FileCheck %s
//
// Behavior of triton-lane-vectorize around memory addressing. Integer/index
// cones that feed memory addresses must not be packed, while pure compute
// cones (including the coalesced slice path) still are.

// The coalesced path needs no allocation and is enabled: a contiguous run
// of extract_slice leaves is packed with one wider slice + reshape.
tt.func @guards_coalesced_slices_still_pack(%src: tensor<8xf32>, %y: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %l0 = tensor.extract_slice %src[0] [4] [1] : tensor<8xf32> to tensor<4xf32>
  %l1 = tensor.extract_slice %src[4] [4] [1] : tensor<8xf32> to tensor<4xf32>
  %r0 = arith.addf %l0, %y : tensor<4xf32>
  %r1 = arith.addf %l1, %y : tensor<4xf32>
  tt.return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: tt.func @guards_coalesced_slices_still_pack(
// CHECK: tensor.extract_slice %{{.*}}[0] [8] [1]
// CHECK: tensor.reshape
// CHECK-NOT: tensor.concat
// CHECK: arith.addf

// Integer/index values that reach tt.addptr are addressing, not data: even
// though their leaves here are a coalescible slice run, the cone must not be
// packed.
tt.func @guards_no_address_cone(%offs: tensor<8xi32>, %c: tensor<4xi32>, %b0: tensor<4x!tt.ptr<f32>>, %b1: tensor<4x!tt.ptr<f32>>) -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>) {
  %l0 = tensor.extract_slice %offs[0] [4] [1] : tensor<8xi32> to tensor<4xi32>
  %l1 = tensor.extract_slice %offs[4] [4] [1] : tensor<8xi32> to tensor<4xi32>
  %a0 = arith.addi %l0, %c : tensor<4xi32>
  %a1 = arith.addi %l1, %c : tensor<4xi32>
  %p0 = tt.addptr %b0, %a0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %p1 = tt.addptr %b1, %a1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  tt.return %p0, %p1 : tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>
}

// CHECK-LABEL: tt.func @guards_no_address_cone(
// CHECK-NOT: [0] [8] [1]
// CHECK-NOT: tensor.reshape
// CHECK: tt.addptr

// Cross-lane combines are lifted into a tt.reduce over the lane axis when the
// leaves can be coalesced; this integer example stays exact.
tt.func @guards_int_cross_lane_still_pack(%src: tensor<8xi32>, %y: tensor<4xi32>) -> tensor<4xi32> {
  %l0 = tensor.extract_slice %src[0] [4] [1] : tensor<8xi32> to tensor<4xi32>
  %l1 = tensor.extract_slice %src[4] [4] [1] : tensor<8xi32> to tensor<4xi32>
  %r0 = arith.muli %l0, %y : tensor<4xi32>
  %r1 = arith.muli %l1, %y : tensor<4xi32>
  %s = arith.addi %r0, %r1 : tensor<4xi32>
  tt.return %s : tensor<4xi32>
}

// CHECK-LABEL: tt.func @guards_int_cross_lane_still_pack(
// CHECK: tensor.reshape
// CHECK: arith.muli
// CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
// CHECK-NOT: tensor.concat
// CHECK: tt.return
