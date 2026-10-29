// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-incubated=compile-on-910-95=true --split-input-file %s | FileCheck %s --check-prefix=LEGACY
// REQUIRES: flir-incubated

// LEGACY: module
// LEGACY-NOT: custom_flagtree_

// Exact inverses must fold before splitting into physical channel buffers.
// CHECK-LABEL: func.func @inverse(
// CHECK-NOT: hivm.hir.custom
// CHECK-NOT: tensor.extract_slice
// CHECK: return %arg0
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @inverse(%x: tensor<32x2xf32>) -> tensor<32x2xf32> {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    %y = tt.join %a, %b : tensor<32xf32> -> tensor<32x2xf32>
    return %y : tensor<32x2xf32>
  }
}

// -----

// Independent inputs cannot be folded away.
// CHECK-LABEL: func.func @independent_join(
// CHECK: tensor.concat
// CHECK: symbol = "custom_flagtree_stride_gather_int32"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @independent_join(%a: tensor<32xf32>, %b: tensor<32xf32>) -> tensor<32x2xf32> {
    %y = tt.join %a, %b : tensor<32xf32> -> tensor<32x2xf32>
    return %y : tensor<32x2xf32>
  }
}

// -----

// CHECK-LABEL: func.func @static_transpose(
// CHECK: hivm.hir.varange
// CHECK: arith.andi
// CHECK: symbol = "custom_flagtree_stride_gather_half"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @static_transpose(%x: tensor<4x32xf16>) -> tensor<32x4xf16> {
    %y = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x32xf16> -> tensor<32x4xf16>
    return %y : tensor<32x4xf16>
  }
}

// -----

// The static permutation's 16 KiB index budget must not capture large tiles.
// CHECK-LABEL: func.func @large_transpose(
// CHECK-NOT: hivm.hir.custom
// CHECK: linalg.transpose
// CHECK-NOT: hivm.hir.custom
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @large_transpose(%x: tensor<64x256xf32>) -> tensor<256x64xf32> {
    %y = tt.trans %x {order = array<i32: 1, 0>} : tensor<64x256xf32> -> tensor<256x64xf32>
    return %y : tensor<256x64xf32>
  }
}

// -----

// A computed/input value cannot assume the layout of a dense single-use load.
// CHECK-LABEL: func.func @general_split(
// CHECK: symbol = "custom_flagtree_split_float"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @general_split(%x: tensor<32x2xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    return %a, %b : tensor<32xf32>, tensor<32xf32>
  }
}

// -----

// Narrow casts must be flat: old CANN's sub-block 2D casts mishandle row strides.
// CHECK-LABEL: func.func @narrow_split(
// CHECK: symbol = "custom_flagtree_byte_to_word"
// CHECK: symbol = "custom_flagtree_split_int16"
// CHECK: symbol = "custom_flagtree_word_to_byte"
// CHECK: symbol = "custom_flagtree_word_to_byte"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @narrow_split(%x: tensor<8x8x2xi8>) -> (tensor<8x8xi8>, tensor<8x8xi8>) {
    %a, %b = tt.split %x : tensor<8x8x2xi8> -> tensor<8x8xi8>
    return %a, %b : tensor<8x8xi8>, tensor<8x8xi8>
  }
}

// -----

// Unit extent makes the stride irrelevant; the base address/offset must stay.
// CHECK-LABEL: func.func @unit_copy(
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [1], strides: [1]
// CHECK: memref.copy {{.*}} : memref<1xi8> to memref<1xi8>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unit_copy(%src: memref<1xi8, strided<[3]>>, %dst: memref<1xi8>) {
    memref.copy %src, %dst : memref<1xi8, strided<[3]>> to memref<1xi8>
    return
  }
}

// -----

// A nonzero descriptor offset cannot be reset by this rewrite.
// CHECK-LABEL: func.func @unit_copy_offset(
// CHECK-NOT: memref.reinterpret_cast
// CHECK: memref.copy {{.*}} : memref<1xi8, strided<[3], offset: 2>> to memref<1xi8>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unit_copy_offset(%src: memref<1xi8, strided<[3], offset: 2>>, %dst: memref<1xi8>) {
    memref.copy %src, %dst : memref<1xi8, strided<[3], offset: 2>> to memref<1xi8>
    return
  }
}

// -----

// Large byte joins retain the existing packed interleave before widening.
// CHECK-LABEL: func.func @large_byte_join(
// CHECK-NOT: custom_flagtree_
// CHECK: tensor.insert_slice
// CHECK: tensor.insert_slice
// CHECK-NOT: custom_flagtree_
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @large_byte_join(%a: tensor<4096xi8>, %b: tensor<4096xi8>) -> tensor<4096x2xi8> {
    %y = tt.join %a, %b : tensor<4096xi8> -> tensor<4096x2xi8>
    return %y : tensor<4096x2xi8>
  }
}

// -----

// Preserving a short final axis still needs a packed physical layout.
// CHECK-LABEL: func.func @unchanged_short_inner_axis(
// CHECK-NOT: linalg.transpose
// CHECK: symbol = "custom_flagtree_stride_gather_float"
// CHECK-NOT: linalg.transpose
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unchanged_short_inner_axis(%x: tensor<16x4x2xf32>) -> tensor<4x16x2xf32> {
    %y = tt.trans %x {order = array<i32: 1, 0, 2>} : tensor<16x4x2xf32> -> tensor<4x16x2xf32>
    return %y : tensor<4x16x2xf32>
  }
}
