// RUN: triton-shared-opt --triton-to-linalg-incubated=named-ops=true --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-incubated="named-ops=true compile-on-910-95=true" --split-input-file %s | FileCheck %s --check-prefix=LEGACY --implicit-check-not=memref_ext.alloc_workspace
// REQUIRES: flir-incubated
// LEGACY: module

// Both split results reach Cube through independent GM/L1 transfers. Reusing
// the first result in an explicit accumulator chain must not add a third copy.
// CHECK-LABEL: func.func @split_rhs(
// CHECK: symbol = "custom_flagtree_split_float"
// CHECK: memref_ext.alloc_workspace()
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK: linalg.matmul
// CHECK: memref_ext.alloc_workspace()
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @split_rhs(%x: tensor<4096x2xf32>, %w: tensor<64x32xf32>) -> (tensor<64x128xf32>, tensor<4096xf32>) {
    %a, %b = tt.split %x : tensor<4096x2xf32> -> tensor<4096xf32>
    %ar = tt.reshape %a : tensor<4096xf32> -> tensor<32x128xf32>
    %br = tt.reshape %b : tensor<4096xf32> -> tensor<32x128xf32>
    %zero = arith.constant dense<0.0> : tensor<64x128xf32>
    %d0 = tt.dot %w, %ar, %zero, inputPrecision = ieee : tensor<64x32xf32> * tensor<32x128xf32> -> tensor<64x128xf32>
    %d1 = tt.dot %w, %br, %d0, inputPrecision = ieee : tensor<64x32xf32> * tensor<32x128xf32> -> tensor<64x128xf32>
    %d2 = tt.dot %w, %ar, %d1, inputPrecision = ieee : tensor<64x32xf32> * tensor<32x128xf32> -> tensor<64x128xf32>
    return %d2, %a : tensor<64x128xf32>, tensor<4096xf32>
  }
}

// -----

// CHECK-LABEL: func.func @split_lhs(
// CHECK: symbol = "custom_flagtree_split_float"
// CHECK: memref_ext.alloc_workspace()
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @split_lhs(%x: tensor<1024x2xf32>, %w: tensor<16x32xf32>) -> tensor<64x32xf32> {
    %a, %b = tt.split %x : tensor<1024x2xf32> -> tensor<1024xf32>
    %ar = tt.reshape %a : tensor<1024xf32> -> tensor<16x64xf32>
    %at = tt.trans %ar {order = array<i32: 1, 0>} : tensor<16x64xf32> -> tensor<64x16xf32>
    %zero = arith.constant dense<0.0> : tensor<64x32xf32>
    %d = tt.dot %at, %w, %zero, inputPrecision = hf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>
    return %d : tensor<64x32xf32>
  }
}

// -----

// Cover the converted tensor.reshape and an intervening legal elementwise op.
// CHECK-LABEL: func.func @custom_vector_reshape(
// CHECK: hivm.hir.custom
// CHECK: memref_ext.alloc_workspace()
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @custom_vector_reshape(%x: tensor<1024xf32>, %w: tensor<32x16xf32>) -> tensor<32x64xf32> {
    %init = bufferization.alloc_tensor() : tensor<1024xf32>
    %v = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "test_vector"} "test.vector" ins(%x : tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
    %shape = arith.constant dense<[16, 64]> : tensor<2xi64>
    %view = tensor.reshape %v(%shape) : (tensor<1024xf32>, tensor<2xi64>) -> tensor<16x64xf32>
    %zeros = arith.constant dense<0.0> : tensor<16x64xf32>
    %sum = arith.addf %view, %zeros : tensor<16x64xf32>
    %zero = arith.constant dense<0.0> : tensor<32x64xf32>
    %d = tt.dot %w, %sum, %zero, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
    return %d : tensor<32x64xf32>
  }
}

// -----

// A Cube custom result already belongs to the consumer core.
// CHECK-LABEL: func.func @custom_cube_keeps_native_path(
// CHECK: hivm.hir.custom
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @custom_cube_keeps_native_path(%x: tensor<16x64xf32>, %w: tensor<32x16xf32>) -> tensor<32x64xf32> {
    %init = bufferization.alloc_tensor() : tensor<16x64xf32>
    %v = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_M>, hivm.tcore_type = #hivm.tcore_type<CUBE>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "test_cube"} "test.cube" ins(%x : tensor<16x64xf32>) outs(%init : tensor<16x64xf32>) -> tensor<16x64xf32>
    %zero = arith.constant dense<0.0> : tensor<32x64xf32>
    %d = tt.dot %w, %v, %zero, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
    return %d : tensor<32x64xf32>
  }
}
