// RUN: triton-shared-opt --triton-to-linalg-incubated=named-ops=true --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-incubated="named-ops=true compile-on-910-95=true" --split-input-file %s | FileCheck %s --check-prefix=LEGACY --implicit-check-not=flagtree.vector_to_cube
// REQUIRES: flir-incubated
// LEGACY: module

// -----

// CHECK-LABEL: func.func @segmented_rhs(
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: linalg.matmul
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: linalg.matmul
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @segmented_rhs(%src: memref<16x16x16xf32>, %weight: tensor<16x16xf32>) -> tensor<16x256xf32> {
  %buf = memref.alloc() : memref<16x16x16xf32>
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%buf : memref<16x16x16xf32>)
  %src_slice = memref.subview %src[0, 0, 0] [16, 15, 16] [1, 1, 1] : memref<16x16x16xf32> to memref<16x15x16xf32, strided<[256, 16, 1]>>
  %dst_slice = memref.subview %buf[0, 0, 0] [16, 15, 16] [1, 1, 1] : memref<16x16x16xf32> to memref<16x15x16xf32, strided<[256, 16, 1]>>
  memref.copy %src_slice, %dst_slice : memref<16x15x16xf32, strided<[256, 16, 1]>> to memref<16x15x16xf32, strided<[256, 16, 1]>>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<16x16x16xf32> to tensor<16x16x16xf32>
  %shape = arith.constant dense<[16, 256]> : tensor<2xi64>
  %matrix = tensor.reshape %loaded(%shape) : (tensor<16x16x16xf32>, tensor<2xi64>) -> tensor<16x256xf32>
  %init = arith.constant dense<0.0> : tensor<16x256xf32>
  %dot0 = tt.dot %weight, %matrix, %init, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x256xf32> -> tensor<16x256xf32>
  %dot1 = tt.dot %weight, %matrix, %dot0, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x256xf32> -> tensor<16x256xf32>
  return %dot1 : tensor<16x256xf32>
 }
}

// -----

// CHECK-LABEL: func.func @unmasked_rhs(
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: linalg.matmul
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: linalg.matmul
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @unmasked_rhs(%src: memref<16x16x64xf32>, %weight: tensor<16x16xf32>) -> tensor<16x1024xf32> {
  %buf = memref.alloc() : memref<16x16x64xf32>
  %zero = arith.constant 0.0 : f32
  memref.copy %src, %buf : memref<16x16x64xf32> to memref<16x16x64xf32>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<16x16x64xf32> to tensor<16x16x64xf32>
  %shape = arith.constant dense<[16, 1024]> : tensor<2xi64>
  %matrix = tensor.reshape %loaded(%shape) : (tensor<16x16x64xf32>, tensor<2xi64>) -> tensor<16x1024xf32>
  %init = arith.constant dense<0.0> : tensor<16x1024xf32>
  %dot0 = tt.dot %weight, %matrix, %init, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x1024xf32> -> tensor<16x1024xf32>
  %dot1 = tt.dot %weight, %matrix, %dot0, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x1024xf32> -> tensor<16x1024xf32>
  return %dot1 : tensor<16x1024xf32>
 }
}

// -----

// CHECK-LABEL: func.func @segmented_lhs(
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: linalg.matmul
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: linalg.matmul
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @segmented_lhs(%src: memref<16x16x16xf32>, %weight: tensor<16x16xf32>) -> tensor<256x16xf32> {
  %buf = memref.alloc() : memref<16x16x16xf32>
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%buf : memref<16x16x16xf32>)
  %src_slice = memref.subview %src[0, 0, 0] [16, 15, 16] [1, 1, 1] : memref<16x16x16xf32> to memref<16x15x16xf32, strided<[256, 16, 1]>>
  %dst_slice = memref.subview %buf[0, 0, 0] [16, 15, 16] [1, 1, 1] : memref<16x16x16xf32> to memref<16x15x16xf32, strided<[256, 16, 1]>>
  memref.copy %src_slice, %dst_slice : memref<16x15x16xf32, strided<[256, 16, 1]>> to memref<16x15x16xf32, strided<[256, 16, 1]>>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<16x16x16xf32> to tensor<16x16x16xf32>
  %shape = arith.constant dense<[256, 16]> : tensor<2xi64>
  %matrix = tensor.reshape %loaded(%shape) : (tensor<16x16x16xf32>, tensor<2xi64>) -> tensor<256x16xf32>
  %init = arith.constant dense<0.0> : tensor<256x16xf32>
  %dot0 = tt.dot %matrix, %weight, %init, inputPrecision = ieee : tensor<256x16xf32> * tensor<16x16xf32> -> tensor<256x16xf32>
  %dot1 = tt.dot %matrix, %weight, %dot0, inputPrecision = ieee : tensor<256x16xf32> * tensor<16x16xf32> -> tensor<256x16xf32>
  return %dot1 : tensor<256x16xf32>
 }
}

// -----

// CHECK-LABEL: func.func @unit_axis(
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @unit_axis(%src: memref<1x16x256xf32>, %weight: tensor<16x16xf32>) -> tensor<16x256xf32> {
  %buf = memref.alloc() : memref<1x16x256xf32>
  %zero = arith.constant 0.0 : f32
  memref.copy %src, %buf : memref<1x16x256xf32> to memref<1x16x256xf32>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<1x16x256xf32> to tensor<1x16x256xf32>
  %shape = arith.constant dense<[16, 256]> : tensor<2xi64>
  %matrix = tensor.reshape %loaded(%shape) : (tensor<1x16x256xf32>, tensor<2xi64>) -> tensor<16x256xf32>
  %init = arith.constant dense<0.0> : tensor<16x256xf32>
  %dot0 = tt.dot %weight, %matrix, %init, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x256xf32> -> tensor<16x256xf32>
  %dot1 = tt.dot %weight, %matrix, %dot0, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x256xf32> -> tensor<16x256xf32>
  return %dot1 : tensor<16x256xf32>
 }
}

// -----

// CHECK-LABEL: func.func @plain_matrix(
// CHECK-NOT: flagtree.vector_to_cube
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @plain_matrix(%src: memref<16x256xf32>, %weight: tensor<16x16xf32>) -> tensor<16x256xf32> {
  %buf = memref.alloc() : memref<16x256xf32>
  %zero = arith.constant 0.0 : f32
  memref.copy %src, %buf : memref<16x256xf32> to memref<16x256xf32>
  %loaded = bufferization.to_tensor %buf restrict writable : memref<16x256xf32> to tensor<16x256xf32>
  %matrix = tensor.cast %loaded : tensor<16x256xf32> to tensor<16x256xf32>
  %init = arith.constant dense<0.0> : tensor<16x256xf32>
  %dot0 = tt.dot %weight, %matrix, %init, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x256xf32> -> tensor<16x256xf32>
  %dot1 = tt.dot %weight, %matrix, %dot0, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x256xf32> -> tensor<16x256xf32>
  return %dot1 : tensor<16x256xf32>
 }
}
