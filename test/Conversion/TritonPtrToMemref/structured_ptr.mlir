// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

module {
  func.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %4 = arith.addi %2, %c1024 : index
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = arith.minsi %4, %5 : index
    %7 = arith.subi %6, %2 : index
    %8 = "tts.load"(%3, %7) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %9 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %10 = "tts.load"(%9, %7) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %11 = arith.addf %8, %10 : tensor<1024xf32>
    %12 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%12, %11, %7) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, index) -> ()
    return
  }
}

// CHECK:   func.func public @add_kernel_01234(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32)
// CHECK-DAG: builtin.unrealized_conversion_cast %arg2 : memref<*xf32> to !tt.ptr<f32>
// CHECK-DAG: builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !tt.ptr<f32>
// CHECK-DAG: builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !tt.ptr<f32>
