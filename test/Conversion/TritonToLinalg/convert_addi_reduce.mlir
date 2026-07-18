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

// RUN: triton-shared-opt --triton-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @addi(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.addi %arg14, %arg15 : i32
      tt.reduce.return %69 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL:   func.func @addi(
// CHECK-SAME:              %[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[VAL_9:.*]] = linalg.fill ins(%[[VAL_7]] : i32) outs(%[[VAL_8]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[VAL_10:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:           %[[VAL_11:.*]] = tensor.insert %[[VAL_7]] into %[[VAL_10]][] : tensor<i32>
// CHECK:           %[[VAL_12:.*]] = linalg.reduce ins(%[[VAL_9]] : tensor<4096xi32>) outs(%[[VAL_11]] : tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32) {
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:               linalg.yield %[[VAL_15]] : i32
// CHECK:             }
// CHECK:           %[[VAL_16:.*]] = tensor.extract %[[VAL_12]][] : tensor<i32>
// CHECK:           %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:           affine.store %[[VAL_16]], %[[VAL_17]][0] : memref<1xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }
