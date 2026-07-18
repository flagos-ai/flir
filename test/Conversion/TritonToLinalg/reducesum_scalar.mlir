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

// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>)
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
    %3 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
    tt.store %res, %3 : !tt.ptr<bf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<128xbf16>
// CHECK:           memref.copy %[[VAL_6]], %[[VAL_7]] : memref<128xbf16, strided<[1]>> to memref<128xbf16>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_tensor %[[VAL_7]] restrict writable : memref<128xbf16>
// CHECK:           %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = tensor.insert %[[VAL_5]] into %[[VAL_9]][] : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<128xbf16>) outs(%[[VAL_10]] : tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_12:.*]]: bf16, %[[VAL_13:.*]]: f32) {
// CHECK:               %[[VAL_14:.*]] = arith.extf %[[VAL_12]] : bf16 to f32
// CHECK:               %[[VAL_15:.*]] = arith.addf %[[VAL_14]], %[[VAL_13]] : f32
// CHECK:               linalg.yield %[[VAL_15]] : f32
// CHECK:             }
// CHECK:           %[[VAL_16:.*]] = tensor.extract %[[VAL_11]][] : tensor<f32>
// CHECK:           %[[VAL_17:.*]] = arith.truncf %[[VAL_16]] : f32 to bf16
// CHECK:           %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1]>>
// CHECK:           affine.store %[[VAL_17]], %[[VAL_18]][0] : memref<1xbf16, strided<[1]>>
// CHECK:           return
// CHECK:         }
