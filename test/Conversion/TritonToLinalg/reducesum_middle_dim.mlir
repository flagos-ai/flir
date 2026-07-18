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
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : !tt.ptr<bf16>,
        %out: tensor<32x16x!tt.ptr<bf16>>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<32xi32>
    %ws = arith.muli %ct256, %0 : tensor<32xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %m2 = tt.broadcast %1 : tensor<32x1xi32> -> tensor<32x256xi32>
    %100 = tt.expand_dims %m2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %moff = tt.broadcast %100 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %33 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %k2 = tt.broadcast %34 : tensor<1x256xi32> -> tensor<32x256xi32>
    %200 = tt.expand_dims %k2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %koff = tt.broadcast %200 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %23 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2 = tt.broadcast %24 : tensor<1x16xi32> -> tensor<256x16xi32>
    %300 = tt.expand_dims %n2 {axis = 0 : i32} : tensor<256x16xi32> -> tensor<1x256x16xi32>
    %noff = tt.broadcast %300 : tensor<1x256x16xi32> -> tensor<32x256x16xi32>
    %mkoff = arith.addi %moff, %koff : tensor<32x256x16xi32>
    %mknoff = arith.addi %mkoff, %noff : tensor<32x256x16xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<32x256x16x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mknoff : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>
    %afm = tt.load %9 : tensor<32x256x16x!tt.ptr<bf16>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 1 : i32} : (tensor<32x256x16xbf16>) -> tensor<32x16xbf16>
    tt.store %out, %5 : tensor<32x16x!tt.ptr<bf16>>
    tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: memref<32x16xbf16>, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32) {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [32, 256, 16], strides: {{\[}}%[[VAL_6]], 1, 1] : memref<*xbf16> to memref<32x256x16xbf16, strided<[?, 1, 1]>>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<32x256x16xbf16>
// CHECK:           memref.copy %[[VAL_8]], %[[VAL_9]] : memref<32x256x16xbf16, strided<[?, 1, 1]>> to memref<32x256x16xbf16>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<32x256x16xbf16>
// CHECK:           %[[VAL_11:.*]] = tensor.empty() : tensor<32x16xbf16>
// CHECK:           %[[VAL_12:.*]] = linalg.fill ins(%[[VAL_7]] : bf16) outs(%[[VAL_11]] : tensor<32x16xbf16>) -> tensor<32x16xbf16>
// CHECK:           %[[VAL_13:.*]] = linalg.reduce ins(%[[VAL_10]] : tensor<32x256x16xbf16>) outs(%[[VAL_12]] : tensor<32x16xbf16>) dimensions = [1]
// CHECK:             (%[[VAL_14:.*]]: bf16, %[[VAL_15:.*]]: bf16) {
// CHECK:               %[[VAL_16:.*]] = arith.addf %[[VAL_14]], %[[VAL_15]] : bf16
// CHECK:               linalg.yield %[[VAL_16]] : bf16
// CHECK:             }
// CHECK:           bufferization.materialize_in_destination %[[VAL_13]] in writable %[[VAL_2]]
// CHECK:           return
// CHECK:         }
