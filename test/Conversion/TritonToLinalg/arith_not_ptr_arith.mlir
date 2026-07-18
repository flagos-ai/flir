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
  tt.func @kernel(
    %a : !tt.ptr<i32>,
    %b : !tt.ptr<i32>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>
        %bm = tt.load %19 : tensor<1024x!tt.ptr<i32>>
        %5 = arith.addi %am, %bm : tensor<1024xi32>
        tt.store %19, %5 : tensor<1024x!tt.ptr<i32>>
        tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: memref<*xi32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK:           %[[VAL_5:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK:           %[[VAL_6:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy %[[VAL_5]], %[[VAL_7]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_tensor %[[VAL_7]] restrict writable : memref<1024xi32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy %[[VAL_6]], %[[VAL_9]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<1024xi32>
// CHECK:           %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_8]], %[[VAL_10]] : tensor<1024xi32>, tensor<1024xi32>) outs(%[[VAL_8]] : tensor<1024xi32>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32):
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:             linalg.yield %[[VAL_15]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           bufferization.materialize_in_destination %[[VAL_16:.*]] in writable %[[VAL_6]]
// CHECK:           return
// CHECK:         }
