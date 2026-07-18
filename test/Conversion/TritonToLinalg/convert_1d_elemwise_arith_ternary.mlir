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
    %a : !tt.ptr<i1>,
    %b : !tt.ptr<f32>,
    %c : !tt.ptr<f32>,
    %d : tensor<1024x!tt.ptr<f32>>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // a pointer
    %8 = tt.splat %a : !tt.ptr<i1> -> tensor<1024x!tt.ptr<i1>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>
    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // c pointer
    %28 = tt.splat %c : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %29 = tt.addptr %28, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %am = tt.load %9 : tensor<1024x!tt.ptr<i1>>
    %bm = tt.load %19 : tensor<1024x!tt.ptr<f32>>
    %cm = tt.load %29 : tensor<1024x!tt.ptr<f32>>
    %10 = arith.select %am, %bm, %cm : tensor<1024xi1>, tensor<1024xf32>
    tt.store %d, %10 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<*xi1>, %[[VAL_1:.*]]: memref<*xf32>, %[[VAL_2:.*]]: memref<*xf32>, %[[VAL_3:.*]]: memref<1024xf32>, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi1> to memref<1024xi1, strided<[1]>>
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<1024xi1>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_10]] : memref<1024xi1, strided<[1]>> to memref<1024xi1>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<1024xi1>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy %[[VAL_8]], %[[VAL_12]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:           %[[VAL_13:.*]] = bufferization.to_tensor %[[VAL_12]] restrict writable : memref<1024xf32>
// CHECK:           %[[VAL_14:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy %[[VAL_9]], %[[VAL_14]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:           %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_14]] restrict writable : memref<1024xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_11]], %[[VAL_13]], %[[VAL_15]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL_13]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i1, %[[VAL_18:.*]]: f32, %[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32):
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : f32
// CHECK:             linalg.yield %[[VAL_21]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           bufferization.materialize_in_destination %[[VAL_22:.*]] in writable %[[VAL_3]]
// CHECK:           return
// CHECK:         }
