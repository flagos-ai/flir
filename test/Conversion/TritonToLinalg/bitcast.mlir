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
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>

    // cast result before doing float add
    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>


    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xi32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
// CHECK:   [[RC_:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1]{{.*}} : memref<*xi32> to memref<1024xi32, strided<[1]>>
// CHECK:   [[RC_0_:%.+]] = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1]{{.*}} : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK:   [[ALLOC_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:   memref.copy [[RC_]], [[ALLOC_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK:   [[VAR_0_:%.+]] = bufferization.to_tensor [[ALLOC_]] restrict writable : memref<1024xi32>
// CHECK:   [[VAR_1_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:   [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<1024xi32>) outs([[VAR_1_]] : tensor<1024xf32>) {
// CHECK:   ^bb0(%in: i32, %out: f32):
// CHECK:     [[VAR_5_:%.+]] = arith.bitcast %in : i32 to f32
// CHECK:     linalg.yield [[VAR_5_]] : f32
// CHECK:   } -> tensor<1024xf32>
// CHECK:   bufferization.materialize_in_destination [[VAR_2_]] in writable [[RC_0_]]
// CHECK:     return
// CHECK:   }
// CHECK: }
