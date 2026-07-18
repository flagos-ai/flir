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

// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s
module {
  tt.func public @minmax_olt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf olt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_olt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.minimumf [[PARAM_1_]], [[PARAM_2_]] : f32

// -----

module {
  tt.func public @minmax_ole(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ole, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_ole
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.minimumf [[PARAM_1_]], [[PARAM_2_]] : f32

// -----

module {
  tt.func public @minmax_ogt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ogt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_ogt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.maximumf [[PARAM_1_]], [[PARAM_2_]] : f32

// -----

module {
  tt.func public @minmax_oge(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf oge, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_oge
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.maximumf [[PARAM_1_]], [[PARAM_2_]] : f32
