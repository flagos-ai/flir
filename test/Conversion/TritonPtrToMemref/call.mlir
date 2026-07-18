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

// RUN: triton-shared-opt --triton-arith-to-linalg --triton-ptr-to-memref %s | FileCheck %s

module {
  tt.func @_sum_combine__fp32(%arg0: !tt.ptr<f32>) -> f32{
    %0 = arith.constant 42.0 : f32
    tt.return %0 : f32
  }
  tt.func @test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) -> f32{
    %0 = tt.call @_sum_combine__fp32(%arg0) : (!tt.ptr<f32>) -> f32
    %1 = tt.call @_sum_combine__fp32(%arg1) : (!tt.ptr<f32>) -> f32
    %2 = arith.addf %0, %1 : f32
    tt.return %2 : f32
  }
}

// CHECK: module {
// CHECK:   func.func @_sum_combine__fp32(%arg0: memref<*xf32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> f32 {
// CHECK:     %cst = arith.constant 4.200000e+01 : f32
// CHECK:     return %cst : f32
// CHECK:   }
// CHECK:   func.func @test(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) -> f32 {
// CHECK:     %0 = call @_sum_combine__fp32(%arg0, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (memref<*xf32>, i32, i32, i32, i32, i32, i32) -> f32
// CHECK:     %1 = call @_sum_combine__fp32(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (memref<*xf32>, i32, i32, i32, i32, i32, i32) -> f32
// CHECK:     %2 = arith.addf %0, %1 : f32
// CHECK:     return %2 : f32
// CHECK:   }
// CHECK: }
