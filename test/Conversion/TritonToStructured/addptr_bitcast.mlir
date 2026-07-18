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

// RUN: triton-shared-opt --triton-to-structured %s | FileCheck %s

module {
  tt.func @test(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : !tt.ptr<i64>
    %1 = tt.int_to_ptr %0 : i64 -> !tt.ptr<f16>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %5 = tt.load %4 : tensor<32x!tt.ptr<f16>>
    %6 = tt.bitcast %arg1 : !tt.ptr<f32> -> !tt.ptr<f16>
    %7 = tt.splat %6 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %8 = tt.addptr %7, %2 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    tt.store %8, %5 : tensor<32x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK: [[IN_SRC:%.+]] = tt.int_to_ptr
// CHECK: [[IN_PTR:%.+]] = tts.make_tptr [[IN_SRC]]
// CHECK: [[OUT_SRC:%.+]] = tt.bitcast
// CHECK: [[OUT_PTR:%.+]] = tts.make_tptr [[OUT_SRC]]
