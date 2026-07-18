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
tt.func public @assert_lol(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
  %1 = tt.splat %0 : i1 -> tensor<1xi1>
  tt.assert %1, "lol": tensor<1xi1>
  tt.return
}

// CHECK: func.func @assert_lol(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
// CHECK:   %c0_i32 = arith.constant 0 : i32
// CHECK:   %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
// CHECK:   cf.assert %0, "Assertion `lol` failed"
// CHECK:   return
// CHECK: }
