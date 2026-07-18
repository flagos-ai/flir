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

// RUN: triton-shared-opt --reconcile-ptr-casts %s | FileCheck %s

module {
  func.func @bitcast_ptr_as_src(%arg0: memref<*xi32>, %arg1: memref<*xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %0 = tptr.type_offset i32  : i32
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<*xi32> to !tt.ptr<i32>
    %2 = builtin.unrealized_conversion_cast %1 : !tt.ptr<i32> to !ptr.ptr
    %3 = builtin.unrealized_conversion_cast %arg0 : memref<*xi32> to !tt.ptr<i32>
    %4 = builtin.unrealized_conversion_cast %3 : !tt.ptr<i32> to !ptr.ptr
    %5 = arith.muli %c1_i32, %0 : i32
    %6 = tptr.ptradd %4 %5 : !ptr.ptr, i32 to !ptr.ptr
    %7 = builtin.unrealized_conversion_cast %6 : !ptr.ptr to !tt.ptr<i64>
    %8 = builtin.unrealized_conversion_cast %7 : !tt.ptr<i64> to memref<*xi64>
    %reinterpret_cast = memref.reinterpret_cast %8 to offset: [%c2], sizes: [16], strides: [1] : memref<*xi64> to memref<16xi64, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<16xi64>
    memref.copy %reinterpret_cast, %alloc : memref<16xi64, strided<[1], offset: ?>> to memref<16xi64>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<16xi64> to tensor<16xi64>
    %10 = tptr.ptradd %2 %5 : !ptr.ptr, i32 to !ptr.ptr
    %11 = builtin.unrealized_conversion_cast %10 : !ptr.ptr to !tt.ptr<i64>
    %12 = builtin.unrealized_conversion_cast %11 : !tt.ptr<i64> to memref<*xi64>
    %reinterpret_cast_0 = memref.reinterpret_cast %12 to offset: [%c2], sizes: [16], strides: [1] : memref<*xi64> to memref<16xi64, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %9 in writable %reinterpret_cast_0 : (tensor<16xi64>, memref<16xi64, strided<[1], offset: ?>>) -> ()
    return
  }
}

// CHECK-NOT: unrealized_conversion_cast
