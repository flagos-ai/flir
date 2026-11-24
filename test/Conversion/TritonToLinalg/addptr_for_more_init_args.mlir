// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 1
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 1
    %3 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg1, sizes: 256, offsets: 1024, strides: 1
    %_arg2, %_ptr_ld, %_arg3, %_ptr_st, %_arg4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%arg2 = %c1, %ptr_ld = %2, %arg3 = %c2, %ptr_st = %4, %arg4 = %c3) -> (index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index) {
        // perform load
        %5 = tt.load %ptr_ld {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr_st, %5 : tensor<256x!tt.ptr<bf16>>
        // pointer updates
        %cast3 = arith.index_cast %c3 : index to i32
        %6 = tt.splat %cast3 : i32 -> tensor<256xi32>
        %ptr_ld_iter = tt.addptr %ptr_ld, %6 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg0, sizes: 256, offsets: 1024 + i*3, strides: 1
        %arg2_iter = arith.addi %arg2, %c3 : index
        %arg3_iter = arith.addi %arg3, %c3 : index
        %arg4_iter = arith.addi %arg4, %c3 : index
        %7 = arith.addi %arg2_iter, %arg3_iter : index
        %8 = arith.addi %7, %arg4_iter : index
        %cast8 = arith.index_cast %8 : index to i32
        %9 = tt.splat %cast8 : i32 -> tensor<256xi32>
        %ptr_st_iter = tt.addptr %ptr_st, %9 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg1, sizes: 256, offsets: 1024 + loop-carry variable*i, strides: 1
        scf.yield %arg2_iter, %ptr_ld_iter, %arg3_iter, %ptr_st_iter, %arg4_iter : index, tensor<256x!tt.ptr<bf16>>, index, tensor<256x!tt.ptr<bf16>>, index
    }
    tt.return
  }
}
// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
// CHECK:     %c1024 = arith.constant 1024 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %c3 = arith.constant 3 : index
// CHECK:     %c12 = arith.constant 12 : index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [1024], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: 1024>>
// CHECK:     %cast = memref.cast %reinterpret_cast : memref<256xbf16, strided<[1], offset: 1024>> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [1024], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: 1024>>
// CHECK:     %cast_1 = memref.cast %reinterpret_cast_0 : memref<256xbf16, strided<[1], offset: 1024>> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:     %0:7 = scf.for %arg8 = %c0 to %c12 step %c3 iter_args(%arg9 = %c1, %arg10 = %cast, %arg11 = %c2, %arg12 = %cast_1, %arg13 = %c3, %arg14 = %c1024, %arg15 = %c1024) -> (index, memref<256xbf16, strided<[?], offset: ?>>, index, memref<256xbf16, strided<[?], offset: ?>>, index, index, index) {
// CHECK:       %alloc = memref.alloc() : memref<256xbf16>
// CHECK:       memref.copy %arg10, %alloc : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
// CHECK:       %1 = bufferization.to_tensor %alloc restrict writable : memref<256xbf16> to tensor<256xbf16>
// CHECK:       bufferization.materialize_in_destination %1 in writable %arg12 : (tensor<256xbf16>, memref<256xbf16, strided<[?], offset: ?>>) -> ()
// CHECK:       %2 = arith.addi %arg14, %c3 : index
// CHECK:       %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
// CHECK:       %cast_3 = memref.cast %reinterpret_cast_2 : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:       %3 = arith.addi %arg9, %c3 : index
// CHECK:       %4 = arith.addi %arg11, %c3 : index
// CHECK:       %5 = arith.addi %arg13, %c3 : index
// CHECK:       %6 = arith.addi %3, %4 : index
// CHECK:       %7 = arith.addi %6, %5 : index
// CHECK:       %8 = arith.addi %arg15, %7 : index
// CHECK:       %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%8], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
// CHECK:       %cast_5 = memref.cast %reinterpret_cast_4 : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:       scf.yield %3, %cast_3, %4, %cast_5, %5, %2, %8 : index, memref<256xbf16, strided<[?], offset: ?>>, index, memref<256xbf16, strided<[?], offset: ?>>, index, index, index
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
