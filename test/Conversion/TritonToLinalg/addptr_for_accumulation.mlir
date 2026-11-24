// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>,
    %arg3 : i32,
    %arg4 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg3splat = tt.splat %arg3 : i32 -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg3splat : tensor<4x256xi32>
    // offset = [%arg3,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %c5 = arith.constant 5 : i32
    %splat6 = tt.splat %c5 : i32 -> tensor<4x256xi32>
    // scalar = 5
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>> // Why is the input unknown
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %19 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>> // this will be replaced with a memref.copy
    %11 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %11, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg1, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %19, %ptr_iter = %12) -> (tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>) {
        %20 = tt.load %ptr_iter {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
        %sum = arith.addf %sum_iter, %20 : tensor<4x256xbf16>
        // pointer updates
        %17 = tt.splat %i_c3 : i32 -> tensor<4x256xi32>
        // offset: [3, 0], size = [4, 256], stride [0, 0]
        %ptr = tt.addptr %ptr_iter, %17 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        // source: %arg1, offset = [%arg3+%i, 0], size = [4, 256], stride = [1, 5]
        scf.yield %sum, %ptr : tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>
    }
    %15 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %16 = tt.addptr %15, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg2, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    tt.store %16, %sum_out : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
// CHECK:     %c3 = arith.constant 3 : index
// CHECK:     %c12 = arith.constant 12 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %0 = arith.index_cast %arg3 : i32 to index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK:     %alloc = memref.alloc() : memref<4x256xbf16>
// CHECK:     memref.copy %reinterpret_cast, %alloc : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16>
// CHECK:     %1 = bufferization.to_tensor %alloc restrict writable : memref<4x256xbf16> to tensor<4x256xbf16>
// CHECK:     %2 = arith.index_cast %arg3 : i32 to index
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK:     %cast = memref.cast %reinterpret_cast_0 : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %3:3 = scf.for %arg11 = %c0 to %c12 step %c3 iter_args(%arg12 = %1, %arg13 = %cast, %arg14 = %2) -> (tensor<4x256xbf16>, memref<4x256xbf16, strided<[?, ?], offset: ?>>, index) {
// CHECK:       %alloc_3 = memref.alloc() : memref<4x256xbf16>
// CHECK:       memref.copy %arg13, %alloc_3 : memref<4x256xbf16, strided<[?, ?], offset: ?>> to memref<4x256xbf16>
// CHECK:       %5 = bufferization.to_tensor %alloc_3 restrict writable : memref<4x256xbf16> to tensor<4x256xbf16>
// CHECK:       %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg12, %5 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%arg12 : tensor<4x256xbf16>) {
// CHECK:       ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
// CHECK:         %8 = arith.addf %in, %in_6 : bf16
// CHECK:         linalg.yield %8 : bf16
// CHECK:       } -> tensor<4x256xbf16>
// CHECK:       %7 = arith.addi %arg14, %c3 : index
// CHECK:       %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK:       %cast_5 = memref.cast %reinterpret_cast_4 : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16, strided<[?, ?], offset: ?>>
// CHECK:       scf.yield %6, %cast_5, %7 : tensor<4x256xbf16>, memref<4x256xbf16, strided<[?, ?], offset: ?>>, index
// CHECK:     }
// CHECK:     %4 = arith.index_cast %arg3 : i32 to index
// CHECK:     %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK:     %cast_2 = memref.cast %reinterpret_cast_1 : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16, strided<[1, ?], offset: ?>>
// CHECK:     bufferization.materialize_in_destination %3#0 in writable %cast_2 : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[1, ?], offset: ?>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
