// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg1splat = tt.splat %arg1 : i32 -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg1splat : tensor<4x256xi32>
    // offset = [%arg1,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %6 = arith.constant 5 : i32
    %splat6 = tt.splat %6 : i32 -> tensor<4x256xi32>
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32>
    // offset = [%arg1, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1, 0], size = [4, 256], stride = [1, 5]
    %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %9, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1+%arg1, 0], size = [4, 256], stride = [2, 10]
    %13 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
    %14 = arith.addf %10, %13 : tensor<4x256xbf16>
    %16 = tt.addptr %12, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1+%arg1+%arg1, 0], size = [4, 256], stride = [3, 15]
    tt.store %16, %14 : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
// CHECK:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK:     %alloc = memref.alloc() : memref<4x256xbf16>
// CHECK:     memref.copy %reinterpret_cast, %alloc : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16>
// CHECK:     %1 = bufferization.to_tensor %alloc restrict writable : memref<4x256xbf16> to tensor<4x256xbf16>
// CHECK:     %2 = arith.index_cast %arg1 : i32 to index
// CHECK:     %3 = arith.index_cast %arg1 : i32 to index
// CHECK:     %4 = arith.addi %2, %3 : index
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [4, 256], strides: [2, 10] : memref<*xbf16> to memref<4x256xbf16, strided<[2, 10], offset: ?>>
// CHECK:     %alloc_1 = memref.alloc() : memref<4x256xbf16>
// CHECK:     memref.copy %reinterpret_cast_0, %alloc_1 : memref<4x256xbf16, strided<[2, 10], offset: ?>> to memref<4x256xbf16>
// CHECK:     %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<4x256xbf16> to tensor<4x256xbf16>
// CHECK:     %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %5 : tensor<4x256xbf16>, tensor<4x256xbf16>) outs(%1 : tensor<4x256xbf16>) {
// CHECK:     ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
// CHECK:       %12 = arith.addf %in, %in_3 : bf16
// CHECK:       linalg.yield %12 : bf16
// CHECK:     } -> tensor<4x256xbf16>
// CHECK:     %7 = arith.index_cast %arg1 : i32 to index
// CHECK:     %8 = arith.index_cast %arg1 : i32 to index
// CHECK:     %9 = arith.addi %7, %8 : index
// CHECK:     %10 = arith.index_cast %arg1 : i32 to index
// CHECK:     %11 = arith.addi %9, %10 : index
// CHECK:     %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [4, 256], strides: [3, 15] : memref<*xbf16> to memref<4x256xbf16, strided<[3, 15], offset: ?>>
// CHECK:     %cast = memref.cast %reinterpret_cast_2 : memref<4x256xbf16, strided<[3, 15], offset: ?>> to memref<4x256xbf16, strided<[3, ?], offset: ?>>
// CHECK:     bufferization.materialize_in_destination %6 in writable %cast : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[3, ?], offset: ?>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
