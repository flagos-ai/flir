// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 1

    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 1

    // gep operand is another gep' output, which is passed into the loop as varible, used after update
    %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
      %6 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>

      %8 = tt.broadcast %7 : tensor<256x1xi32> -> tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 0], strides: [1, 0]

      %9 = tt.make_range {end = 512 : i32, start = 256 : i32} : tensor<256xi32>
      %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>

      %11 = tt.broadcast %10 : tensor<1x256xi32> -> tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 256], strides: [0, 1]

      %12 = arith.addi %8, %11 : tensor<256x256xi32>
      // sizes: [256, 256], offsets: [0, 256], strides: [1, 1]

      %13 = tt.expand_dims %ptr {axis = 1 : i32} : tensor<256x!tt.ptr<bf16>> -> tensor<256x1x!tt.ptr<bf16>>
      %14 = tt.broadcast %13 : tensor<256x1x!tt.ptr<bf16>> -> tensor<256x256x!tt.ptr<bf16>>

      %15 = tt.addptr %14, %12 : tensor<256x256x!tt.ptr<bf16>>, tensor<256x256xi32>
      // source: arg0, sizes: [256, 256], offsets: [1024 + i, 256], strides: [2, 1]

      // perform load
      %16 = tt.load %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x256x!tt.ptr<bf16>>
      tt.store %15, %16 : tensor<256x256x!tt.ptr<bf16>>
      // pointer updates
      %17 = tt.splat %i_c3 : i32 -> tensor<256xi32>
      // sizes: 256, offsets: 3, strides: 0
      %ptr_iter = tt.addptr %ptr, %17 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
      // source: arg0, sizes: 256, offsets: 1024 + i, strides: 4
      scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}
// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
// CHECK:     %c256 = arith.constant 256 : index
// CHECK:     %c1024 = arith.constant 1024 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c12 = arith.constant 12 : index
// CHECK:     %c3 = arith.constant 3 : index
// CHECK:     %0 = scf.for %arg7 = %c0 to %c12 step %c3 iter_args(%arg8 = %c1024) -> (index) {
// CHECK:       %1 = arith.addi %arg8, %c256 : index
// CHECK:       %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256, 256], strides: [2, 1] : memref<*xbf16> to memref<256x256xbf16, strided<[2, 1], offset: ?>>
// CHECK:       %cast = memref.cast %reinterpret_cast : memref<256x256xbf16, strided<[2, 1], offset: ?>> to memref<256x256xbf16, strided<[?, 1], offset: ?>>
// CHECK:       %alloc = memref.alloc() : memref<256x256xbf16>
// CHECK:       memref.copy %reinterpret_cast, %alloc : memref<256x256xbf16, strided<[2, 1], offset: ?>> to memref<256x256xbf16>
// CHECK:       %2 = bufferization.to_tensor %alloc restrict writable : memref<256x256xbf16> to tensor<256x256xbf16>
// CHECK:       bufferization.materialize_in_destination %2 in writable %cast : (tensor<256x256xbf16>, memref<256x256xbf16, strided<[?, 1], offset: ?>>) -> ()
// CHECK:       %3 = arith.addi %arg8, %c3 : index
// CHECK:       scf.yield %3 : index
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
