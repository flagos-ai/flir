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
    // Example 2, gep operand is another gep's output, which is passed into the loop as varible, used before update
    %_ptr2 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
        // perform load
        %3 = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr, %3 : tensor<256x!tt.ptr<bf16>>
        // pointer updates
        %4 = tt.splat %i_c3 : i32 -> tensor<256xi32>
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    // Expected output
    // %offset_dim0 = arith.constant 1024                                                       <- insert instructions to initialize init arg(new)
    // for iter_args (%offset_dim0_iter = %offset_dim0) {                                       <- replace varibles passed in as init arg (new)
    //  %subview = memref.subview %arg0, [%offset_dim0_iter][256][4] : memref<> -> memref<>     <- generate subview on load (new)
    //  ...
    //  %4 = %offset_dim0_iter + %c3                                                            <- replace gep of splat with add (already done)
    //  scf.yield %4                                                                            <- replace yielding an gep output with the corresponding dim variable (new)
    // }
    tt.return
  }
}
// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xbf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
// CHECK:     %c1024 = arith.constant 1024 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c12 = arith.constant 12 : index
// CHECK:     %c3 = arith.constant 3 : index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [1024], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: 1024>>
// CHECK:     %cast = memref.cast %reinterpret_cast : memref<256xbf16, strided<[1], offset: 1024>> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:     %0:2 = scf.for %arg7 = %c0 to %c12 step %c3 iter_args(%arg8 = %cast, %arg9 = %c1024) -> (memref<256xbf16, strided<[?], offset: ?>>, index) {
// CHECK:       %alloc = memref.alloc() : memref<256xbf16>
// CHECK:       memref.copy %arg8, %alloc : memref<256xbf16, strided<[?], offset: ?>> to memref<256xbf16>
// CHECK:       %1 = bufferization.to_tensor %alloc restrict writable : memref<256xbf16> to tensor<256xbf16>
// CHECK:       bufferization.materialize_in_destination %1 in writable %arg8 : (tensor<256xbf16>, memref<256xbf16, strided<[?], offset: ?>>) -> ()
// CHECK:       %2 = arith.addi %arg9, %c3 : index
// CHECK:       %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
// CHECK:       %cast_1 = memref.cast %reinterpret_cast_0 : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:       scf.yield %cast_1, %2 : memref<256xbf16, strided<[?], offset: ?>>, index
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
