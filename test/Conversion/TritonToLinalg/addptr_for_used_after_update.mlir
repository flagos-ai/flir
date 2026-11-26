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
        // pointer updates
        %4 = tt.splat %i_c3 : i32 -> tensor<256xi32>
        // sizes: 256, offsets: 3, strides: 0
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg0, sizes: 256, offsets: 1024 + i, strides: 1
        // perform load
        %3 = tt.load %ptr_iter {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr_iter, %3 : tensor<256x!tt.ptr<bf16>>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    // Expected output
    // %offset_dim0 = arith.constant 1024                                                       <- insert instructions to initialize init arg(new)
    // for iter_args (%offset_dim0_iter = %offset_dim0) {                                       <- replace varibles passed in as init arg (new)
    //  %4 = %offset_dim0_iter + %c3                                                            <- replace gep of splat with add (already done)
    //  %subview = memref.subview %arg0, [%4][256][4] : memref<> -> memref<>                    <- generate subview on getelementptr (already done)
    //  ...
    //  scf.yield %4                                                                            <- replace yielding an gep output with the corresponding dim variable (new)
    // }
    // TODO: examples below are not supported since scf.for does not support returning a tensor type
    // Example 3, gep operand is a vector of i32, which is passed into the loop as variable, pointer updated using step, used after update
    //%_ptr3 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %1) -> (tensor<256xi32>) {
    //    // offset update
    //    %3 = tt.splat %c3 : i32 -> tensor<256xi32>
    //    %ptr_iter = arith.addi %3, %ptr : tensor<256xi32>
    //    // generate pointer
    //    %gep_ptr = tt.addptr %0, %ptr_iter : tensor<256x!tt.ptr<bf16>>
    //    // perform load
    //    %4 = tt.load %gep_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
    //    tt.store %gep_ptr, %4 : tensor<256x!tt.ptr<bf16>>
    //    scf.yield %ptr_iter : tensor<256xi32>
    //}
    // Expected output
    // %offset_dim0 = arith.constant 1024                                                       <- insert instructions to initialize init arg(new)
    // for iter_args (%offset_dim0_iter = %offset_dim0) {                                       <- replace varibles passed in as init arg (new)
    //  %4 = %offset_dim0_iter + %c3                                                            <- replace gep of splat with add (already done)
    //  %subview = memref.subview %arg0, [%offset_dim0_iter][256][4] : memref<> -> memref<>     <- generate subview on load (new)
    //  ...
    //  scf.yield %4                                                                            <- replace yielding an gep output with the corresponding dim variable (new)
    // }
    //// Example 4, gep operand is a vector of i32, which is passed into the loop as variable, pointer updated using step, used before update
    //%_ptr4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %1) -> (tensor<256xi32>) {
    //    // generate pointer
    //    %gep_ptr = tt.addptr %0, %ptr : tensor<256x!tt.ptr<bf16>>
    //
    //    // perform load
    //    %4 = tt.load %gep_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
    //    tt.store %gep_ptr, %4 : tensor<256x!tt.ptr<bf16>>
    //    // offset update
    //    %3 = tt.splat %c3 : i32 -> tensor<256xi32>
    //    %ptr_iter = arith.addi %3, %ptr : tensor<256xi32>
    //    scf.yield %ptr_iter : tensor<256xi32>
    //}
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
// CHECK:     %0 = scf.for %arg7 = %c0 to %c12 step %c3 iter_args(%arg8 = %c1024) -> (index) {
// CHECK:       %1 = arith.addi %arg8, %c3 : index
// CHECK:       %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
// CHECK:       %cast = memref.cast %reinterpret_cast : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16, strided<[?], offset: ?>>
// CHECK:       %alloc = memref.alloc() : memref<256xbf16>
// CHECK:       memref.copy %reinterpret_cast, %alloc : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16>
// CHECK:       %2 = bufferization.to_tensor %alloc restrict writable : memref<256xbf16> to tensor<256xbf16>
// CHECK:       bufferization.materialize_in_destination %2 in writable %cast : (tensor<256xbf16>, memref<256xbf16, strided<[?], offset: ?>>) -> ()
// CHECK:       scf.yield %1 : index
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
