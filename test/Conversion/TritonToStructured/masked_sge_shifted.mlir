// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

// Mask `(offs - 128) >= 0` has a non-zero lower bound: the valid region is
// the suffix [128, 256). The interval mask model expresses it as offset 128,
// size 128, so the load/store must be rewritten to structured tts ops with
// both mask dims and mask offsets instead of falling back to the
// unstructured gather/scatter path.

module {
  tt.func @masked_shifted_sge_load(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>
    %c0 = arith.constant dense<0> : tensor<256xi32>
    %c128 = arith.constant dense<128> : tensor<256xi32>
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %idx = arith.subi %offs, %c128 : tensor<256xi32>
    %mask = arith.cmpi sge, %idx, %c0 : tensor<256xi32>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %idx : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %v = tt.load %ldptr_1, %mask, %cst : tensor<256x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %stptr_1, %v, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }

  // Control: `offs >= 0` has lower bound 0, valid region is the full prefix.
  tt.func @masked_plain_sge_load(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>
    %c0 = arith.constant dense<0> : tensor<256xi32>
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %mask = arith.cmpi sge, %offs, %c0 : tensor<256xi32>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %v = tt.load %ldptr_1, %mask, %cst : tensor<256x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %stptr_1, %v, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: tt.func @masked_shifted_sge_load
// CHECK: "tts.load"
// CHECK-SAME: static_mask_dims = array<i64: 128>
// CHECK-SAME: static_mask_offsets = array<i64: 128>
// CHECK: "tts.store"
// CHECK-SAME: static_mask_dims = array<i64: 128>
// CHECK-SAME: static_mask_offsets = array<i64: 128>
// CHECK-LABEL: tt.func @masked_plain_sge_load
// CHECK: "tts.load"
// CHECK-SAME: static_mask_dims = array<i64: 256>
// CHECK-SAME: static_mask_offsets = array<i64: 0>
