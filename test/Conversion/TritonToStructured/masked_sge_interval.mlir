// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

// Interval mask model: masks of the form `(offs - n) >= m` describe the valid
// region [max(m + n, 0), dims) — a suffix with a possibly dynamic offset.
// These must all stay structured (structured load/store) with mask offsets, instead of
// falling back to the unstructured gather/scatter path.

module {
  // n and m both runtime: offset and size are runtime expressions.
  tt.func @sge_dynamic_rhs_lhs(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>,
    %arg2: i32,
    %arg3: i32
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %n = tt.splat %arg2 : i32 -> tensor<256xi32>
    %m = tt.splat %arg3 : i32 -> tensor<256xi32>
    %idx = arith.subi %offs, %n : tensor<256xi32>
    %mask = arith.cmpi sge, %idx, %m : tensor<256xi32>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %idx : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %v = tt.load %ldptr_1, %mask, %cst : tensor<256x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %stptr_1, %v, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }

  // AND of a shifted sge (dynamic lower bound) and an slt (dynamic upper
  // bound): interval intersection, still one structured load.
  tt.func @sge_and_slt_interval(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>,
    %arg2: i32,
    %arg3: i32
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>
    %c0 = arith.constant dense<0> : tensor<256xi32>
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %n = tt.splat %arg2 : i32 -> tensor<256xi32>
    %M = tt.splat %arg3 : i32 -> tensor<256xi32>
    %idx = arith.subi %offs, %n : tensor<256xi32>
    %mlo = arith.cmpi sge, %idx, %c0 : tensor<256xi32>
    %mhi = arith.cmpi slt, %offs, %M : tensor<256xi32>
    %mask = arith.andi %mlo, %mhi : tensor<256xi1>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %idx : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %v = tt.load %ldptr_1, %mask, %cst : tensor<256x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %stptr_1, %v, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }

  // Empty interval: `(offs - 300) >= 0` over a 256-lane block is always
  // false. dim must fold to 0 (fill only, no copy).
  tt.func @sge_empty_interval(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf16>
    %c0 = arith.constant dense<0> : tensor<256xi32>
    %c300 = arith.constant dense<300> : tensor<256xi32>
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %idx = arith.subi %offs, %c300 : tensor<256xi32>
    %mask = arith.cmpi sge, %idx, %c0 : tensor<256xi32>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %idx : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %v = tt.load %ldptr_1, %mask, %cst : tensor<256x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %offs : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %stptr_1, %v, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }

  // 2D: per-dim offsets, combined with AND. dim0: (row - 2) >= 0 -> offset 2,
  // size 14; dim1: (col - 4) >= 0 -> offset 4, size 4.
  tt.func @sge_2d_offsets(
    %arg0: !tt.ptr<f16>,
    %arg1: !tt.ptr<f16>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf16>
    %c0_0 = arith.constant dense<0> : tensor<16x1xi32>
    %c0_1 = arith.constant dense<0> : tensor<1x8xi32>
    %c2 = arith.constant dense<2> : tensor<16x1xi32>
    %c4 = arith.constant dense<4> : tensor<1x8xi32>
    %r0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %e0 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %b0 = tt.broadcast %e0 : tensor<16x1xi32> -> tensor<16x8xi32>
    %s0 = arith.subi %e0, %c2 : tensor<16x1xi32>
    %m0 = arith.cmpi sge, %s0, %c0_0 : tensor<16x1xi32>
    %bm0 = tt.broadcast %m0 : tensor<16x1xi1> -> tensor<16x8xi1>
    %r1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %e1 = tt.expand_dims %r1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %b1 = tt.broadcast %e1 : tensor<1x8xi32> -> tensor<16x8xi32>
    %s1 = arith.subi %e1, %c4 : tensor<1x8xi32>
    %m1 = arith.cmpi sge, %s1, %c0_1 : tensor<1x8xi32>
    %bm1 = tt.broadcast %m1 : tensor<1x8xi1> -> tensor<16x8xi1>
    %mask = arith.andi %bm0, %bm1 : tensor<16x8xi1>
    %ldptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x8x!tt.ptr<f16>>
    %ldptr_1 = tt.addptr %ldptr, %b0 : tensor<16x8x!tt.ptr<f16>>, tensor<16x8xi32>
    %ldptr_2 = tt.addptr %ldptr_1, %b1 : tensor<16x8x!tt.ptr<f16>>, tensor<16x8xi32>
    %v = tt.load %ldptr_2, %mask, %cst : tensor<16x8x!tt.ptr<f16>>
    %stptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x8x!tt.ptr<f16>>
    %stptr_1 = tt.addptr %stptr, %b0 : tensor<16x8x!tt.ptr<f16>>, tensor<16x8xi32>
    %stptr_2 = tt.addptr %stptr_1, %b1 : tensor<16x8x!tt.ptr<f16>>, tensor<16x8xi32>
    tt.store %stptr_2, %v, %mask : tensor<16x8x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: tt.func @sge_dynamic_rhs_lhs
// CHECK: "tts.load"
// CHECK-SAME: static_mask_dims = array<i64: -9223372036854775808>
// CHECK-SAME: static_mask_offsets = array<i64: -9223372036854775808>
// CHECK-LABEL: tt.func @sge_and_slt_interval
// CHECK: "tts.load"
// CHECK-SAME: static_mask_dims = array<i64: -9223372036854775808>
// CHECK-SAME: static_mask_offsets = array<i64: -9223372036854775808>
// CHECK-LABEL: tt.func @sge_empty_interval
// CHECK: "tts.load"
// CHECK-SAME: static_mask_dims = array<i64: 0>
// CHECK-SAME: static_mask_offsets = array<i64: 0>
// CHECK-LABEL: tt.func @sge_2d_offsets
// CHECK: "tts.load"
// CHECK-SAME: static_mask_dims = array<i64: 14, 4>
// CHECK-SAME: static_mask_offsets = array<i64: 2, 4>
