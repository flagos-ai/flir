// RUN: triton-shared-opt --triton-to-linalg-incubated=named-ops=true --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-incubated="named-ops=true compile-on-910-95=true" --split-input-file %s | FileCheck %s --check-prefix=LEGACY --implicit-check-not=memref_ext.alloc_workspace
// REQUIRES: flir-incubated
// LEGACY: module


// -----

// CHECK-LABEL: func.func @rhs_bitcast(
// CHECK: arith.bitcast
// CHECK: memref_ext.alloc_workspace()
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @rhs_bitcast(%X: !tt.ptr<f32>, %W: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %Side: !tt.ptr<f32>) {
  %x_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %x_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %x_stride = arith.constant dense<64> : tensor<16xi32>
  %x_offset_r = arith.muli %x_r, %x_stride : tensor<16xi32>
  %x_re = tt.expand_dims %x_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %x_ce = tt.expand_dims %x_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %x_rb = tt.broadcast %x_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %x_cb = tt.broadcast %x_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %x_offset = arith.addi %x_rb, %x_cb : tensor<16x64xi32>
  %x_base = tt.splat %X : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %x_ptr = tt.addptr %x_base, %x_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %raw = tt.load %x_ptr : tensor<16x64x!tt.ptr<f32>>
  %two = arith.constant dense<2.0> : tensor<16x64xf32>
  %math = arith.mulf %raw, %two : tensor<16x64xf32>
  %bits = tt.bitcast %math : tensor<16x64xf32> -> tensor<16x64xi32>
  %mask = arith.constant dense<-8192> : tensor<16x64xi32>
  %integer = arith.andi %bits, %mask : tensor<16x64xi32>
  %computed = tt.bitcast %integer : tensor<16x64xi32> -> tensor<16x64xf32>
  %side_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %side_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %side_stride = arith.constant dense<64> : tensor<16xi32>
  %side_offset_r = arith.muli %side_r, %side_stride : tensor<16xi32>
  %side_re = tt.expand_dims %side_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %side_ce = tt.expand_dims %side_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %side_rb = tt.broadcast %side_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %side_cb = tt.broadcast %side_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %side_offset = arith.addi %side_rb, %side_cb : tensor<16x64xi32>
  %side_base = tt.splat %Side : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %side_ptr = tt.addptr %side_base, %side_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  tt.store %side_ptr, %computed : tensor<16x64x!tt.ptr<f32>>
  %w_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %w_c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %w_stride = arith.constant dense<16> : tensor<32xi32>
  %w_offset_r = arith.muli %w_r, %w_stride : tensor<32xi32>
  %w_re = tt.expand_dims %w_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %w_ce = tt.expand_dims %w_c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %w_rb = tt.broadcast %w_re : tensor<32x1xi32> -> tensor<32x16xi32>
  %w_cb = tt.broadcast %w_ce : tensor<1x16xi32> -> tensor<32x16xi32>
  %w_offset = arith.addi %w_rb, %w_cb : tensor<32x16xi32>
  %w_base = tt.splat %W : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>>
  %w_ptr = tt.addptr %w_base, %w_offset : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
  %weight = tt.load %w_ptr : tensor<32x16x!tt.ptr<f32>>
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %d0 = tt.dot %weight, %computed, %zero, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %d1 = tt.dot %weight, %computed, %d0, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %y_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %y_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %y_stride = arith.constant dense<64> : tensor<32xi32>
  %y_offset_r = arith.muli %y_r, %y_stride : tensor<32xi32>
  %y_re = tt.expand_dims %y_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %y_ce = tt.expand_dims %y_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %y_rb = tt.broadcast %y_re : tensor<32x1xi32> -> tensor<32x64xi32>
  %y_cb = tt.broadcast %y_ce : tensor<1x64xi32> -> tensor<32x64xi32>
  %y_offset = arith.addi %y_rb, %y_cb : tensor<32x64xi32>
  %y_base = tt.splat %Y : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
  %y_ptr = tt.addptr %y_base, %y_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
  tt.store %y_ptr, %d1 : tensor<32x64x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @lhs_bitcast_transpose(
// CHECK: arith.bitcast
// CHECK: memref_ext.alloc_workspace()
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @lhs_bitcast_transpose(%X: !tt.ptr<f32>, %W: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %Side: !tt.ptr<f32>) {
  %x_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %x_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %x_stride = arith.constant dense<64> : tensor<16xi32>
  %x_offset_r = arith.muli %x_r, %x_stride : tensor<16xi32>
  %x_re = tt.expand_dims %x_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %x_ce = tt.expand_dims %x_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %x_rb = tt.broadcast %x_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %x_cb = tt.broadcast %x_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %x_offset = arith.addi %x_rb, %x_cb : tensor<16x64xi32>
  %x_base = tt.splat %X : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %x_ptr = tt.addptr %x_base, %x_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %raw = tt.load %x_ptr : tensor<16x64x!tt.ptr<f32>>
  %two = arith.constant dense<2.0> : tensor<16x64xf32>
  %math = arith.mulf %raw, %two : tensor<16x64xf32>
  %bits = tt.bitcast %math : tensor<16x64xf32> -> tensor<16x64xi32>
  %mask = arith.constant dense<-8192> : tensor<16x64xi32>
  %integer = arith.andi %bits, %mask : tensor<16x64xi32>
  %computed = tt.bitcast %integer : tensor<16x64xi32> -> tensor<16x64xf32>
  %side_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %side_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %side_stride = arith.constant dense<64> : tensor<16xi32>
  %side_offset_r = arith.muli %side_r, %side_stride : tensor<16xi32>
  %side_re = tt.expand_dims %side_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %side_ce = tt.expand_dims %side_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %side_rb = tt.broadcast %side_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %side_cb = tt.broadcast %side_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %side_offset = arith.addi %side_rb, %side_cb : tensor<16x64xi32>
  %side_base = tt.splat %Side : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %side_ptr = tt.addptr %side_base, %side_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  tt.store %side_ptr, %computed : tensor<16x64x!tt.ptr<f32>>
  %w_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %w_c = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %w_stride = arith.constant dense<32> : tensor<16xi32>
  %w_offset_r = arith.muli %w_r, %w_stride : tensor<16xi32>
  %w_re = tt.expand_dims %w_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %w_ce = tt.expand_dims %w_c {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %w_rb = tt.broadcast %w_re : tensor<16x1xi32> -> tensor<16x32xi32>
  %w_cb = tt.broadcast %w_ce : tensor<1x32xi32> -> tensor<16x32xi32>
  %w_offset = arith.addi %w_rb, %w_cb : tensor<16x32xi32>
  %w_base = tt.splat %W : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
  %w_ptr = tt.addptr %w_base, %w_offset : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
  %weight = tt.load %w_ptr : tensor<16x32x!tt.ptr<f32>>
  %transposed = tt.trans %computed {order = array<i32: 1, 0>} : tensor<16x64xf32> -> tensor<64x16xf32>
  %zero = arith.constant dense<0.0> : tensor<64x32xf32>
  %d0 = tt.dot %transposed, %weight, %zero, inputPrecision = hf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>
  %d1 = tt.dot %transposed, %weight, %d0, inputPrecision = hf32 : tensor<64x16xf32> * tensor<16x32xf32> -> tensor<64x32xf32>
  %y_r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %y_c = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %y_stride = arith.constant dense<32> : tensor<64xi32>
  %y_offset_r = arith.muli %y_r, %y_stride : tensor<64xi32>
  %y_re = tt.expand_dims %y_offset_r {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %y_ce = tt.expand_dims %y_c {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %y_rb = tt.broadcast %y_re : tensor<64x1xi32> -> tensor<64x32xi32>
  %y_cb = tt.broadcast %y_ce : tensor<1x32xi32> -> tensor<64x32xi32>
  %y_offset = arith.addi %y_rb, %y_cb : tensor<64x32xi32>
  %y_base = tt.splat %Y : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>>
  %y_ptr = tt.addptr %y_base, %y_offset : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
  tt.store %y_ptr, %d1 : tensor<64x32x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @rhs_bitcast_add_zero(
// CHECK: arith.bitcast
// CHECK: memref_ext.alloc_workspace()
// CHECK: flagtree.vector_to_cube
// CHECK: hivm.hir.load
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {flagtree.auto_blockify = false, hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @rhs_bitcast_add_zero(%X: !tt.ptr<f32>, %W: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %Side: !tt.ptr<f32>) {
  %x_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %x_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %x_stride = arith.constant dense<64> : tensor<16xi32>
  %x_offset_r = arith.muli %x_r, %x_stride : tensor<16xi32>
  %x_re = tt.expand_dims %x_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %x_ce = tt.expand_dims %x_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %x_rb = tt.broadcast %x_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %x_cb = tt.broadcast %x_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %x_offset = arith.addi %x_rb, %x_cb : tensor<16x64xi32>
  %x_base = tt.splat %X : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %x_ptr = tt.addptr %x_base, %x_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %raw = tt.load %x_ptr : tensor<16x64x!tt.ptr<f32>>
  %two = arith.constant dense<2.0> : tensor<16x64xf32>
  %math = arith.mulf %raw, %two : tensor<16x64xf32>
  %bits = tt.bitcast %math : tensor<16x64xf32> -> tensor<16x64xi32>
  %mask = arith.constant dense<-8192> : tensor<16x64xi32>
  %integer = arith.andi %bits, %mask : tensor<16x64xi32>
  %computed = tt.bitcast %integer : tensor<16x64xi32> -> tensor<16x64xf32>
  %side_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %side_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %side_stride = arith.constant dense<64> : tensor<16xi32>
  %side_offset_r = arith.muli %side_r, %side_stride : tensor<16xi32>
  %side_re = tt.expand_dims %side_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %side_ce = tt.expand_dims %side_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %side_rb = tt.broadcast %side_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %side_cb = tt.broadcast %side_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %side_offset = arith.addi %side_rb, %side_cb : tensor<16x64xi32>
  %side_base = tt.splat %Side : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %side_ptr = tt.addptr %side_base, %side_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  tt.store %side_ptr, %computed : tensor<16x64x!tt.ptr<f32>>
  %w_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %w_c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %w_stride = arith.constant dense<16> : tensor<32xi32>
  %w_offset_r = arith.muli %w_r, %w_stride : tensor<32xi32>
  %w_re = tt.expand_dims %w_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %w_ce = tt.expand_dims %w_c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %w_rb = tt.broadcast %w_re : tensor<32x1xi32> -> tensor<32x16xi32>
  %w_cb = tt.broadcast %w_ce : tensor<1x16xi32> -> tensor<32x16xi32>
  %w_offset = arith.addi %w_rb, %w_cb : tensor<32x16xi32>
  %w_base = tt.splat %W : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>>
  %w_ptr = tt.addptr %w_base, %w_offset : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
  %weight = tt.load %w_ptr : tensor<32x16x!tt.ptr<f32>>
  %z = arith.constant dense<0.0> : tensor<16x64xf32>
  %materialized = arith.addf %computed, %z : tensor<16x64xf32>
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %d0 = tt.dot %weight, %materialized, %zero, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %d1 = tt.dot %weight, %materialized, %d0, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %y_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %y_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %y_stride = arith.constant dense<64> : tensor<32xi32>
  %y_offset_r = arith.muli %y_r, %y_stride : tensor<32xi32>
  %y_re = tt.expand_dims %y_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %y_ce = tt.expand_dims %y_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %y_rb = tt.broadcast %y_re : tensor<32x1xi32> -> tensor<32x64xi32>
  %y_cb = tt.broadcast %y_ce : tensor<1x64xi32> -> tensor<32x64xi32>
  %y_offset = arith.addi %y_rb, %y_cb : tensor<32x64xi32>
  %y_base = tt.splat %Y : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
  %y_ptr = tt.addptr %y_base, %y_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
  tt.store %y_ptr, %d1 : tensor<32x64x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @native_load(
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @native_load(%X: !tt.ptr<f32>, %W: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %Side: !tt.ptr<f32>) {
  %x_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %x_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %x_stride = arith.constant dense<64> : tensor<16xi32>
  %x_offset_r = arith.muli %x_r, %x_stride : tensor<16xi32>
  %x_re = tt.expand_dims %x_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %x_ce = tt.expand_dims %x_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %x_rb = tt.broadcast %x_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %x_cb = tt.broadcast %x_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %x_offset = arith.addi %x_rb, %x_cb : tensor<16x64xi32>
  %x_base = tt.splat %X : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %x_ptr = tt.addptr %x_base, %x_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %raw = tt.load %x_ptr : tensor<16x64x!tt.ptr<f32>>
  %computed = arith.addf %raw, %raw : tensor<16x64xf32>
  %side_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %side_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %side_stride = arith.constant dense<64> : tensor<16xi32>
  %side_offset_r = arith.muli %side_r, %side_stride : tensor<16xi32>
  %side_re = tt.expand_dims %side_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %side_ce = tt.expand_dims %side_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %side_rb = tt.broadcast %side_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %side_cb = tt.broadcast %side_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %side_offset = arith.addi %side_rb, %side_cb : tensor<16x64xi32>
  %side_base = tt.splat %Side : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %side_ptr = tt.addptr %side_base, %side_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  tt.store %side_ptr, %computed : tensor<16x64x!tt.ptr<f32>>
  %w_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %w_c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %w_stride = arith.constant dense<16> : tensor<32xi32>
  %w_offset_r = arith.muli %w_r, %w_stride : tensor<32xi32>
  %w_re = tt.expand_dims %w_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %w_ce = tt.expand_dims %w_c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %w_rb = tt.broadcast %w_re : tensor<32x1xi32> -> tensor<32x16xi32>
  %w_cb = tt.broadcast %w_ce : tensor<1x16xi32> -> tensor<32x16xi32>
  %w_offset = arith.addi %w_rb, %w_cb : tensor<32x16xi32>
  %w_base = tt.splat %W : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>>
  %w_ptr = tt.addptr %w_base, %w_offset : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
  %weight = tt.load %w_ptr : tensor<32x16x!tt.ptr<f32>>
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %d0 = tt.dot %weight, %computed, %zero, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %d1 = tt.dot %weight, %computed, %d0, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %y_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %y_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %y_stride = arith.constant dense<64> : tensor<32xi32>
  %y_offset_r = arith.muli %y_r, %y_stride : tensor<32xi32>
  %y_re = tt.expand_dims %y_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %y_ce = tt.expand_dims %y_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %y_rb = tt.broadcast %y_re : tensor<32x1xi32> -> tensor<32x64xi32>
  %y_cb = tt.broadcast %y_ce : tensor<1x64xi32> -> tensor<32x64xi32>
  %y_offset = arith.addi %y_rb, %y_cb : tensor<32x64xi32>
  %y_base = tt.splat %Y : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
  %y_ptr = tt.addptr %y_base, %y_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
  tt.store %y_ptr, %d1 : tensor<32x64x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @half_bitcast(
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @half_bitcast(%X: !tt.ptr<f16>, %W: !tt.ptr<f16>, %Y: !tt.ptr<f32>, %Side: !tt.ptr<f16>) {
  %x_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %x_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %x_stride = arith.constant dense<64> : tensor<16xi32>
  %x_offset_r = arith.muli %x_r, %x_stride : tensor<16xi32>
  %x_re = tt.expand_dims %x_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %x_ce = tt.expand_dims %x_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %x_rb = tt.broadcast %x_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %x_cb = tt.broadcast %x_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %x_offset = arith.addi %x_rb, %x_cb : tensor<16x64xi32>
  %x_base = tt.splat %X : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  %x_ptr = tt.addptr %x_base, %x_offset : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
  %raw = tt.load %x_ptr : tensor<16x64x!tt.ptr<f16>>
  %two = arith.constant dense<2.0> : tensor<16x64xf16>
  %math = arith.mulf %raw, %two : tensor<16x64xf16>
  %bits = tt.bitcast %math : tensor<16x64xf16> -> tensor<16x64xi16>
  %mask = arith.constant dense<-8192> : tensor<16x64xi16>
  %integer = arith.andi %bits, %mask : tensor<16x64xi16>
  %computed = tt.bitcast %integer : tensor<16x64xi16> -> tensor<16x64xf16>
  %side_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %side_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %side_stride = arith.constant dense<64> : tensor<16xi32>
  %side_offset_r = arith.muli %side_r, %side_stride : tensor<16xi32>
  %side_re = tt.expand_dims %side_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %side_ce = tt.expand_dims %side_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %side_rb = tt.broadcast %side_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %side_cb = tt.broadcast %side_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %side_offset = arith.addi %side_rb, %side_cb : tensor<16x64xi32>
  %side_base = tt.splat %Side : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  %side_ptr = tt.addptr %side_base, %side_offset : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
  tt.store %side_ptr, %computed : tensor<16x64x!tt.ptr<f16>>
  %w_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %w_c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %w_stride = arith.constant dense<16> : tensor<32xi32>
  %w_offset_r = arith.muli %w_r, %w_stride : tensor<32xi32>
  %w_re = tt.expand_dims %w_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %w_ce = tt.expand_dims %w_c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %w_rb = tt.broadcast %w_re : tensor<32x1xi32> -> tensor<32x16xi32>
  %w_cb = tt.broadcast %w_ce : tensor<1x16xi32> -> tensor<32x16xi32>
  %w_offset = arith.addi %w_rb, %w_cb : tensor<32x16xi32>
  %w_base = tt.splat %W : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>>
  %w_ptr = tt.addptr %w_base, %w_offset : tensor<32x16x!tt.ptr<f16>>, tensor<32x16xi32>
  %weight = tt.load %w_ptr : tensor<32x16x!tt.ptr<f16>>
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %d0 = tt.dot %weight, %computed, %zero, inputPrecision = hf32 : tensor<32x16xf16> * tensor<16x64xf16> -> tensor<32x64xf32>
  %d1 = tt.dot %weight, %computed, %d0, inputPrecision = hf32 : tensor<32x16xf16> * tensor<16x64xf16> -> tensor<32x64xf32>
  %y_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %y_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %y_stride = arith.constant dense<64> : tensor<32xi32>
  %y_offset_r = arith.muli %y_r, %y_stride : tensor<32xi32>
  %y_re = tt.expand_dims %y_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %y_ce = tt.expand_dims %y_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %y_rb = tt.broadcast %y_re : tensor<32x1xi32> -> tensor<32x64xi32>
  %y_cb = tt.broadcast %y_ce : tensor<1x64xi32> -> tensor<32x64xi32>
  %y_offset = arith.addi %y_rb, %y_cb : tensor<32x64xi32>
  %y_base = tt.splat %Y : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
  %y_ptr = tt.addptr %y_base, %y_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
  tt.store %y_ptr, %d1 : tensor<32x64x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @other_arch(
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: linalg.matmul
// CHECK-NOT: memref_ext.alloc_workspace
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910A">} {
 tt.func public @other_arch(%X: !tt.ptr<f32>, %W: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %Side: !tt.ptr<f32>) {
  %x_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %x_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %x_stride = arith.constant dense<64> : tensor<16xi32>
  %x_offset_r = arith.muli %x_r, %x_stride : tensor<16xi32>
  %x_re = tt.expand_dims %x_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %x_ce = tt.expand_dims %x_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %x_rb = tt.broadcast %x_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %x_cb = tt.broadcast %x_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %x_offset = arith.addi %x_rb, %x_cb : tensor<16x64xi32>
  %x_base = tt.splat %X : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %x_ptr = tt.addptr %x_base, %x_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %raw = tt.load %x_ptr : tensor<16x64x!tt.ptr<f32>>
  %two = arith.constant dense<2.0> : tensor<16x64xf32>
  %math = arith.mulf %raw, %two : tensor<16x64xf32>
  %bits = tt.bitcast %math : tensor<16x64xf32> -> tensor<16x64xi32>
  %mask = arith.constant dense<-8192> : tensor<16x64xi32>
  %integer = arith.andi %bits, %mask : tensor<16x64xi32>
  %computed = tt.bitcast %integer : tensor<16x64xi32> -> tensor<16x64xf32>
  %side_r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %side_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %side_stride = arith.constant dense<64> : tensor<16xi32>
  %side_offset_r = arith.muli %side_r, %side_stride : tensor<16xi32>
  %side_re = tt.expand_dims %side_offset_r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %side_ce = tt.expand_dims %side_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %side_rb = tt.broadcast %side_re : tensor<16x1xi32> -> tensor<16x64xi32>
  %side_cb = tt.broadcast %side_ce : tensor<1x64xi32> -> tensor<16x64xi32>
  %side_offset = arith.addi %side_rb, %side_cb : tensor<16x64xi32>
  %side_base = tt.splat %Side : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %side_ptr = tt.addptr %side_base, %side_offset : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  tt.store %side_ptr, %computed : tensor<16x64x!tt.ptr<f32>>
  %w_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %w_c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %w_stride = arith.constant dense<16> : tensor<32xi32>
  %w_offset_r = arith.muli %w_r, %w_stride : tensor<32xi32>
  %w_re = tt.expand_dims %w_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %w_ce = tt.expand_dims %w_c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %w_rb = tt.broadcast %w_re : tensor<32x1xi32> -> tensor<32x16xi32>
  %w_cb = tt.broadcast %w_ce : tensor<1x16xi32> -> tensor<32x16xi32>
  %w_offset = arith.addi %w_rb, %w_cb : tensor<32x16xi32>
  %w_base = tt.splat %W : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>>
  %w_ptr = tt.addptr %w_base, %w_offset : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
  %weight = tt.load %w_ptr : tensor<32x16x!tt.ptr<f32>>
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %d0 = tt.dot %weight, %computed, %zero, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %d1 = tt.dot %weight, %computed, %d0, inputPrecision = hf32 : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %y_r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %y_c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %y_stride = arith.constant dense<64> : tensor<32xi32>
  %y_offset_r = arith.muli %y_r, %y_stride : tensor<32xi32>
  %y_re = tt.expand_dims %y_offset_r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %y_ce = tt.expand_dims %y_c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %y_rb = tt.broadcast %y_re : tensor<32x1xi32> -> tensor<32x64xi32>
  %y_cb = tt.broadcast %y_ce : tensor<1x64xi32> -> tensor<32x64xi32>
  %y_offset = arith.addi %y_rb, %y_cb : tensor<32x64xi32>
  %y_base = tt.splat %Y : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
  %y_ptr = tt.addptr %y_base, %y_offset : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
  tt.store %y_ptr, %d1 : tensor<32x64x!tt.ptr<f32>>
  tt.return
 }
}
