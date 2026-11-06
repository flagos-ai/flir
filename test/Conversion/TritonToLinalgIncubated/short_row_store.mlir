// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @short_f32(
// CHECK: symbol = "custom_flagtree_short_row_load_{{.*}}"
// CHECK-COUNT-2: memref.alloc() : memref<256xi32, #hivm.address_space<ub>>
// CHECK: symbol = "custom_flagtree_short_row_store_float"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @short_f32(%out: !tt.ptr<f32>, %input: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<1024> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<32x4xi32>
  %off = arith.addi %rr, %cc : tensor<32x4xi32>
  %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %ip = tt.splat %input : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %iptr = tt.addptr %ip, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %loaded = tt.load %iptr : tensor<32x4x!tt.ptr<f32>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x4xf32>
  %value = arith.addf %loaded, %one : tensor<32x4xf32>
  tt.store %ptr, %value : tensor<32x4x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @short_f16(
// CHECK: symbol = "custom_flagtree_short_row_load_{{.*}}"
// CHECK-COUNT-2: memref.alloc() : memref<512xi32, #hivm.address_space<ub>>
// CHECK: symbol = "custom_flagtree_short_row_store_half"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @short_f16(%out: !tt.ptr<f16>, %input: !tt.ptr<f16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %s = arith.constant dense<512> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x8xi32>
  %cc = tt.broadcast %c1 : tensor<1x8xi32> -> tensor<32x8xi32>
  %off = arith.addi %rr, %cc : tensor<32x8xi32>
  %p = tt.splat %out : !tt.ptr<f16> -> tensor<32x8x!tt.ptr<f16>>
  %ptr = tt.addptr %p, %off : tensor<32x8x!tt.ptr<f16>>, tensor<32x8xi32>
  %ip = tt.splat %input : !tt.ptr<f16> -> tensor<32x8x!tt.ptr<f16>>
  %iptr = tt.addptr %ip, %off : tensor<32x8x!tt.ptr<f16>>, tensor<32x8xi32>
  %loaded = tt.load %iptr : tensor<32x8x!tt.ptr<f16>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x8xf16>
  %value = arith.addf %loaded, %one : tensor<32x8xf16>
  tt.store %ptr, %value : tensor<32x8x!tt.ptr<f16>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @short_bf16(
// CHECK: symbol = "custom_flagtree_short_row_load_{{.*}}"
// CHECK-COUNT-2: memref.alloc() : memref<512xi32, #hivm.address_space<ub>>
// CHECK: symbol = "custom_flagtree_short_row_store_bf16"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @short_bf16(%out: !tt.ptr<bf16>, %input: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<512> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<32x4xi32>
  %off = arith.addi %rr, %cc : tensor<32x4xi32>
  %p = tt.splat %out : !tt.ptr<bf16> -> tensor<32x4x!tt.ptr<bf16>>
  %ptr = tt.addptr %p, %off : tensor<32x4x!tt.ptr<bf16>>, tensor<32x4xi32>
  %ip = tt.splat %input : !tt.ptr<bf16> -> tensor<32x4x!tt.ptr<bf16>>
  %iptr = tt.addptr %ip, %off : tensor<32x4x!tt.ptr<bf16>>, tensor<32x4xi32>
  %loaded = tt.load %iptr : tensor<32x4x!tt.ptr<bf16>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x4xbf16>
  %value = arith.addf %loaded, %one : tensor<32x4xbf16>
  tt.store %ptr, %value : tensor<32x4x!tt.ptr<bf16>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @short_i32(
// CHECK: symbol = "custom_flagtree_short_row_load_{{.*}}"
// CHECK-COUNT-2: memref.alloc() : memref<128xi32, #hivm.address_space<ub>>
// CHECK: symbol = "custom_flagtree_short_row_store_int32"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @short_i32(%out: !tt.ptr<i32>, %input: !tt.ptr<i32>) {
  %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %c = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %s = arith.constant dense<127> : tensor<16xi32>
  %r0 = arith.muli %r, %s : tensor<16xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %rr = tt.broadcast %r1 : tensor<16x1xi32> -> tensor<16x2xi32>
  %cc = tt.broadcast %c1 : tensor<1x2xi32> -> tensor<16x2xi32>
  %off = arith.addi %rr, %cc : tensor<16x2xi32>
  %p = tt.splat %out : !tt.ptr<i32> -> tensor<16x2x!tt.ptr<i32>>
  %ptr = tt.addptr %p, %off : tensor<16x2x!tt.ptr<i32>>, tensor<16x2xi32>
  %ip = tt.splat %input : !tt.ptr<i32> -> tensor<16x2x!tt.ptr<i32>>
  %iptr = tt.addptr %ip, %off : tensor<16x2x!tt.ptr<i32>>, tensor<16x2xi32>
  %loaded = tt.load %iptr : tensor<16x2x!tt.ptr<i32>>
  %one = arith.constant dense<1> : tensor<16x2xi32>
  %value = arith.addi %loaded, %one : tensor<16x2xi32>
  tt.store %ptr, %value : tensor<16x2x!tt.ptr<i32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @short_i16(
// CHECK: symbol = "custom_flagtree_short_row_load_{{.*}}"
// CHECK-COUNT-2: memref.alloc() : memref<128xi32, #hivm.address_space<ub>>
// CHECK: symbol = "custom_flagtree_short_row_store_int16"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @short_i16(%out: !tt.ptr<i16>, %input: !tt.ptr<i16>) {
  %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<31> : tensor<8xi32>
  %r0 = arith.muli %r, %s : tensor<8xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<8x1xi32> -> tensor<8x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<8x4xi32>
  %off = arith.addi %rr, %cc : tensor<8x4xi32>
  %p = tt.splat %out : !tt.ptr<i16> -> tensor<8x4x!tt.ptr<i16>>
  %ptr = tt.addptr %p, %off : tensor<8x4x!tt.ptr<i16>>, tensor<8x4xi32>
  %ip = tt.splat %input : !tt.ptr<i16> -> tensor<8x4x!tt.ptr<i16>>
  %iptr = tt.addptr %ip, %off : tensor<8x4x!tt.ptr<i16>>, tensor<8x4xi32>
  %loaded = tt.load %iptr : tensor<8x4x!tt.ptr<i16>>
  %one = arith.constant dense<1> : tensor<8x4xi16>
  %value = arith.addi %loaded, %one : tensor<8x4xi16>
  tt.store %ptr, %value : tensor<8x4x!tt.ptr<i16>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @one_row(
// CHECK-NOT: custom_flagtree_short_row_store_
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @one_row(%out: !tt.ptr<f32>, %input: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 1 : i32} : tensor<1xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<1024> : tensor<1xi32>
  %r0 = arith.muli %r, %s : tensor<1xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<1x1xi32> -> tensor<1x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<1x4xi32>
  %off = arith.addi %rr, %cc : tensor<1x4xi32>
  %p = tt.splat %out : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
  %ip = tt.splat %input : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
  %iptr = tt.addptr %ip, %off : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
  %loaded = tt.load %iptr : tensor<1x4x!tt.ptr<f32>>
  %one = arith.constant dense<1.000000e+00> : tensor<1x4xf32>
  %value = arith.addf %loaded, %one : tensor<1x4xf32>
  tt.store %ptr, %value : tensor<1x4x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @full_block(
// CHECK-NOT: custom_flagtree_short_row_store_
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @full_block(%out: !tt.ptr<f32>, %input: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %s = arith.constant dense<1024> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x8xi32>
  %cc = tt.broadcast %c1 : tensor<1x8xi32> -> tensor<32x8xi32>
  %off = arith.addi %rr, %cc : tensor<32x8xi32>
  %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x8x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<32x8x!tt.ptr<f32>>, tensor<32x8xi32>
  %ip = tt.splat %input : !tt.ptr<f32> -> tensor<32x8x!tt.ptr<f32>>
  %iptr = tt.addptr %ip, %off : tensor<32x8x!tt.ptr<f32>>, tensor<32x8xi32>
  %loaded = tt.load %iptr : tensor<32x8x!tt.ptr<f32>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x8xf32>
  %value = arith.addf %loaded, %one : tensor<32x8xf32>
  tt.store %ptr, %value : tensor<32x8x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @unsupported_width(
// CHECK-NOT: custom_flagtree_short_row_store_
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @unsupported_width(%out: !tt.ptr<f64>, %input: !tt.ptr<f64>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<1024> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<32x4xi32>
  %off = arith.addi %rr, %cc : tensor<32x4xi32>
  %p = tt.splat %out : !tt.ptr<f64> -> tensor<32x4x!tt.ptr<f64>>
  %ptr = tt.addptr %p, %off : tensor<32x4x!tt.ptr<f64>>, tensor<32x4xi32>
  %ip = tt.splat %input : !tt.ptr<f64> -> tensor<32x4x!tt.ptr<f64>>
  %iptr = tt.addptr %ip, %off : tensor<32x4x!tt.ptr<f64>>, tensor<32x4xi32>
  %loaded = tt.load %iptr : tensor<32x4x!tt.ptr<f64>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x4xf64>
  %value = arith.addf %loaded, %one : tensor<32x4xf64>
  tt.store %ptr, %value : tensor<32x4x!tt.ptr<f64>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @different_architecture(
// CHECK-NOT: custom_flagtree_short_row_store_
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910A">} {
 tt.func public @different_architecture(%out: !tt.ptr<f32>, %input: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<1024> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<32x4xi32>
  %off = arith.addi %rr, %cc : tensor<32x4xi32>
  %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %ip = tt.splat %input : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %iptr = tt.addptr %ip, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %loaded = tt.load %iptr : tensor<32x4x!tt.ptr<f32>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x4xf32>
  %value = arith.addf %loaded, %one : tensor<32x4xf32>
  tt.store %ptr, %value : tensor<32x4x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @volatile_load(
// CHECK-NOT: custom_flagtree_short_row_load_
// CHECK: memref.copy
// CHECK-NOT: custom_flagtree_short_row_load_
// CHECK: symbol = "custom_flagtree_short_row_store_float"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @volatile_load(%out: !tt.ptr<f32>, %input: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<1024> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<32x4xi32>
  %off = arith.addi %rr, %cc : tensor<32x4xi32>
  %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %ip = tt.splat %input : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %iptr = tt.addptr %ip, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %loaded = tt.load %iptr {isVolatile = true} : tensor<32x4x!tt.ptr<f32>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x4xf32>
  %value = arith.addf %loaded, %one : tensor<32x4xf32>
  tt.store %ptr, %value : tensor<32x4x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @cached_load(
// CHECK-NOT: custom_flagtree_short_row_load_
// CHECK: memref.copy
// CHECK-NOT: custom_flagtree_short_row_load_
// CHECK: symbol = "custom_flagtree_short_row_store_float"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @cached_load(%out: !tt.ptr<f32>, %input: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<1024> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<32x4xi32>
  %off = arith.addi %rr, %cc : tensor<32x4xi32>
  %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %ip = tt.splat %input : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %iptr = tt.addptr %ip, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %loaded = tt.load %iptr {cache = 2 : i32} : tensor<32x4x!tt.ptr<f32>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x4xf32>
  %value = arith.addf %loaded, %one : tensor<32x4xf32>
  tt.store %ptr, %value : tensor<32x4x!tt.ptr<f32>>
  tt.return
 }
}

// -----

// CHECK-LABEL: func.func @eviction_hint_load(
// CHECK-NOT: custom_flagtree_short_row_load_
// CHECK: memref.copy
// CHECK-NOT: custom_flagtree_short_row_load_
// CHECK: symbol = "custom_flagtree_short_row_store_float"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 tt.func public @eviction_hint_load(%out: !tt.ptr<f32>, %input: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %s = arith.constant dense<1024> : tensor<32xi32>
  %r0 = arith.muli %r, %s : tensor<32xi32>
  %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c1 = tt.expand_dims %c {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %rr = tt.broadcast %r1 : tensor<32x1xi32> -> tensor<32x4xi32>
  %cc = tt.broadcast %c1 : tensor<1x4xi32> -> tensor<32x4xi32>
  %off = arith.addi %rr, %cc : tensor<32x4xi32>
  %p = tt.splat %out : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %ptr = tt.addptr %p, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %ip = tt.splat %input : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
  %iptr = tt.addptr %ip, %off : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
  %loaded = tt.load %iptr {evict = 2 : i32} : tensor<32x4x!tt.ptr<f32>>
  %one = arith.constant dense<1.000000e+00> : tensor<32x4xf32>
  %value = arith.addf %loaded, %one : tensor<32x4xf32>
  tt.store %ptr, %value : tensor<32x4x!tt.ptr<f32>>
  tt.return
 }
}
