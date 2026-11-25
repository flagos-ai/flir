// RUN: triton-opt --pass-pipeline='builtin.module(discrete-mask-access-conversion,triton-to-structured-incubated,triton-to-unstructure-incubated,triton-to-structured-incubated,triton-to-linalg-incubated{named-ops=true})' %s | FileCheck %s
// REQUIRES: flir-incubated

// The index clamp is tensor arithmetic, even though the pointer has one lane.
// The load must stay inside the mask condition because an inactive offset
// need not be in bounds.
// CHECK-LABEL: func.func @singleton_indirect_load
// CHECK: arith.maxsi
// CHECK-NOT: memref.load
// CHECK: scf.if {{.*}} -> (i32) {
// CHECK: %[[LOADED:.*]] = memref.load
// CHECK-NEXT: scf.yield %[[LOADED]] : i32
// CHECK-NEXT: } else {
// CHECK-NOT: memref.load
// CHECK: scf.yield {{.*}} : i32
// CHECK-NEXT: }
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @singleton_indirect_load(%input: !tt.ptr<i32>,
                                          %mapping: !tt.ptr<i32>,
                                          %output: !tt.ptr<i32>) {
    %zero = arith.constant dense<0> : tensor<1xi32>
    %indices = tt.splat %mapping : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %index = tt.load %indices : tensor<1x!tt.ptr<i32>>
    %mask = arith.cmpi sge, %index, %zero : tensor<1xi32>
    %clamped = arith.maxsi %index, %zero : tensor<1xi32>
    %base = tt.splat %input : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %ptr = tt.addptr %base, %clamped : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %loaded = tt.load %ptr, %mask, %zero : tensor<1x!tt.ptr<i32>>
    %out = tt.splat %output : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    tt.store %out, %loaded : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// Indirect vector loads also need to preserve the per-lane mask while the
// pointer is scalarized. Invalid gathered indices must not trigger GM reads.
// CHECK-LABEL: func.func @masked_vector_indirect_load
// CHECK: scf.for
// CHECK: scf.if
// CHECK: memref.load
// CHECK: scf.yield
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @masked_vector_indirect_load(%input: !tt.ptr<i32>,
                                              %indices: !tt.ptr<i32>,
                                              %output: !tt.ptr<i32>) {
    %range = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %index_ptrs = tt.splat %indices : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %index_ptr = tt.addptr %index_ptrs, %range : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %index = tt.load %index_ptr : tensor<2x!tt.ptr<i32>>
    %zero = arith.constant dense<0> : tensor<2xi32>
    %limit = arith.constant dense<8> : tensor<2xi32>
    %nonnegative = arith.cmpi sge, %index, %zero : tensor<2xi32>
    %in_bounds = arith.cmpi slt, %index, %limit : tensor<2xi32>
    %mask = arith.andi %nonnegative, %in_bounds : tensor<2xi1>
    %input_ptrs = tt.splat %input : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %ptr = tt.addptr %input_ptrs, %index : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %loaded = tt.load %ptr, %mask, %zero : tensor<2x!tt.ptr<i32>>
    %output_ptrs = tt.splat %output : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %output_ptr = tt.addptr %output_ptrs, %range : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    tt.store %output_ptr, %loaded : tensor<2x!tt.ptr<i32>>
    tt.return
  }
}
