// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// A one-element mask can be shared by the load providing an indirect address
// and the atomic using that address. Both memory accesses must retain its
// runtime extent, including the zero-length case.
// CHECK-LABEL: func.func @shared_load_atomic_mask(
// CHECK: %[[EXTENT:.*]] = arith.index_castui %{{.*}} : i1 to index
// CHECK: memref.subview {{.*}}[0] [%[[EXTENT]]] [1]
// CHECK: memref.copy
// CHECK: memref.subview {{.*}}[0] [%[[EXTENT]]] [1]
// CHECK: tensor.extract_slice {{.*}}[0] [%[[EXTENT]]] [1]
// CHECK: hivm.hir.store
// CHECK-SAME: atomic = <add>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @shared_load_atomic_mask(%ids: !tt.ptr<i32>, %out: !tt.ptr<i32>, %active: i1) {
    %zero = arith.constant dense<0> : tensor<1xi32>
    %one = arith.constant dense<1> : tensor<1xi32>
    %mask = tt.splat %active : i1 -> tensor<1xi1>
    %ids_vec = tt.splat %ids : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %index = tt.load %ids_vec, %mask, %zero : tensor<1x!tt.ptr<i32>>
    %out_vec = tt.splat %out : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %ptr = tt.addptr %out_vec, %index : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %old = tt.atomic_rmw add, acq_rel, gpu, %ptr, %one, %mask
      : (tensor<1x!tt.ptr<i32>>, tensor<1xi32>, tensor<1xi1>) -> tensor<1xi32>
    tt.return
  }
}

// -----

// The scalar predicate also gates an entire vector when its only user is an
// atomic. The extent must depend on the predicate rather than remain 8.
// CHECK-LABEL: func.func @atomic_only_splat_mask(
// CHECK: arith.index_castui %{{.*}} : i1 to index
// CHECK: memref.subview {{.*}}[%{{.*}}] [1]
// CHECK: tensor.extract_slice
// CHECK: hivm.hir.store
// CHECK-SAME: atomic = <add>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @atomic_only_splat_mask(%out: !tt.ptr<i32>, %active: i1) {
    %one = arith.constant dense<1> : tensor<8xi32>
    %offset = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %mask = tt.splat %active : i1 -> tensor<8xi1>
    %out_vec = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %out_vec, %offset : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %old = tt.atomic_rmw add, acq_rel, gpu, %ptr, %one, %mask
      : (tensor<8x!tt.ptr<i32>>, tensor<8xi32>, tensor<8xi1>) -> tensor<8xi32>
    tt.return
  }
}
