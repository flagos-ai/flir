// RUN: triton-shared-opt \
// RUN:   --triton-to-structured \
// RUN:   --triton-to-unstructured \
// RUN:   %s | FileCheck %s

// ============================================================
// (1) Scalar FADD – all-true mask → 无条件执行，无 scf.if
// ============================================================
// CHECK-LABEL: func @scalar_fadd_true_mask
// CHECK-NOT:     tt.atomic_rmw
// CHECK:         memref.load
// CHECK:         arith.addf
// CHECK:         memref.store
// CHECK-NOT:     scf.if
tt.func @scalar_fadd_true_mask(%ptr: !tt.ptr<f32>, %val: f32) -> f32 {
  %mask = arith.constant true
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val, %mask
    : (!tt.ptr<f32>, f32, i1) -> f32
  tt.return %old : f32
}

// ============================================================
// (2) Scalar FADD – all-false mask → op 被消除，不产生访存
// ============================================================
// CHECK-LABEL: func @scalar_fadd_false_mask
// CHECK-NOT:     tt.atomic_rmw
// CHECK-NOT:     memref.load
// CHECK-NOT:     memref.store
tt.func @scalar_fadd_false_mask(%ptr: !tt.ptr<f32>, %val: f32) -> f32 {
  %mask = arith.constant false
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val, %mask
    : (!tt.ptr<f32>, f32, i1) -> f32
  tt.return %old : f32
}

// ============================================================
// (3) Scalar FADD – runtime mask → 有 scf.if 保护 store
// ============================================================
// CHECK-LABEL: func @scalar_fadd_runtime_mask
// CHECK-NOT:     tt.atomic_rmw
// CHECK:         memref.load
// CHECK:         scf.if
// CHECK:           arith.addf
// CHECK:           memref.store
tt.func @scalar_fadd_runtime_mask(%ptr: !tt.ptr<f32>, %val: f32, %mask: i1) -> f32 {
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val, %mask
    : (!tt.ptr<f32>, f32, i1) -> f32
  tt.return %old : f32
}

// ============================================================
// (4) Tensor FADD – all-true mask → linalg.generic
// ============================================================
// CHECK-LABEL: func @tensor_fadd
// CHECK-NOT:     tt.atomic_rmw
// CHECK:         linalg.generic
// CHECK:           arith.addf
// CHECK:           linalg.yield
tt.func @tensor_fadd(%ptr: tensor<16x!tt.ptr<f32>>,
                     %val: tensor<16xf32>) -> tensor<16xf32> {
  %mask = arith.constant dense<true> : tensor<16xi1>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val, %mask
    : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  tt.return %old : tensor<16xf32>
}

// ============================================================
// (5) Tensor FADD – partial mask → linalg.generic + arith.select
// ============================================================
// CHECK-LABEL: func @tensor_fadd_partial_mask
// CHECK-NOT:     tt.atomic_rmw
// CHECK:         linalg.generic
// CHECK:           arith.addf
// CHECK:           arith.select
// CHECK:           linalg.yield
tt.func @tensor_fadd_partial_mask(%ptr: tensor<16x!tt.ptr<f32>>,
                                  %val: tensor<16xf32>,
                                  %mask: tensor<16xi1>) -> tensor<16xf32> {
  %old = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val, %mask
    : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  tt.return %old : tensor<16xf32>
}
