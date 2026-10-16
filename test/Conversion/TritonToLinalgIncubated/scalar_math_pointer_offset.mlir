// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// Scalar address math must not become a tensor<1xf32> operation: use analysis
// marks that operation MetaUse and erases it, leaving a live tensor.extract.
// CHECK-LABEL: func.func @scalar_offset(
// CHECK: %[[F:.*]] = arith.sitofp
// CHECK: %[[S:.*]] = arith.mulf %[[F]], %{{.*}} : f32
// CHECK-NEXT: %[[I:.*]] = arith.fptosi %[[S]] : f32 to i32
// CHECK-NEXT: %[[OFFSET:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK: memref.reinterpret_cast {{.*}} to offset: [%[[OFFSET]]], sizes: [1]
// CHECK-NOT: tensor.extract
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @scalar_offset(
      %table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %k: !tt.ptr<i32>) {
    %two = arith.constant 2.000000e+00 : f32
    %pid = tt.get_program_id x : i32
    %kptr = tt.addptr %k, %pid : !tt.ptr<i32>, i32
    %kval = tt.load %kptr : !tt.ptr<i32>
    %f = arith.sitofp %kval : i32 to f32
    %scaled = arith.mulf %f, %two : f32
    %idx = arith.fptosi %scaled : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %value = tt.load %ptr : !tt.ptr<f32>
    %optr = tt.addptr %out, %pid : !tt.ptr<f32>, i32
    tt.store %optr, %value : !tt.ptr<f32>
    tt.return
  }
}

// -----

// Unlike scalar_offset, there is no scalar tt.addptr using %idx. The traversal
// must cross tt.splat and tensor arith.addi to reach the vector pointer offset.
// CHECK-LABEL: func.func @vector_offset(
// CHECK: %[[VF:.*]] = arith.sitofp
// CHECK: %[[VS:.*]] = arith.mulf %[[VF]], %{{.*}} : f32
// CHECK-NEXT: %[[VI:.*]] = arith.fptosi %[[VS]] : f32 to i32
// CHECK-NEXT: %[[VOFFSET:.*]] = arith.index_cast %[[VI]] : i32 to index
// CHECK: memref.reinterpret_cast {{.*}} to offset: [%[[VOFFSET]]], sizes: [16], strides: [1]
// CHECK-NOT: tensor.extract
// CHECK: memref.copy
// CHECK-NOT: tensor.extract
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @vector_offset(
      %table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %k: !tt.ptr<i32>) {
    %two = arith.constant 2.000000e+00 : f32
    %pid = tt.get_program_id x : i32
    %kptr = tt.addptr %k, %pid : !tt.ptr<i32>, i32
    %kval = tt.load %kptr : !tt.ptr<i32>
    %f = arith.sitofp %kval : i32 to f32
    %scaled = arith.mulf %f, %two : f32
    %idx = arith.fptosi %scaled : f32 to i32
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %idxv = tt.splat %idx : i32 -> tensor<16xi32>
    %offsets = arith.addi %idxv, %range : tensor<16xi32>
    %tablev = tt.splat %table : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %ptr = tt.addptr %tablev, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %value = tt.load %ptr : tensor<16x!tt.ptr<f32>>
    %outv = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %optr = tt.addptr %outv, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %optr, %value : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Data-only math must still be canonicalized. tensor.insert alone is not a
// sufficient check: scalar store lowering also creates one without this rewrite.
// CHECK-LABEL: func.func @scalar_data(
// CHECK: %[[PRODUCT:.*]] = linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: linalg.yield %[[MUL]] : f32
// CHECK-NEXT: } -> tensor<1xf32>
// CHECK-NEXT: %[[EXTRACT:.*]] = tensor.extract %[[PRODUCT]][%{{.*}}] : tensor<1xf32>
// CHECK: tensor.insert %[[EXTRACT]]
// CHECK: bufferization.materialize_in_destination
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @scalar_data(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %two = arith.constant 2.000000e+00 : f32
    %pid = tt.get_program_id x : i32
    %iptr = tt.addptr %in, %pid : !tt.ptr<f32>, i32
    %value = tt.load %iptr : !tt.ptr<f32>
    %scaled = arith.mulf %value, %two : f32
    %optr = tt.addptr %out, %pid : !tt.ptr<f32>, i32
    tt.store %optr, %scaled : !tt.ptr<f32>
    tt.return
  }
}

// -----

// A value may feed both a pointer offset and stored data. One offset use is
// enough to keep it scalar; do not require that all users are address users.
// CHECK-LABEL: func.func @mixed_use(
// CHECK: %[[MF:.*]] = arith.sitofp
// CHECK: %[[MS:.*]] = arith.mulf %[[MF]], %{{.*}} : f32
// CHECK: arith.fptosi %[[MS]] : f32 to i32
// CHECK: tensor.insert %[[MS]]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @mixed_use(
      %table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %data: !tt.ptr<f32>, %k: !tt.ptr<i32>) {
    %two = arith.constant 2.000000e+00 : f32
    %pid = tt.get_program_id x : i32
    %kptr = tt.addptr %k, %pid : !tt.ptr<i32>, i32
    %kval = tt.load %kptr : !tt.ptr<i32>
    %f = arith.sitofp %kval : i32 to f32
    %scaled = arith.mulf %f, %two : f32
    %idx = arith.fptosi %scaled : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %value = tt.load %ptr : !tt.ptr<f32>
    %optr = tt.addptr %out, %pid : !tt.ptr<f32>, i32
    tt.store %optr, %value : !tt.ptr<f32>
    %dptr = tt.addptr %data, %pid : !tt.ptr<f32>, i32
    tt.store %dptr, %scaled : !tt.ptr<f32>
    tt.return
  }
}
