// RUN: triton-shared-opt --triton-to-unstructure-incubated --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-unstructure-incubated --triton-to-linalg-incubated --split-input-file %s | FileCheck %s --check-prefix=LOWER
// REQUIRES: flir-incubated

// Four BF16 elements are eight bytes, not four i64 elements.
// CHECK-LABEL: tt.func public @constant_offset(
// CHECK: %[[EIGHT:.*]] = arith.constant 8 : i64
// CHECK: %[[BASE:.*]] = tt.ptr_to_int %{{.*}} : !tt.ptr<bf16> -> i64
// CHECK: %[[ADDR:.*]] = arith.addi %[[BASE]], %[[EIGHT]] : i64
// CHECK: %[[PTR:.*]] = tt.int_to_ptr %[[ADDR]] : i64 -> !tt.ptr<i64>
// CHECK: tt.load %[[PTR]]
// LOWER-LABEL: func.func @constant_offset(
// LOWER-NOT: tt.bitcast
// LOWER: hivm.hir.pointer_cast
// LOWER: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @constant_offset(%src: !tt.ptr<bf16>, %out: !tt.ptr<i64>) {
    %four = arith.constant 4 : i32
    %offset = tt.addptr %src, %four : !tt.ptr<bf16>, i32
    %cast = tt.bitcast %offset : !tt.ptr<bf16> -> !tt.ptr<i64>
    %value = tt.load %cast : !tt.ptr<i64>
    tt.store %out, %value : !tt.ptr<i64>
    tt.return
  }
}

// -----

// Sign-extend each source offset before multiplying by two; leave offsets
// applied after the cast in i64-element units.
// CHECK-LABEL: tt.func public @dynamic_chain(
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[BASE:.*]] = tt.ptr_to_int %{{.*}} : !tt.ptr<bf16> -> i64
// CHECK: %[[I64:.*]] = arith.extsi %{{.*}} : i32 to i64
// CHECK: %[[IB:.*]] = arith.muli %[[I64]], %[[TWO]]
// CHECK: %[[FIRST:.*]] = arith.addi %[[BASE]], %[[IB]]
// CHECK: %[[KB:.*]] = arith.muli %{{.*}}, %[[TWO]]
// CHECK: %[[ADDR:.*]] = arith.addi %[[FIRST]], %[[KB]]
// CHECK: %[[PTR:.*]] = tt.int_to_ptr %[[ADDR]]
// CHECK: %[[FINAL:.*]] = tt.addptr %[[PTR]], %{{.*}} : !tt.ptr<i64>, i32
// CHECK: tt.load %[[FINAL]]
// LOWER-LABEL: func.func @dynamic_chain(
// LOWER-NOT: tt.bitcast
// LOWER: hivm.hir.pointer_cast
// LOWER: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @dynamic_chain(
      %src: !tt.ptr<bf16>, %out: !tt.ptr<i64>, %i: i32, %k: i64, %j: i32) {
    %first = tt.addptr %src, %i : !tt.ptr<bf16>, i32
    %second = tt.addptr %first, %k : !tt.ptr<bf16>, i64
    %cast = tt.bitcast %second : !tt.ptr<bf16> -> !tt.ptr<i64>
    %final = tt.addptr %cast, %j : !tt.ptr<i64>, i32
    %value = tt.load %final : !tt.ptr<i64>
    tt.store %out, %value : !tt.ptr<i64>
    tt.return
  }
}

// -----

// A cast directly on the argument needs no element-offset scaling.
// CHECK-LABEL: tt.func public @base_store(
// CHECK: %[[BASE:.*]] = tt.ptr_to_int
// CHECK-NEXT: %[[PTR:.*]] = tt.int_to_ptr %[[BASE]] : i64 -> !tt.ptr<i64>
// CHECK: tt.store %[[PTR]]
// LOWER-LABEL: func.func @base_store(
// LOWER-NOT: tt.bitcast
// LOWER: hivm.hir.pointer_cast
// LOWER: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @base_store(%src: !tt.ptr<bf16>, %value: i64) {
    %cast = tt.bitcast %src : !tt.ptr<bf16> -> !tt.ptr<i64>
    tt.store %cast, %value : !tt.ptr<i64>
    tt.return
  }
}
