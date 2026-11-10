// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @ceil_offset(
// CHECK: linalg.generic
// CHECK: math.ceil
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @ceil_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.ceil %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @floor_offset(
// CHECK: linalg.generic
// CHECK: math.floor
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @floor_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.floor %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @exp_offset(
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @exp_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.exp %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @exp2_offset(
// CHECK: linalg.generic
// CHECK: math.exp2
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @exp2_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.exp2 %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @log2_offset(
// CHECK: linalg.generic
// CHECK: math.log2
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @log2_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.log2 %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @sin_offset(
// CHECK: linalg.generic
// CHECK: math.sin
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @sin_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.sin %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @cos_offset(
// CHECK: linalg.generic
// CHECK: math.cos
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @cos_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.cos %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @erf_offset(
// CHECK: linalg.generic
// CHECK: math.erf
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: memref.reinterpret_cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @erf_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
    %r = math.erf %x : f32
    %idx = arith.fptosi %r : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %v = tt.load %ptr : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}
