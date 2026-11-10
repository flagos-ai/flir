// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// Scalar rounding must stay in the tensor path, without an integer round-trip.
// CHECK-LABEL: func.func @ceil_offset(
// CHECK: linalg.generic
// CHECK: math.ceil
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @ceil_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %half = arith.constant 5.000000e-01 : f32
    %pid = tt.get_program_id x : i32
    %f = arith.sitofp %pid : i32 to f32
    %scaled = arith.mulf %f, %half : f32
    %rounded = math.ceil %scaled : f32
    %idx = arith.fptosi %rounded : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %value = tt.load %ptr : !tt.ptr<f32>
    %optr = tt.addptr %out, %pid : !tt.ptr<f32>, i32
    tt.store %optr, %value : !tt.ptr<f32>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @floor_offset(
// CHECK: linalg.generic
// CHECK: math.floor
// CHECK: tensor.extract
// CHECK: arith.fptosi
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @floor_offset(%table: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %half = arith.constant 5.000000e-01 : f32
    %pid = tt.get_program_id x : i32
    %f = arith.sitofp %pid : i32 to f32
    %scaled = arith.mulf %f, %half : f32
    %rounded = math.floor %scaled : f32
    %idx = arith.fptosi %rounded : f32 to i32
    %ptr = tt.addptr %table, %idx : !tt.ptr<f32>, i32
    %value = tt.load %ptr : !tt.ptr<f32>
    %optr = tt.addptr %out, %pid : !tt.ptr<f32>, i32
    tt.store %optr, %value : !tt.ptr<f32>
    tt.return
  }
}

// -----

// Rounding on data, not on an address, must keep using the device library
// implementation: it is vectorized into a linalg body and links fine.
// CHECK-LABEL: func.func @ceil_data(
// CHECK: linalg.generic
// CHECK: math.ceil
// CHECK-NOT: arith.select
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @ceil_data(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %inv = tt.splat %in : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %iptr = tt.addptr %inv, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %value = tt.load %iptr : tensor<16x!tt.ptr<f32>>
    %rounded = math.ceil %value : tensor<16xf32>
    %outv = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %optr = tt.addptr %outv, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %optr, %rounded : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
