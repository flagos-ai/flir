// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// The index is a carried value, not necessarily zero, and must retain i64.
// CHECK-LABEL: func.func @singleton_i64(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64{{[^,]*}}, %[[INDEX:[a-zA-Z0-9_]+]]: i64
// CHECK-NOT: tt.reduce
// CHECK: return %[[VALUE]], %[[INDEX]] : i64, i64
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @singleton_i64(%input: i64, %input_index: i64) -> (i64, i64) {
    %value = tt.splat %input : i64 -> tensor<1xi64>
    %index = tt.splat %input_index : i64 -> tensor<1xi64>
    %v, %i = "tt.reduce"(%value, %index) ({
    ^bb0(%a: i64, %ai: i64, %b: i64, %bi: i64):
      %better = arith.cmpi sgt, %a, %b : i64
      %rv = arith.select %better, %a, %b : i64
      %ri = arith.select %better, %ai, %bi : i64
      tt.reduce.return %rv, %ri : i64, i64
    }) {axis = 0 : i32} : (tensor<1xi64>, tensor<1xi64>) -> (i64, i64)
    tt.return %v, %i : i64, i64
  }
}

// -----

// Generic tuple reductions have no distinguished "index" operand.
// CHECK-LABEL: func.func @singleton_tuple(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: f32{{[^,]*}}, %[[Y:[a-zA-Z0-9_]+]]: i64{{[^,]*}}, %[[Z:[a-zA-Z0-9_]+]]: i32
// CHECK-NOT: tt.reduce
// CHECK: return %[[X]], %[[Y]], %[[Z]] : f32, i64, i32
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @singleton_tuple(%ix: f32, %iy: i64, %iz: i32) -> (f32, i64, i32) {
    %x = tt.splat %ix : f32 -> tensor<1xf32>
    %y = tt.splat %iy : i64 -> tensor<1xi64>
    %z = tt.splat %iz : i32 -> tensor<1xi32>
    %a, %b, %c = "tt.reduce"(%x, %y, %z) ({
    ^bb0(%a0: f32, %b0: i64, %c0: i32, %a1: f32, %b1: i64, %c1: i32):
      %ra = arith.addf %a0, %a1 : f32
      %rb = arith.addi %b0, %b1 : i64
      %rc = arith.addi %c0, %c1 : i32
      tt.reduce.return %ra, %rb, %rc : f32, i64, i32
    }) {axis = 0 : i32} : (tensor<1xf32>, tensor<1xi64>, tensor<1xi32>) -> (f32, i64, i32)
    tt.return %a, %b, %c : f32, i64, i32
  }
}

// -----

// CHECK-LABEL: func.func @singleton_rank2(
// CHECK-SAME: %[[X2:[a-zA-Z0-9_]+]]: f32{{[^,]*}}, %[[Y2:[a-zA-Z0-9_]+]]: i64
// CHECK-NOT: tt.reduce
// CHECK: %[[VX:[a-zA-Z0-9_]+]] = tensor.insert %[[X2]] into
// CHECK: %[[VY:[a-zA-Z0-9_]+]] = tensor.insert %[[Y2]] into
// CHECK: %[[CX:[a-zA-Z0-9_]+]] = tensor.collapse_shape %[[VX]]
// CHECK: %[[CY:[a-zA-Z0-9_]+]] = tensor.collapse_shape %[[VY]]
// CHECK: %[[EX:[a-zA-Z0-9_]+]] = tensor.extract %[[CX]]
// CHECK: %[[EY:[a-zA-Z0-9_]+]] = tensor.extract %[[CY]]
// CHECK: return %[[EX]], %[[EY]] : f32, i64
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @singleton_rank2(%ix: f32, %iy: i64) -> (f32, i64) {
    %x = tt.splat %ix : f32 -> tensor<1x1xf32>
    %y = tt.splat %iy : i64 -> tensor<1x1xi64>
    %a, %b = "tt.reduce"(%x, %y) ({
    ^bb0(%a0: f32, %b0: i64, %a1: f32, %b1: i64):
      %ra = arith.addf %a0, %a1 : f32
      %rb = arith.addi %b0, %b1 : i64
      tt.reduce.return %ra, %rb : f32, i64
    }) {axis = 1 : i32} : (tensor<1x1xf32>, tensor<1x1xi64>) -> (tensor<1xf32>, tensor<1xi64>)
    %zero = arith.constant 0 : index
    %a0 = tensor.extract %a[%zero] : tensor<1xf32>
    %b0 = tensor.extract %b[%zero] : tensor<1xi64>
    tt.return %a0, %b0 : f32, i64
  }
}
