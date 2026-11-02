// RUN: not triton-shared-opt --triton-to-linalg-incubated=named-ops=true --split-input-file %s 2>&1 | FileCheck %s
// REQUIRES: flir-incubated
// CHECK: Cube-to-Vector custom/bitcast operand forwarded through control flow
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @forwarded_if(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %cond: i1) -> tensor<1024xf32> {
  %result = scf.if %cond -> (tensor<32x64xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
   scf.yield %dot : tensor<32x64xf32>
  } else {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
   scf.yield %dot : tensor<32x64xf32>
  }
  %flatshape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %result(%flatshape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  return %left : tensor<1024xf32>
 }
}

// -----

// CHECK: Cube-to-Vector custom/bitcast operand forwarded through control flow
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @forwarded_for(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %n: index) -> tensor<1024xf32> {
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %dot = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant dense<0.0> : tensor<1024xf32>
  %result:2 = scf.for %i = %c0 to %n step %c1 iter_args(%carried = %dot, %old = %init) -> (tensor<32x64xf32>, tensor<1024xf32>) {
  %shape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %carried(%shape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
    %updated = tt.dot %a, %b, %carried, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
    scf.yield %updated, %left : tensor<32x64xf32>, tensor<1024xf32>
  }
  return %result#1 : tensor<1024xf32>
 }
}

// -----

// CHECK: Cube-to-Vector custom/bitcast operand forwarded through control flow
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @forwarded_while(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %cond: i1) -> tensor<1024xf32> {
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %dot = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %result = scf.while (%before = %dot) : (tensor<32x64xf32>) -> tensor<32x64xf32> {
    scf.condition(%cond) %before : tensor<32x64xf32>
  } do {
  ^bb0(%after: tensor<32x64xf32>):
    %updated = tt.dot %a, %b, %after, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
    scf.yield %updated : tensor<32x64xf32>
  }
  %shape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %result(%shape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  return %left : tensor<1024xf32>
 }
}
