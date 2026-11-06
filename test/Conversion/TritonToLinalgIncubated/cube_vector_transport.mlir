// RUN: triton-shared-opt --triton-to-linalg-incubated=named-ops=true --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-incubated="named-ops=true compile-on-910-95=true" --split-input-file %s | FileCheck %s --check-prefix=LEGACY --implicit-check-not=flagtree.cube_to_vector
// REQUIRES: flir-incubated
// LEGACY: module

// -----

// CHECK-LABEL: func.func @direct_split(
// CHECK: flagtree.cube_to_vector
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @direct_split(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> tensor<1024xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %flatshape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %dot(%flatshape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  return %left : tensor<1024xf32>
 }
}

// -----

// CHECK-LABEL: func.func @distinct_slices(
// CHECK: flagtree.cube_to_vector
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: tensor.extract_slice {{.*}}[0, 0] [16, 64]
// CHECK: tensor.extract_slice {{.*}}[16, 0] [16, 64]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @distinct_slices(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> (tensor<512xf32>, tensor<512xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %first = tensor.extract_slice %dot[0, 0] [16, 64] [1, 1] : tensor<32x64xf32> to tensor<16x64xf32>
  %second = tensor.extract_slice %dot[16, 0] [16, 64] [1, 1] : tensor<32x64xf32> to tensor<16x64xf32>
  %shape = arith.constant dense<[512, 2]> : tensor<2xi64>
  %f = tensor.reshape %first(%shape) : (tensor<16x64xf32>, tensor<2xi64>) -> tensor<512x2xf32>
  %s = tensor.reshape %second(%shape) : (tensor<16x64xf32>, tensor<2xi64>) -> tensor<512x2xf32>
  %fl, %fr = tt.split %f : tensor<512x2xf32> -> tensor<512xf32>
  %sl, %sr = tt.split %s : tensor<512x2xf32> -> tensor<512xf32>
  return %fl, %sr : tensor<512xf32>, tensor<512xf32>
 }
}

// -----

// CHECK-LABEL: func.func @join_bitcasts(
// CHECK: flagtree.cube_to_vector
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @join_bitcasts(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> tensor<1024x2xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %first = tensor.extract_slice %dot[0, 0] [16, 64] [1, 1] : tensor<32x64xf32> to tensor<16x64xf32>
  %second = tensor.extract_slice %dot[16, 0] [16, 64] [1, 1] : tensor<32x64xf32> to tensor<16x64xf32>
  %shape = arith.constant dense<1024> : tensor<1xi64>
  %f = tensor.reshape %first(%shape) : (tensor<16x64xf32>, tensor<1xi64>) -> tensor<1024xf32>
  %s = tensor.reshape %second(%shape) : (tensor<16x64xf32>, tensor<1xi64>) -> tensor<1024xf32>
  %joined = tt.join %f, %s : tensor<1024xf32> -> tensor<1024x2xf32>
  return %joined : tensor<1024x2xf32>
 }
}

// -----

// CHECK-LABEL: func.func @shared_accumulator(
// CHECK: %[[DOT:[a-zA-Z0-9_]+]] = linalg.matmul
// CHECK: flagtree.cube_to_vector
// CHECK: linalg.matmul {{.*}}outs(%[[DOT]] : tensor<32x64xf32>)
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @shared_accumulator(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> (tensor<1024xf32>, tensor<32x64xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %flatshape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %dot(%flatshape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  %next = tt.dot %a, %b, %dot, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  return %left, %next : tensor<1024xf32>, tensor<32x64xf32>
 }
}

// -----

// CHECK-LABEL: func.func @direct_gm_side(
// CHECK: %[[DOT:[a-zA-Z0-9_]+]] = linalg.matmul
// CHECK: flagtree.cube_to_vector
// CHECK: return {{.*}}, %[[DOT]] :
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @direct_gm_side(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> (tensor<1024xf32>, tensor<32x64xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %flatshape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %dot(%flatshape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  return %left, %dot : tensor<1024xf32>, tensor<32x64xf32>
 }
}

// -----

// CHECK-LABEL: func.func @foldable_elementwise(
// CHECK: flagtree.cube_to_vector
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @foldable_elementwise(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> tensor<1024xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %one = arith.constant dense<1.0> : tensor<32x64xf32>
  %math = arith.mulf %dot, %one : tensor<32x64xf32>
  %flatshape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %math(%flatshape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  return %left : tensor<1024xf32>
 }
}

// -----

// CHECK-LABEL: func.func @bitcast_view(
// CHECK: flagtree.cube_to_vector
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @bitcast_view(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> tensor<32x64xi32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %bits = arith.bitcast %dot : tensor<32x64xf32> to tensor<32x64xi32>
  return %bits : tensor<32x64xi32>
 }
}

// -----

// CHECK-LABEL: func.func @nested_consumer(
// CHECK: flagtree.cube_to_vector
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @nested_consumer(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %cond: i1) -> tensor<1024xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %answer = scf.if %cond -> (tensor<1024xf32>) {
  %flatshape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %dot(%flatshape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
    scf.yield %left : tensor<1024xf32>
  } else {
    %empty_result = tensor.empty() : tensor<1024xf32>
    scf.yield %empty_result : tensor<1024xf32>
  }
  return %answer : tensor<1024xf32>
 }
}

// -----

// CHECK-LABEL: func.func @plain_dot(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @plain_dot(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> tensor<32x64xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  return %dot : tensor<32x64xf32>
 }
}

// -----

// CHECK-LABEL: func.func @native_vector(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @native_vector(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>) -> tensor<32x64xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %other = arith.addf %dot, %dot : tensor<32x64xf32>
  return %other : tensor<32x64xf32>
 }
}

// -----

// CHECK-LABEL: func.func @batched_split(
// CHECK: flagtree.cube_to_vector
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @batched_split(%a: tensor<2x32x16xf32>, %b: tensor<2x16x64xf32>) -> tensor<2048xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<2x32x16xf32> * tensor<2x16x64xf32> -> tensor<2x32x64xf32>
  %flatshape = arith.constant dense<[2048, 2]> : tensor<2xi64>
  %flat = tensor.reshape %dot(%flatshape) : (tensor<2x32x64xf32>, tensor<2xi64>) -> tensor<2048x2xf32>
  %left, %right = tt.split %flat : tensor<2048x2xf32> -> tensor<2048xf32>
  return %left : tensor<2048xf32>
 }
}

// -----

// CHECK-LABEL: func.func @vector_only_split(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @vector_only_split(%input: tensor<32x64xf32>) -> tensor<1024xf32> {
  %shape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %view = tensor.reshape %input(%shape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %view : tensor<1024x2xf32> -> tensor<1024xf32>
  return %left : tensor<1024xf32>
 }
}

// -----

// CHECK-LABEL: func.func @cube_vector_cube(
// CHECK: flagtree.cube_to_vector
// CHECK: flagtree.vector_to_cube
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @cube_vector_cube(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %c: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %flatshape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %dot(%flatshape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  %shape = arith.constant dense<[32, 32]> : tensor<2xi64>
  %selected = tensor.reshape %left(%shape) : (tensor<1024xf32>, tensor<2xi64>) -> tensor<32x32xf32>
  %empty_out = tensor.empty() : tensor<32x16xf32>
  %init_out = linalg.fill ins(%zero : f32) outs(%empty_out : tensor<32x16xf32>) -> tensor<32x16xf32>
  %next = tt.dot %selected, %c, %init_out, inputPrecision = ieee : tensor<32x32xf32> * tensor<32x16xf32> -> tensor<32x16xf32>
  return %next : tensor<32x16xf32>
 }
}


// -----

// CHECK-LABEL: func.func @nested_producer_consumer(
// CHECK: scf.for
// CHECK: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @nested_producer_consumer(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %n: index) -> tensor<1024xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant dense<0.0> : tensor<1024xf32>
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%old = %init) -> (tensor<1024xf32>) {
  %zero = arith.constant dense<0.0> : tensor<32x64xf32>
  %dot = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x64xf32> -> tensor<32x64xf32>
  %shape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %dot(%shape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
   scf.yield %left : tensor<1024xf32>
  }
  return %result : tensor<1024xf32>
 }
}

// -----

// Shared native expressions must be visited once, even with no Cube source.
// CHECK-LABEL: func.func @shared_native_dag(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
 func.func @shared_native_dag(%input: tensor<32x64xf32>) -> tensor<1024xf32> {
  %node0 = arith.addf %input, %input : tensor<32x64xf32>
  %node1 = arith.addf %node0, %node0 : tensor<32x64xf32>
  %node2 = arith.addf %node1, %node1 : tensor<32x64xf32>
  %node3 = arith.addf %node2, %node2 : tensor<32x64xf32>
  %node4 = arith.addf %node3, %node3 : tensor<32x64xf32>
  %node5 = arith.addf %node4, %node4 : tensor<32x64xf32>
  %node6 = arith.addf %node5, %node5 : tensor<32x64xf32>
  %node7 = arith.addf %node6, %node6 : tensor<32x64xf32>
  %node8 = arith.addf %node7, %node7 : tensor<32x64xf32>
  %node9 = arith.addf %node8, %node8 : tensor<32x64xf32>
  %node10 = arith.addf %node9, %node9 : tensor<32x64xf32>
  %node11 = arith.addf %node10, %node10 : tensor<32x64xf32>
  %node12 = arith.addf %node11, %node11 : tensor<32x64xf32>
  %node13 = arith.addf %node12, %node12 : tensor<32x64xf32>
  %node14 = arith.addf %node13, %node13 : tensor<32x64xf32>
  %node15 = arith.addf %node14, %node14 : tensor<32x64xf32>
  %node16 = arith.addf %node15, %node15 : tensor<32x64xf32>
  %node17 = arith.addf %node16, %node16 : tensor<32x64xf32>
  %node18 = arith.addf %node17, %node17 : tensor<32x64xf32>
  %node19 = arith.addf %node18, %node18 : tensor<32x64xf32>
  %node20 = arith.addf %node19, %node19 : tensor<32x64xf32>
  %node21 = arith.addf %node20, %node20 : tensor<32x64xf32>
  %node22 = arith.addf %node21, %node21 : tensor<32x64xf32>
  %node23 = arith.addf %node22, %node22 : tensor<32x64xf32>
  %node24 = arith.addf %node23, %node23 : tensor<32x64xf32>
  %node25 = arith.addf %node24, %node24 : tensor<32x64xf32>
  %node26 = arith.addf %node25, %node25 : tensor<32x64xf32>
  %node27 = arith.addf %node26, %node26 : tensor<32x64xf32>
  %node28 = arith.addf %node27, %node27 : tensor<32x64xf32>
  %node29 = arith.addf %node28, %node28 : tensor<32x64xf32>
  %node30 = arith.addf %node29, %node29 : tensor<32x64xf32>
  %node31 = arith.addf %node30, %node30 : tensor<32x64xf32>
  %shape = arith.constant dense<[1024, 2]> : tensor<2xi64>
  %flat = tensor.reshape %node31(%shape) : (tensor<32x64xf32>, tensor<2xi64>) -> tensor<1024x2xf32>
  %left, %right = tt.split %flat : tensor<1024x2xf32> -> tensor<1024xf32>
  return %left : tensor<1024xf32>
 }
}
