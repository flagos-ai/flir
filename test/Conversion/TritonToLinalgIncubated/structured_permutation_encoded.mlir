// RUN: not triton-shared-opt --triton-to-linalg-incubated %s 2>&1 | FileCheck %s --implicit-check-not=custom_flagtree_
// REQUIRES: flir-incubated

// This Ascend pipeline consumes unencoded TTIR, not TTG distributed layouts.
// Retain its compile-time rejection for encoded inputs. In particular, private
// UB permutations must not erase a distributed encoding and silently admit it.
// CHECK: error: failed to legalize unresolved materialization
// CHECK: error: failed to apply Conversion Patterns
#parent = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
module attributes {hacc.target = #hacc.target<"Ascend910B4">, "ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  func.func @encoded_join(%a: tensor<32xf32, #slice>, %b: tensor<32xf32, #slice>) -> tensor<32x2xf32, #parent> {
    %y = tt.join %a, %b : tensor<32xf32, #slice> -> tensor<32x2xf32, #parent>
    return %y : tensor<32x2xf32, #parent>
  }
}
