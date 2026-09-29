//===----------------------------------------------------------------------===//
// Passes.td - LaneVectorize pass definition -------------------------------===//
//
// This pass lives in lib/Transforms/LaneVectorize and is tablegenned with
// `-name LaneVectorize` (see CMakeLists.txt), so the generated registration
// function is registerLaneVectorizePasses(). The pass name, its `...Base`
// class and the create<Name>() factory keep the Triton prefix that comes from
// the def name. See lib/Transforms/LaneVectorize/LaneVectorize.cpp.
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_TRANSFORMS_LANE_VECTORIZE_PASSES
#define TRITON_SHARED_TRANSFORMS_LANE_VECTORIZE_PASSES

include "mlir/Pass/PassBase.td"

def TritonLaneVectorize : Pass<"triton-lane-vectorize", "mlir::ModuleOp"> {
  let summary = "Recognize lane-parallel tensor update patterns";
  let description = [{
    This pass recognizes groups of Triton tensors that represent independent
    lanes of the same computation and packs them into a single tensor with a
    synthesized leading lane dimension (Superword Level Parallelism). It is
    the TTIR implementation in lib/Transforms/LaneVectorize/LaneVectorize.cpp.

    Example (2 lanes of tensor<4xf32>):

    ```
    %0:2 = scf.for %iv = %lb to %ub step %step
             iter_args(%lane0 = %arg0, %lane1 = %arg1)
             -> (tensor<4xf32>, tensor<4xf32>) {
      %r0 = arith.addf %lane0, %y0 : tensor<4xf32>
      %r1 = arith.addf %lane1, %y1 : tensor<4xf32>
      scf.yield %r0, %r1 : tensor<4xf32>, tensor<4xf32>
    }
    ```

    becomes

    ```
    %pinit = tensor.concat dim(0)
               (tensor.reshape %arg0 -> 1x4), (tensor.reshape %arg1 -> 1x4)
               : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
    %0 = scf.for %iv = %lb to %ub step %step
           iter_args(%p = %pinit) -> (tensor<2x4xf32>) {
      %yb = tt.broadcast ... : tensor<2x4xf32>
      %rp = arith.addf %p, %yb : tensor<2x4xf32>
      scf.yield %rp : tensor<2x4xf32>
    }
    ```

    Two modes share one packer: loop mode packs the same-typed tensor
    iter_args of an scf.for and replays its body once; block mode (SLP)
    discovers a lane cone in straight-line code, seeded by a tensor.concat
    packing boundary or by shallow structural signatures refined by operand
    congruence.

    Matching is structural rather than semantic: an operation is lifted when
    each of its per-lane instances computes the same math and its operands are
    either already packed or lane-invariant. Associative combines across the
    lanes (for example an add tree over all lanes) are recognized as reductions
    over the lane axis. Any operation whose packed form is mathematically
    equivalent is allowed; no semantic pattern is hard-coded.

    Effectful ops (loads/stores/copies/barriers), pointer/buffer/memref/token
    types, arith.select and tt.reshape with allow_reorder are deliberately NOT
    packed; integer/index cones that feed memory addresses are also rejected
    (see isAddressProducer). See the file header for the rationale.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::triton::TritonDialect"];
}

#endif // TRITON_SHARED_TRANSFORMS_LANE_VECTORIZE_PASSES
