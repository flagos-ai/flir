#ifndef TRITON_SHARED_TRANSFORMS_LANE_VECTORIZE_PASSES_H
#define TRITON_SHARED_TRANSFORMS_LANE_VECTORIZE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton {

// Declarations of the lane-vectorize TTIR pass: the create<Name>() factory
// (used by the Python bindings) and registerLaneVectorizePasses() (used by the
// *-opt tools). See Passes.td.
#define GEN_PASS_DECL
#include "triton-shared/Transforms/LaneVectorize/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton-shared/Transforms/LaneVectorize/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_SHARED_TRANSFORMS_LANE_VECTORIZE_PASSES_H
