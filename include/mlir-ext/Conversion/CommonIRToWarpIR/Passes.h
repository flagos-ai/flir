#ifndef COMMON_IR_TO_WARP_IR_CONVERSION_PASSES_H
#define COMMON_IR_TO_WARP_IR_CONVERSION_PASSES_H

#include "mlir-ext/Conversion/CommonIRToWarpIR/CommonIRToWarpIR.h"
#include "mlir-ext/Conversion/CommonIRToWarpIR/GenericOpConversion.h"
#include "mlir-ext/Conversion/CommonIRToWarpIR/InferWarpDistribution.h"

namespace mlir {
namespace triton {
namespace warp {
#define GEN_PASS_REGISTRATION
#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h.inc"

} // namespace warp
} // namespace triton
} // namespace mlir

#endif