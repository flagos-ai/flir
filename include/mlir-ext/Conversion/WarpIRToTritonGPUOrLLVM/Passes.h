#ifndef WARP_IR_TO_TRITON_GPU_OR_LLMV_CONVERSION_PASSES_H
#define WARP_IR_TO_TRITON_GPU_OR_LLMV_CONVERSION_PASSES_H

#include "mlir-ext/Conversion/WarpIRToTritonGPUOrLLVM/WarpIRToLLVM.h"
#include "mlir-ext/Conversion/WarpIRToTritonGPUOrLLVM/WarpIRToTritonGPU.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace warp {

#define GEN_PASS_DECL
#include "mlir-ext/Conversion/WarpIRToTritonGPUOrLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir-ext/Conversion/WarpIRToTritonGPUOrLLVM/Passes.h.inc"

} // namespace warp
} // namespace triton
} // namespace mlir

#endif