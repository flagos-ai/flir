#include "mlir-ext/Conversion/WarpIRToTritonGPUOrLLVM/Passes.h"
// #include "mlir-ext/Conversion/WarpIRToTritonGPUOrLLVM/WarpTypeConverter.h"
#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
namespace warp{
#define GEN_PASS_DEF_WARPIRTOTRITONGPUORLLVM
#include "mlir-ext/Conversion/WarpIRToTritonGPUOrLLVM/Passes.h.inc"
} // namespace warp
} // namespace triton
} // namespace mlir