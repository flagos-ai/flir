#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
namespace warp{
#define GEN_PASS_DEF_COMMONIRTOWARPIR
#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h.inc"
} // namespace warp
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "commonir-to-warpir-conversion"

namespace {

class CommonIRToWarpIRPass
    : public ::mlir::triton::warp::impl::CommonIRToWarpIRBase<CommonIRToWarpIRPass> {
  using CommonIRToWarpIRBase<
      CommonIRToWarpIRPass>::CommonIRToWarpIRBase;

    void runOnOperation() override {
        auto moduleOp = getOperation();
    }
};

}