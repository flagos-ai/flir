#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h"
#include "mlir-ext/Conversion/CommonIRToWarpIR/WarpTypeConverter.h"
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
#define GEN_PASS_DEF_COMMONIRTOWARPIR
#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h.inc"
} // namespace warp
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "commonir-to-warpir-conversion"

namespace mlir{
namespace triton{
namespace warp{


inline void setLegalityForTarget(ConversionTarget &target, const WarpTypeConverter &tc) {
    target.addLegalDialect<
        mlir::triton::warp::WarpIRDialect>();
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalDialect<mlir::triton::TritonDialect,
                                    arith::ArithDialect,
                                    math::MathDialect>(
      [&tc](Operation *op) { return tc.isLegal(op); });

    target.addDynamicallyLegalOp<mlir::triton::FuncOp>(
      [&tc](mlir::triton::FuncOp f) {
        return tc.isSignatureLegal(f.getFunctionType()) &&
               tc.isLegal(&f.getBody());
      });

    target.addDynamicallyLegalOp<mlir::triton::LoadOp>(
      [](mlir::triton::LoadOp op) {
        return !isa<RankedTensorType>(op.getType());
      });
}

class CommonIRToWarpIRPass
    : public impl::CommonIRToWarpIRBase<CommonIRToWarpIRPass> {
  using CommonIRToWarpIRBase<
      CommonIRToWarpIRPass>::CommonIRToWarpIRBase;

    void runOnOperation() override {
        auto moduleOp = getOperation();
        mlir::RewritePatternSet patterns(&getContext());
        ConversionTarget target(*moduleOp.getContext());
        WarpTypeConverter typeConverter(&getContext(), numWarps, threadsPerWarp, numCTAs);

        populateTritonPatterns(patterns, typeConverter, numWarps, threadsPerWarp, numCTAs);    
        populateGenericPatterns(patterns, typeConverter);
        populateCommonIRPatterns(patterns, typeConverter);
        setLegalityForTarget(target, typeConverter);

        if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};


} // namespace warp
} // namespace triton
} // namespace mlir
