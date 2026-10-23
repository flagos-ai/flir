#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h"
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
namespace warp {

using namespace mlir::triton;

class AddFOpConversionPattern : public OpConversionPattern<arith::AddFOp>{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
        return success();
}
};

class LoadOpConversionPattern : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); 
    auto src = op.getPtr();
    auto tensorType = mlir::dyn_cast<RankedTensorType>(op.getType());
    mlir::Type pointeeType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    mlir::SmallVector<mlir::Attribute> shapeElements;
    for (int64_t dim : shape) {
    shapeElements.push_back(
        rewriter.getI64IntegerAttr(dim));
    }
    auto shapeAttr = rewriter.getArrayAttr(shapeElements);
    auto layout = inferDistribution(
        rewriter.getContext(),
        shape,
        /*numWarps*/4,
        /*warpSize*/32
    );
    auto resultType = warp::RegType::get(
        rewriter.getContext(),
        shape,
        pointeeType,
        layout
    );
    auto newOp = rewriter.create<warp::LoadOp>(
        loc,
        resultType,
        adaptor.getPtr()
    );
        
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}
};

void populateTritonPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter){
    MLIRContext *context = patterns.getContext();
    patterns.add<LoadOpConversionPattern, AddFOpConversionPattern>(typeConverter, context);
}

} // namespace warp
} // namespace triton
} // namespace mlir