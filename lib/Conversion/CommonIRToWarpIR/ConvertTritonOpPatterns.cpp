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

class MakeRangeOpConversionPattern : public OpConversionPattern<triton::MakeRangeOp> {
    using OpConversionPattern::OpConversionPattern;
    public:
    MakeRangeOpConversionPattern(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::MakeRangeOp>(typeConverter, context){}
    
    LogicalResult
    matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Type> retTypes;
        if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
                return failure();
        auto resultTensorType = dyn_cast<RankedTensorType>(retTypes[0]);
        auto _prod = 1;
        for (auto dim : resultTensorType.getShape()) {
            _prod *= dim;
        }
        rewriter.modifyOpInPlace(op, [&]() {
            op.setEndAttr(rewriter.getI32IntegerAttr(_prod));
        });
        rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());
        return success();
    }
};

class LoadOpConversionPattern : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  public:
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
 
  LoadOpConversionPattern(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto resultTy = getTypeConverter()->convertType(op.getResult().getType());
    auto newOp = rewriter.create<warp::LoadOp>(
        op.getLoc(), TypeRange{resultTy}, 
        adaptor.getPtr(), adaptor.getMask(), adaptor.getOther(),
        op.getBoundaryCheckAttr(), op.getPaddingAttr(), op.getCacheAttr(),
        op.getEvictAttr(), op.getIsVolatileAttr(), op.getFlagtreeHintsAttr()
    );

    newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
}
};

void populateTritonPatterns(RewritePatternSet &patterns,
                            TypeConverter &typeConverter, 
                            int numWarps, int threadsPerWarp, int numCTAs){
    MLIRContext *context = patterns.getContext();
    patterns.add<LoadOpConversionPattern, MakeRangeOpConversionPattern>(typeConverter,context);
}
} // namespace warp
} // namespace triton
} // namespace mlir