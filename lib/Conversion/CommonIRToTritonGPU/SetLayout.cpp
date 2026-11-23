#include "mlir-ext/Conversion/CommonIRToTritonGPU/Patterns.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::tile {
namespace {

namespace ttg = mlir::triton::gpu;

struct SetLayoutPattern : OpConversionPattern<SetLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SetLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto encoding = op.getLayout();
    auto type = op.getResult().getType().cloneWithEncoding(encoding);
    Value source = adaptor.getSource();
    if (source.getType() == type &&
        succeeded(trySetExplicitValueEncoding(source, encoding))) {
      rewriter.replaceOp(op, source);
    } else {
      auto convert = rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
          op, type, source);
      setExplicitResultEncoding(convert, 0, encoding);
    }
    return success();
  }
};

} // namespace

void populateSetLayoutToTritonGPUPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        ConversionTarget &target) {
  patterns.add<SetLayoutPattern>(typeConverter, patterns.getContext());
  target.addIllegalOp<SetLayoutOp>();
}

} // namespace mlir::triton::tile
