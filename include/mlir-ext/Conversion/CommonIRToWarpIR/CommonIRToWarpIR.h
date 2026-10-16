#ifndef TRITON_CONVERSION_COMMON_IR_TO_WARP_IR_CONVERSION_H
#define TRITON_CONVERSION_COMMON_IR_TO_WARP_IR_CONVERSION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {
namespace warp {
#define GEN_PASS_DECL
#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h.inc"

void populateAddWarpConfigsPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter);
std::unique_ptr<OperationPass<ModuleOp>> createCommonIRToWarpIRPass();

} // namespace warp
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_COMMON_IR_TO_WARP_IR_CONVERSION_H
