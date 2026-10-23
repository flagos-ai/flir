#include "mlir-ext/Conversion/CommonIRToWarpIR/CommonIRToWarpIR.h"
#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::warp {
using namespace llvm;
DistributionAttr inferDistribution(mlir::MLIRContext *ctx,
                                   const llvm::ArrayRef<int64_t> shape,
                                   int64_t numWarps, int64_t warpSize);

} // namespace mlir::triton::warp