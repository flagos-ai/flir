# include "mlir-ext/Conversion/CommonIRToWarpIR/InferWarpDistribution.h"
# include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::warp {
using namespace llvm;

DistributionAttr inferDistribution(
    mlir::MLIRContext *ctx,
    const llvm::ArrayRef<int64_t> shape,
    int64_t numWarps,
    int64_t warpSize) {

    auto encoding = mlir::triton::gpu::getDefaultBlockedEncoding(ctx, shape, numWarps, warpSize, 8); 
    return DistributionAttr::get(ctx, warpSize, encoding.getThreadsPerWarp(), encoding.getSizePerThread(), encoding.getOrder());

}
}