# include "mlir-ext/Conversion/CommonIRToWarpIR/InferWarpDistribution.h"
# include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::warp {
using namespace llvm;

DistributionAttr inferDistribution(
    mlir::MLIRContext *ctx,
    const llvm::ArrayRef<int64_t> shape,
    int64_t numWarps,
    int64_t threadsPerWarp){

    auto encoding = mlir::triton::gpu::getDefaultBlockedEncoding(ctx, shape, numWarps, threadsPerWarp, 8); 
    return DistributionAttr::get(ctx, threadsPerWarp, encoding.getSizePerThread(), encoding.getOrder());

}
}