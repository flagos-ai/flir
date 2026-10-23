# include "mlir-ext/Conversion/CommonIRToWarpIR/InferWarpDistribution.h"

namespace mlir::triton::warp {
using namespace llvm;
        /* num_warps -> (threadsPerDim, elementsPerThread, order)*/
DistributionAttr inferDistribution(
    mlir::MLIRContext *ctx,
    const llvm::ArrayRef<int64_t> shape,
    int64_t numWarps,
    int64_t warpSize){
    
    auto threadsPerDim = SmallVector<int64_t>{4, 8};
    auto elementsPerThread = SmallVector<int64_t>{1, 4};
    auto order = SmallVector<int64_t>{0, 1};

    return DistributionAttr::get(
            ctx,
            /*warpSize*/warpSize,
            /*threadsPerDim*/threadsPerDim,
            /*elementsPerThread*/elementsPerThread,
            /*order*/order,
            /*numWarps*/numWarps);

}

}