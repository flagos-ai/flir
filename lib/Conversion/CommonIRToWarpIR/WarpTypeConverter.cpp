#include "mlir-ext/Conversion/CommonIRToWarpIR/WarpTypeConverter.h"
#include <algorithm>
#include <numeric>
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h"
#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"


using namespace mlir;
using namespace mlir::triton::warp;
using namespace mlir::triton::tile;

//
// TypeConverter
//
WarpTypeConverter::WarpTypeConverter(MLIRContext *context,
                                     int numWarps, int threadsPerWarp,
                                     int numCTAs)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](Type type) { return type; });

  addConversion([this](RankedTensorType t) -> Type {           
      if (t.getEncoding() || t.getRank() == 0)
        return t;
      auto warpLayout = inferDistribution(this->context, t.getShape(), this->numWarps, this->threadsPerWarp);
      auto _prod = this -> threadsPerWarp;
      for (auto e : warpLayout.getElementsPerThread())
        _prod *= e;

      if (!warpLayout)
        return Type();                                      
      return RankedTensorType::get(_prod, t.getElementType(), warpLayout);
    });

}
