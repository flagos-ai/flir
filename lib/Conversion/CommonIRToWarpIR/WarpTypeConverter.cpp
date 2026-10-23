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

  addConversion([this](triton::PointerType ptrType) -> warp::PtrType {
    llvm::errs() << "addConversion[0]";
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());

    mlir::Attribute memorySpace =
    mlir::IntegerAttr::get(
        mlir::IntegerType::get(this->context, 32),
        ptrType.getAddressSpace());

    return warp::PtrType::get(
        this->context,
        pointeeTensorType,
        memorySpace
    );

  });
}
