//===- WarpIROps.cpp - WarpIR operation implementations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements WarpIR operations with their verification logic.
//
//===----------------------------------------------------------------------===//

#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::triton::warp;
//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  return success();
}

LogicalResult StoreOp::verify() {
  return success();
}

LogicalResult RegToTensorOp::verify() {
  return success();
}

LogicalResult DistributionAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    int64_t warpSize,
    ArrayRef<int64_t> threadsPerDim,
    ArrayRef<int64_t> elementsPerThread,
    ArrayRef<int64_t> repeat,
    ArrayRef<int64_t> order) {

  size_t rank = threadsPerDim.size();
  if (rank == 0)
    return emitError() << "rank must be > 0";
  if (elementsPerThread.size() != rank || repeat.size() != rank ||
      order.size() != rank)
    return emitError() << "threads_per_dim, elements_per_thread, repeat and "
                          "order must have the same length";

  int64_t prod = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (threadsPerDim[i] < 1 || elementsPerThread[i] < 1 || repeat[i] < 1)
      return emitError() << "all entries must be >= 1";
    prod *= threadsPerDim[i];
  }
  if (prod != warpSize)
    return emitError() << "product of threads_per_dim (" << prod
                       << ") must equal warp_size (" << warpSize << ")";

  SmallVector<bool> seen(rank, false);
  for (int64_t o : order) {
    if (o < 0 || o >= (int64_t)rank || seen[o])
      return emitError() << "order must be a permutation of [0, rank)";
    seen[o] = true;
  }
  return success();
}

LogicalResult MmaAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    MmaKind kind,
    llvm::ArrayRef<int64_t> instrShape,
    MmaOperand operand,
    int64_t elemBitWidth,
    llvm::ArrayRef<int64_t> repeat,
    llvm::ArrayRef<int64_t> repeatOrder,
    int64_t warpSize) {
    return success();
}