//===- CommonIROps.cpp - CommonIR operation implementations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CommonIR operations with their verification logic.
//
//===----------------------------------------------------------------------===//

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::triton::tile;

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//

LogicalResult SubViewOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// GmOffsetOp
//===----------------------------------------------------------------------===//

LogicalResult GmOffsetOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::verify() {
  if (getShape().empty())
    return success(); // Legacy full-region copy.
  for (Value operand : {getSrc(), getDst()}) {
    Type type = operand.getType();
    if (auto ptr = dyn_cast<mlir::triton::PointerType>(type))
      type = ptr.getPointeeType();
    ArrayRef<int64_t> capacity;
    if (auto buf = dyn_cast<BufType>(type))
      capacity = buf.getShape();
    else if (auto tensor = dyn_cast<mlir::triton::tile::TensorType>(type))
      capacity = tensor.getShape();
    else if (auto shaped = dyn_cast<ShapedType>(type); shaped && shaped.hasRank())
      capacity = shaped.getShape();
    else
      return emitOpError("explicit copy extent requires ranked operands");
    if (capacity.size() != getShape().size())
      return emitOpError("copy extent rank must match both operands");
    for (auto [extent, bound] : llvm::zip(getShape(), capacity)) {
      APInt constant;
      if (!matchPattern(extent, m_ConstantInt(&constant)))
        continue;
      int64_t size = constant.getSExtValue();
      if (size < 0)
        return emitOpError("copy extents must be non-negative");
      if (!ShapedType::isDynamic(bound) && size > bound)
        return emitOpError("copy extent exceeds operand capacity");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// VecFillOp
//===----------------------------------------------------------------------===//

LogicalResult VecFillOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// CubeLaunchOp
//===----------------------------------------------------------------------===//

LogicalResult CubeLaunchOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// CubeWaitOp
//===----------------------------------------------------------------------===//

LogicalResult CubeWaitOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// ElemwiseOp
//===----------------------------------------------------------------------===//

LogicalResult ElemwiseOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ToTensorOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// StoreTensorOp
//===----------------------------------------------------------------------===//

LogicalResult StoreTensorOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// SetFlagOp
//===----------------------------------------------------------------------===//

LogicalResult SetFlagOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// WaitFlagOp
//===----------------------------------------------------------------------===//

LogicalResult WaitFlagOp::verify() {
  return success();
}

//===----------------------------------------------------------------------===//
// PipeBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult PipeBarrierOp::verify() {
  return success();
}