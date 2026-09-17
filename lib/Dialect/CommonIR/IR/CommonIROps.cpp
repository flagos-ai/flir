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
  auto srcOffsets = getSrcOffsets();
  auto dstOffsets = getDstOffsets();
  auto sizes = getSizes();

  if ((!srcOffsets.empty() || !dstOffsets.empty()) && !sizes)
    return emitOpError("slice copy requires 'sizes' attribute");

  if (sizes) {
    auto numDims = sizes->size();
    if (!srcOffsets.empty() && srcOffsets.size() != numDims)
      return emitOpError("src_offsets count (")
             << srcOffsets.size() << ") must match sizes rank (" << numDims
             << ")";
    if (!dstOffsets.empty() && dstOffsets.size() != numDims)
      return emitOpError("dst_offsets count (")
             << dstOffsets.size() << ") must match sizes rank (" << numDims
             << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp — custom assembly format
//
//   tile.copy %src -> %dst
//       [src_offsets [%a, %b] dst_offsets [%c, %d]]
//       {attrs} : type($src), type($dst)
//===----------------------------------------------------------------------===//

void CopyOp::print(OpAsmPrinter &p) {
  p << ' ' << getSrc() << " -> " << getDst();
  if (!getSrcOffsets().empty()) {
    p << " src_offsets [";
    llvm::interleaveComma(getSrcOffsets(), p,
                          [&](Value v) { p.printOperand(v); });
    p << "]";
  }
  if (!getDstOffsets().empty()) {
    p << " dst_offsets [";
    llvm::interleaveComma(getDstOffsets(), p,
                          [&](Value v) { p.printOperand(v); });
    p << "]";
  }
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"operandSegmentSizes"});
  p << " : " << getSrc().getType() << ", " << getDst().getType();
}

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcType, dstType;

  if (parser.parseOperand(src) || parser.parseArrow() ||
      parser.parseOperand(dst))
    return failure();

  // Parse optional src_offsets [...]
  SmallVector<OpAsmParser::UnresolvedOperand> srcOffsets, dstOffsets;
  if (succeeded(parser.parseOptionalKeyword("src_offsets"))) {
    if (parser.parseLSquare() ||
        parser.parseOperandList(srcOffsets) ||
        parser.parseRSquare())
      return failure();
  }

  // Parse optional dst_offsets [...]
  if (succeeded(parser.parseOptionalKeyword("dst_offsets"))) {
    if (parser.parseLSquare() ||
        parser.parseOperandList(dstOffsets) ||
        parser.parseRSquare())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(srcType) || parser.parseComma() ||
      parser.parseType(dstType))
    return failure();

  // Resolve operands
  auto indexType = parser.getBuilder().getIndexType();
  if (parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands) ||
      parser.resolveOperands(srcOffsets, indexType, result.operands) ||
      parser.resolveOperands(dstOffsets, indexType, result.operands))
    return failure();

  // Set operand segment sizes: [src, dst, src_offsets, dst_offsets]
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, 1, static_cast<int32_t>(srcOffsets.size()),
           static_cast<int32_t>(dstOffsets.size())}));
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