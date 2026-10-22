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
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  return success();
}

// tile.load %src attr-dict : type($src) -> type($result)
// tile.load %src [%i0, %i1] attr-dict : type($src) -> type($result)
void LoadOp::print(OpAsmPrinter &p) {
  p << ' ' << getSrc();
  auto indices = getIndices();
  if (!indices.empty()) {
    p << " [";
    llvm::interleaveComma(indices, p,
                          [&](Value v) { p.printOperand(v); });
    p << "]";
  }
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSrc().getType();
  for (Value index : indices)
    p << ", " << index.getType();
  p << " -> " << getResult().getType();
}

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
  Type srcType, resultType;

  if (parser.parseOperand(src))
    return failure();

  // Optional [indices]
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperandList(indices) || parser.parseRSquare())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(srcType))
    return failure();

  SmallVector<Type, 4> indexTypes;
  for (unsigned i = 0, e = indices.size(); i < e; ++i)
    if (parser.parseComma() || parser.parseType(indexTypes.emplace_back()))
      return failure();

  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();

  if (parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperands(indices, indexTypes, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  result.addTypes(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  return success();
}

// tile.store %src -> %dst attr-dict : type($src), type($dst)
// tile.store %src -> %dst [%i0, %i1] attr-dict : type($src), type($dst)
void StoreOp::print(OpAsmPrinter &p) {
  p << ' ' << getSrc() << " -> " << getDst();
  auto indices = getIndices();
  if (!indices.empty()) {
    p << " [";
    llvm::interleaveComma(indices, p,
                          [&](Value v) { p.printOperand(v); });
    p << "]";
  }
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSrc().getType() << ", " << getDst().getType();
  for (Value index : indices)
    p << ", " << index.getType();
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
  Type srcType, dstType;

  if (parser.parseOperand(src) || parser.parseArrow() ||
      parser.parseOperand(dst))
    return failure();

  // Optional [indices]
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperandList(indices) || parser.parseRSquare())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(srcType) || parser.parseComma() ||
      parser.parseType(dstType))
    return failure();

  SmallVector<Type, 4> indexTypes;
  for (unsigned i = 0, e = indices.size(); i < e; ++i)
    if (parser.parseComma() || parser.parseType(indexTypes.emplace_back()))
      return failure();

  if (parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands) ||
      parser.resolveOperands(indices, indexTypes, parser.getCurrentLocation(),
                             result.operands))
    return failure();

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