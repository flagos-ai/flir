//===- TensorViewOps.cpp - TensorView operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/TensorView/IR/TensorViewBuilder.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::tv;

//===----------------------------------------------------------------------===//
// ViewOpInterface
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/TensorView/IR/TensorViewInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TensorView/IR/TensorViewOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

// Verify that `res` matches `src` in everything but the encoding, and that
// `src` itself is a base (encoding-free) view.
static LogicalResult verifyEncodingParent(Operation *op, TensorViewType src,
                                          TensorViewType res) {
  if (src.getEncoding())
    return op->emitOpError("source must be a base view (no encoding)");
  if (src.getShape() != res.getShape() ||
      src.getStrides() != res.getStrides() ||
      src.getElementType() != res.getElementType())
    return op->emitOpError("result must match source except for the encoding");
  if (getEncodingPaddingValue(res.getEncoding()) != PaddingValue::ZERO &&
      !isa<FloatType>(res.getElementType()))
    return op->emitOpError(
        "nan and infinity padding require a floating-point element type");
  return success();
}

//===----------------------------------------------------------------------===//
// make_tensor_view
//===----------------------------------------------------------------------===//

LogicalResult MakeTensorViewOp::verify() {
  auto resTy = cast<TensorViewType>(getResult().getType());
  if (resTy.getEncoding())
    return emitOpError("result must be a base view (no encoding)");
  int64_t rank = resTy.getRank();
  if (static_cast<int64_t>(getSizes().size()) != rank)
    return emitOpError("expected ")
           << rank << " size operands but got " << getSizes().size();
  if (static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError("expected ")
           << rank << " stride operands but got " << getStrides().size();
  // The operand constraint already restricts the source to a tt.ptr.
  Type srcPointee = getPtrPointeeType(getSource().getType());
  if (resTy.getElementType() != srcPointee)
    return emitOpError("result element type must match source pointee type");
  return success();
}

//===----------------------------------------------------------------------===//
// make_partition_view / make_strided_view / make_gather_scatter_view
//===----------------------------------------------------------------------===//

TensorViewType MakePartitionViewOp::getEncodedView() {
  return cast<TensorViewType>(getResult().getType());
}
TensorViewType MakeStridedViewOp::getEncodedView() {
  return cast<TensorViewType>(getResult().getType());
}
TensorViewType MakeGatherScatterViewOp::getEncodedView() {
  return cast<TensorViewType>(getResult().getType());
}

LogicalResult MakePartitionViewOp::verify() {
  auto src = cast<TensorViewType>(getSource().getType());
  auto res = cast<TensorViewType>(getResult().getType());
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<PartitionViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.partition_view encoding");
  int64_t rank = res.getRank();
  if (static_cast<int64_t>(enc.getTile().size()) != rank)
    return emitOpError("partition tile rank must match the view rank");
  for (int64_t d : enc.getDimMap())
    if (d < 0 || d >= rank)
      return emitOpError("dim_map entry ")
             << d << " out of range [0, " << rank << ")";
  return success();
}

LogicalResult MakeStridedViewOp::verify() {
  auto src = cast<TensorViewType>(getSource().getType());
  auto res = cast<TensorViewType>(getResult().getType());
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<StridedViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.strided_view encoding");
  int64_t rank = res.getRank();
  if (static_cast<int64_t>(enc.getTile().size()) != rank)
    return emitOpError("strided tile rank must match the view rank");
  for (int64_t d : enc.getDimMap())
    if (d < 0 || d >= rank)
      return emitOpError("dim_map entry ")
             << d << " out of range [0, " << rank << ")";
  return success();
}

LogicalResult MakeGatherScatterViewOp::verify() {
  auto src = cast<TensorViewType>(getSource().getType());
  auto res = cast<TensorViewType>(getResult().getType());
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<GatherScatterViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.gather_scatter_view encoding");
  int64_t rank = res.getRank();
  if (static_cast<int64_t>(enc.getTile().size()) != rank)
    return emitOpError("gather/scatter tile rank must match the view rank");
  for (int64_t d : enc.getSparseDim())
    if (d < 0 || d >= rank)
      return emitOpError("sparse_dim entry ")
             << d << " out of range [0, " << rank << ")";
  return success();
}
