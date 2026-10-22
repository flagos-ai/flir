//===- TensorViewOps.cpp - TensorView operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  Type srcPointee;
  if (auto tvPtrTy = dyn_cast<PtrType>(getSource().getType()))
    srcPointee = tvPtrTy.getPointeeType();
  else
    srcPointee =
        cast<triton::PointerType>(getSource().getType()).getPointeeType();
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

//===----------------------------------------------------------------------===//
// view_load / view_store
//===----------------------------------------------------------------------===//

TensorViewType ViewLoadOp::getEncodedView() {
  return cast<TensorViewType>(getView().getType());
}
TensorViewType ViewStoreOp::getEncodedView() {
  return cast<TensorViewType>(getView().getType());
}

LogicalResult ViewLoadOp::verify() {
  auto view = cast<TensorViewType>(getView().getType());
  Attribute enc = view.getEncoding();
  if (!isViewEncoding(enc))
    return emitOpError("view operand must carry a view encoding");
  if (static_cast<int64_t>(getIndices().size()) !=
      getEncodingIndexSpaceRank(enc))
    return emitOpError("expected ")
           << getEncodingIndexSpaceRank(enc) << " indices but got "
           << getIndices().size();
  auto gather = dyn_cast<GatherScatterViewAttr>(enc);
  for (auto [d, index] : llvm::enumerate(getIndices())) {
    bool isSparse = gather && llvm::is_contained(gather.getSparseDim(), d);
    if (isSparse) {
      auto indexTy = dyn_cast<RankedTensorType>(index.getType());
      if (!indexTy || indexTy.getRank() != 1)
        return emitOpError("index for sparse dimension ")
               << d << " must be a rank-1 tensor";
      if (!indexTy.getElementType().isIndex() &&
          !indexTy.getElementType().isInteger(32) &&
          !indexTy.getElementType().isInteger(64))
        return emitOpError("index tensor for sparse dimension ")
               << d << " must have index, i32, or i64 elements";
      int64_t length = indexTy.getShape()[0];
      if (!ShapedType::isDynamic(length) && length != gather.getTile()[d])
        return emitOpError("index tensor length for sparse dimension ")
               << d << " must equal the corresponding tile size";
    } else if (!index.getType().isIndex()) {
      return emitOpError("index for regular dimension ")
             << d << " must be a scalar index";
    }
  }
  auto resTy = cast<RankedTensorType>(getResult().getType());
  SmallVector<int64_t> tile = getEncodingTileShape(enc);
  if (resTy.getShape() != ArrayRef<int64_t>(tile))
    return emitOpError("result shape must equal the view tile shape");
  if (resTy.getElementType() != view.getElementType())
    return emitOpError("result element type must match the view element type");
  return success();
}

LogicalResult ViewStoreOp::verify() {
  auto view = cast<TensorViewType>(getView().getType());
  Attribute enc = view.getEncoding();
  if (!isViewEncoding(enc))
    return emitOpError("view operand must carry a view encoding");
  if (static_cast<int64_t>(getIndices().size()) !=
      getEncodingIndexSpaceRank(enc))
    return emitOpError("expected ")
           << getEncodingIndexSpaceRank(enc) << " indices but got "
           << getIndices().size();
  auto gather = dyn_cast<GatherScatterViewAttr>(enc);
  for (auto [d, index] : llvm::enumerate(getIndices())) {
    bool isSparse = gather && llvm::is_contained(gather.getSparseDim(), d);
    if (isSparse) {
      auto indexTy = dyn_cast<RankedTensorType>(index.getType());
      if (!indexTy || indexTy.getRank() != 1)
        return emitOpError("index for sparse dimension ")
               << d << " must be a rank-1 tensor";
      if (!indexTy.getElementType().isIndex() &&
          !indexTy.getElementType().isInteger(32) &&
          !indexTy.getElementType().isInteger(64))
        return emitOpError("index tensor for sparse dimension ")
               << d << " must have index, i32, or i64 elements";
      int64_t length = indexTy.getShape()[0];
      if (!ShapedType::isDynamic(length) && length != gather.getTile()[d])
        return emitOpError("index tensor length for sparse dimension ")
               << d << " must equal the corresponding tile size";
    } else if (!index.getType().isIndex()) {
      return emitOpError("index for regular dimension ")
             << d << " must be a scalar index";
    }
  }
  auto valTy = cast<RankedTensorType>(getValue().getType());
  SmallVector<int64_t> tile = getEncodingTileShape(enc);
  if (valTy.getShape() != ArrayRef<int64_t>(tile))
    return emitOpError("value shape must equal the view tile shape");
  if (valTy.getElementType() != view.getElementType())
    return emitOpError("value element type must match the view element type");
  return success();
}

//===----------------------------------------------------------------------===//
// ptr_load / ptr_store
//===----------------------------------------------------------------------===//

// Verify one rank-1 coordinate tensor per pointer-tensor dimension. Coordinates
// at the same lane are zipped into one N-D pointer-tensor lookup.
static LogicalResult verifyPointerCoordinates(Operation *op,
                                              RankedTensorType pointersType,
                                              OperandRange indices,
                                              int64_t &len) {
  if (pointersType.getRank() == 0)
    return op->emitOpError("pointers must have at least one dimension");
  if (indices.size() != static_cast<size_t>(pointersType.getRank()))
    return op->emitOpError("expected one coordinate tensor per pointer-tensor "
                           "dimension");
  len = ShapedType::kDynamic;
  for (Value idx : indices) {
    auto it = cast<RankedTensorType>(idx.getType());
    if (it.getRank() != 1)
      return op->emitOpError("coordinate tensors must be rank-1");
    int64_t l = it.getShape()[0];
    if (ShapedType::isDynamic(len))
      len = l;
    else if (!ShapedType::isDynamic(l) && l != len)
      return op->emitOpError("coordinate tensors must have equal length");
  }
  return success();
}

LogicalResult PtrLoadOp::verify() {
  auto pointersType = cast<RankedTensorType>(getPointers().getType());
  auto pointerType =
      dyn_cast<triton::PointerType>(pointersType.getElementType());
  if (!pointerType)
    return emitOpError("pointers must be a tensor of tt.ptr elements");
  int64_t len = ShapedType::kDynamic;
  if (failed(verifyPointerCoordinates(*this, pointersType, getIndices(), len)))
    return failure();
  auto resTy = cast<RankedTensorType>(getResult().getType());
  if (resTy.getRank() != 1)
    return emitOpError("result must be rank-1");
  if (!ShapedType::isDynamic(len) &&
      !ShapedType::isDynamic(resTy.getShape()[0]) && resTy.getShape()[0] != len)
    return emitOpError("result length must match the coordinate length");
  if (resTy.getElementType() != pointerType.getPointeeType())
    return emitOpError(
        "result element type must match the pointer pointee type");
  return success();
}

LogicalResult PtrStoreOp::verify() {
  auto pointersType = cast<RankedTensorType>(getPointers().getType());
  auto pointerType =
      dyn_cast<triton::PointerType>(pointersType.getElementType());
  if (!pointerType)
    return emitOpError("pointers must be a tensor of tt.ptr elements");
  int64_t len = ShapedType::kDynamic;
  if (failed(verifyPointerCoordinates(*this, pointersType, getIndices(), len)))
    return failure();
  auto valTy = cast<RankedTensorType>(getValue().getType());
  if (valTy.getRank() != 1)
    return emitOpError("value must be rank-1");
  if (!ShapedType::isDynamic(len) &&
      !ShapedType::isDynamic(valTy.getShape()[0]) && valTy.getShape()[0] != len)
    return emitOpError("value length must match the coordinate length");
  if (valTy.getElementType() != pointerType.getPointeeType())
    return emitOpError(
        "value element type must match the pointer pointee type");
  return success();
}
