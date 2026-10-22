//===- TensorViewBuilder.cpp - Direct TensorView construction ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/TensorView/IR/TensorViewBuilder.h"

#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>

using namespace mlir;
namespace tv = mlir::triton::tv;

namespace {

static Value createBaseView(OpBuilder &b, Location loc, Value basePtr,
                            Type elementType, ArrayRef<int64_t> strideStatic,
                            ArrayRef<Value> strideDyn, ArrayRef<Value> extent) {
  // The base pointer is used as it comes in: a `tt.ptr` stays a `tt.ptr`, so
  // the value keeps the type the rest of the pipeline lowers. Which pointers a
  // view can actually be built on is decided later, by
  // `commonir-tv-ptr-legalize`, once every use of the pointer is visible.
  unsigned rank = strideStatic.size();

  SmallVector<Value> strideOperands;
  SmallVector<int64_t> strideTy;
  for (unsigned d = 0; d < rank; ++d) {
    if (strideStatic[d] == ShapedType::kDynamic) {
      strideOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), strideDyn[d]));
      strideTy.push_back(ShapedType::kDynamic);
    } else {
      strideOperands.push_back(
          b.create<arith::ConstantIndexOp>(loc, strideStatic[d]));
      strideTy.push_back(strideStatic[d]);
    }
  }

  SmallVector<Value> sizeOperands;
  for (unsigned d = 0; d < rank; ++d) {
    if (d < extent.size() && extent[d])
      sizeOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), extent[d]));
    else
      sizeOperands.push_back(b.create<arith::ConstantIndexOp>(
          loc, std::numeric_limits<int64_t>::max()));
  }

  SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
  auto baseTy = tv::TensorViewType::get(dynShape, elementType, strideTy,
                                        /*encoding=*/Attribute());
  return b
      .create<tv::MakeTensorViewOp>(loc, baseTy, basePtr, sizeOperands,
                                    strideOperands)
      .getResult();
}

static tv::TensorViewType getEncodedViewType(Value baseView,
                                             Attribute encoding) {
  auto baseTy = cast<tv::TensorViewType>(baseView.getType());
  return tv::TensorViewType::get(baseTy.getShape(), baseTy.getElementType(),
                                 baseTy.getStrides(), encoding);
}

} // namespace

namespace mlir {
namespace triton {
namespace tv {

Type getPtrPointeeType(Type type) {
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    return ptr.getPointeeType();
  return Type();
}

FailureOr<Value> createTensorViewBase(OpBuilder &b, Location loc, Value basePtr,
                                      ValueRange shape, ValueRange strides) {
  if (basePtr.getDefiningOp<triton::AddPtrOp>()) {
    emitError(loc)
        << "TensorView base pointer must not be computed with pointer "
           "addition; represent the complete logical tensor in "
           "`shape`/`strides` and select positions through the encoded "
           "view's `index=` argument";
    return failure();
  }

  unsigned rank = shape.size();
  Type elementType = getPtrPointeeType(basePtr.getType());
  SmallVector<int64_t> strideStatic(rank, ShapedType::kDynamic);
  SmallVector<Value> strideDyn(strides.begin(), strides.end());
  SmallVector<Value> extent(shape.begin(), shape.end());
  return createBaseView(b, loc, basePtr, elementType, strideStatic, strideDyn,
                        extent);
}

Value createPartitionView(OpBuilder &b, Location loc, Value baseView,
                          ArrayRef<int64_t> tile,
                          PaddingValue paddingValue) {
  auto encoding =
      tv::PartitionViewAttr::get(b.getContext(), tile, paddingValue);
  auto resultType = getEncodedViewType(baseView, encoding);
  return b.create<tv::MakePartitionViewOp>(loc, resultType, baseView)
      .getResult();
}

Value createStridedView(OpBuilder &b, Location loc, Value baseView,
                        ArrayRef<int64_t> tile,
                        ArrayRef<int64_t> traversalStrides,
                        PaddingValue paddingValue) {
  auto encoding = tv::StridedViewAttr::get(b.getContext(), tile,
                                           traversalStrides, paddingValue);
  auto resultType = getEncodedViewType(baseView, encoding);
  return b.create<tv::MakeStridedViewOp>(loc, resultType, baseView).getResult();
}

Value createGatherScatterView(OpBuilder &b, Location loc, Value baseView,
                              ArrayRef<int64_t> tile,
                              ArrayRef<int64_t> sparseDims,
                              PaddingValue paddingValue) {
  auto encoding = tv::GatherScatterViewAttr::get(b.getContext(), tile,
                                                 sparseDims, paddingValue);
  auto resultType = getEncodedViewType(baseView, encoding);
  return b.create<tv::MakeGatherScatterViewOp>(loc, resultType, baseView)
      .getResult();
}

} // namespace tv
} // namespace triton
} // namespace mlir
