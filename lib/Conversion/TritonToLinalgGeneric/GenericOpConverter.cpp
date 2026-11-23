/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "triton-shared/Conversion/TritonToLinalgGeneric/GenericOpConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVectorExtras.h"

namespace TTOpConverters {

LogicalResult PreciseDivConverter::matchAndRewrite(
    triton::PreciseDivFOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value opa = adaptor.getX();
  Value opb = adaptor.getY();
  auto loc = op.getLoc();

  if (opa.getType() != opb.getType())
    return rewriter.notifyMatchFailure(op, "operands must have the same type");

  // Let DivFOp infer its result type from converted operands. PreciseDivFOp is
  // valid for both scalar and tensor floating-point values; casting the
  // original result to RankedTensorType made scalar divisions produce a null
  // result type during partial conversion.
  auto divOp = rewriter.create<arith::DivFOp>(loc, opa, opb);
  rewriter.replaceOp(op, divOp.getResult());
  return success();
}

LogicalResult DenseConstantConverter::matchAndRewrite(
    arith::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto denseAttr = cast<DenseElementsAttr>(op.getValue());
  auto loc = op.getLoc();
  auto constSplatOp = arith::ConstantOp::materialize(
      rewriter, denseAttr.getSplatValue<Attribute>(),
      denseAttr.getElementType(), loc);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
      denseAttr.getElementType());

  rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{constSplatOp},
                                              ValueRange{emptyOp});

  return success();
}

LogicalResult
ClampFConverter::matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto input = adaptor.getX();
  auto min_para = adaptor.getMin();
  auto max_para = adaptor.getMax();
  auto propagateNan_para = adaptor.getPropagateNan();

  if (auto input_type = dyn_cast<RankedTensorType>(input.getType())) {
    if (isa<FloatType>(min_para.getType())) {
      auto minEmptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, input_type.getShape(), input_type.getElementType());
      min_para = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{min_para},
                                             ValueRange{minEmptyTensor})
                     .result();
    }
    if (isa<FloatType>(max_para.getType())) {
      auto maxEmptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, input_type.getShape(), input_type.getElementType());
      max_para = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{max_para},
                                             ValueRange{maxEmptyTensor})
                     .result();
    }
  }

  if (propagateNan_para == PropagateNan::NONE) {
    auto minOp = rewriter.create<arith::MinNumFOp>(loc, input, max_para);
    auto maxOp = rewriter.create<arith::MaxNumFOp>(loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else if (propagateNan_para == PropagateNan::ALL) {
    auto minOp = rewriter.create<arith::MinimumFOp>(loc, input, max_para);
    auto maxOp = rewriter.create<arith::MaximumFOp>(loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else {
    return failure();
  }

  return success();
}

LogicalResult
JoinConverter::matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  Value emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                   resType.getElementType());

  auto shape = dyn_cast<RankedTensorType>(opa.getType()).getShape();
  auto sizes = llvm::map_to_vector(shape, [&](int64_t t) {
    return OpFoldResult(rewriter.getI64IntegerAttr(t));
  });
  sizes.push_back(rewriter.getI64IntegerAttr(1));

  int64_t rank = resType.getRank();

  // Set last dimension stride to 2 in layout. As last dimension size is always
  // 1, last dimension stride here could be either 1 or 2, while stride `2`
  // could carry interleave trait and it's convenient for next lower.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

  auto insert0 = rewriter.create<tensor::InsertSliceOp>(
      loc, opa, emptyOp, offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto insert1 = rewriter.create<tensor::InsertSliceOp>(
      loc, opb, insert0, offsets, sizes, strides);
  rewriter.replaceOp(op, insert1);
  return success();
}

LogicalResult
CatConverter::matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resType || resType.getRank() != 1) {
    return rewriter.notifyMatchFailure(op, "only support 1D cat");
  }

  auto inputTypeA = dyn_cast<RankedTensorType>(opa.getType());
  auto inputTypeB = dyn_cast<RankedTensorType>(opb.getType());
  if (!inputTypeA || !inputTypeB || inputTypeA.getRank() != 1 ||
      inputTypeB.getRank() != 1) {
    return rewriter.notifyMatchFailure(op, "inputs must be 1D tensors");
  }

  int64_t sizeA = inputTypeA.getShape()[0];
  int64_t sizeB = inputTypeB.getShape()[0];

  // Only handle the case where both inputs have size 1 (i.e., scalar-like).
  if (sizeA == 1 && sizeB == 1) {
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                    resType.getElementType());

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value scalarA = rewriter.create<tensor::ExtractOp>(loc, opa, zero);
    Value scalarB = rewriter.create<tensor::ExtractOp>(loc, opb, zero);

    Value inserted0 =
        rewriter.create<tensor::InsertOp>(loc, scalarA, emptyOp, zero);
    Value inserted1 =
        rewriter.create<tensor::InsertOp>(loc, scalarB, inserted0, one);

    rewriter.replaceOp(op, inserted1);
    return success();
  }

  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                  resType.getElementType());

  auto rank = resType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  auto inputType = dyn_cast<RankedTensorType>(opa.getType());

  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      });

  auto insert0 = rewriter.create<tensor::InsertSliceOp>(
      loc, opa, emptyOp, offsets, sizes, strides);

  offsets[0] =
      rewriter.getIndexAttr(inputType.getRank() ? inputType.getShape()[0] : 1);
  auto insert1 = rewriter.create<tensor::InsertSliceOp>(
      loc, opb, insert0, offsets, sizes, strides);

  rewriter.replaceOp(op, insert1);
  return success();
}

LogicalResult
SplitConverter::matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Value input = op.getSrc();
  auto loc = op.getLoc();
  auto inputType = cast<RankedTensorType>(input.getType());

  int64_t rank = inputType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  auto outType = dyn_cast<RankedTensorType>(op.getOutLHS().getType());
  auto sizes = llvm::map_to_vector(outType.getShape(), [&](int64_t t) {
    return OpFoldResult(rewriter.getIndexAttr(t));
  });
  sizes.push_back(rewriter.getIndexAttr(1));

  auto slice0 = rewriter.create<tensor::ExtractSliceOp>(
      loc, outType, input, offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
      loc, outType, input, offsets, sizes, strides);

  SmallVector<Value, 2> slices = {slice0.getResult(), slice1.getResult()};
  rewriter.replaceOp(op, ValueRange(slices));
  return success();
}

LogicalResult TritonMulhiuiConverter::matchAndRewrite(
    triton::MulhiUIOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value opl = op.getX();
  Value opr = op.getY();
  Value res = op.getResult();
  auto newMulOp = rewriter.create<arith::MulUIExtendedOp>(
      loc, res.getType(), res.getType(), opl, opr);
  // Triton only needs the high value.
  rewriter.replaceOp(op, ValueRange{newMulOp.getHigh()});
  return success();
}

LogicalResult TritonPreciseSqrtConverter::matchAndRewrite(
    triton::PreciseSqrtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<math::SqrtOp>(op, adaptor.getOperands());
  return success();
}

} // namespace TTOpConverters
