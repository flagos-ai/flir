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

#ifndef TRITON_SHARED_TritonToLinalgGeneric_GENERIC_OP_CONVERTER_H
#define TRITON_SHARED_TritonToLinalgGeneric_GENERIC_OP_CONVERTER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

class MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
public:
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class SplatConverter : public OpConversionPattern<triton::SplatOp> {
public:
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
public:
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
public:
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
public:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TransposeConverter : public OpConversionPattern<triton::TransOp> {
public:
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
public:
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct PreciseDivConverter : public OpConversionPattern<triton::PreciseDivFOp> {
public:
  using OpConversionPattern<triton::PreciseDivFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::PreciseDivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ClampFConverter : public OpConversionPattern<triton::ClampFOp> {
public:
  using OpConversionPattern<triton::ClampFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class JoinConverter : public OpConversionPattern<triton::JoinOp> {
public:
  using OpConversionPattern<triton::JoinOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class SplitConverter : public OpConversionPattern<triton::SplitOp> {
public:
  using OpConversionPattern<triton::SplitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class CatConverter : public OpConversionPattern<triton::CatOp> {
public:
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TritonMulhiuiConverter : public OpConversionPattern<triton::MulhiUIOp> {
public:
  using OpConversionPattern<triton::MulhiUIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TritonPreciseSqrtConverter
    : public OpConversionPattern<triton::PreciseSqrtOp> {
public:
  using OpConversionPattern<triton::PreciseSqrtOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace TTOpConverters

#endif // TRITON_SHARED_TritonToLinalgGeneric_GENERIC_OP_CONVERTER_H
