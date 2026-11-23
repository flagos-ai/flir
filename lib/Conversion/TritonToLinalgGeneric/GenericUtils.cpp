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

#include "triton-shared/Conversion/TritonToLinalgGeneric/GenericUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cassert>

namespace TTOpConverters {
using namespace mlir;
using namespace llvm;
namespace GenericUtils {

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(order);
  SmallVector<int64_t> originalShape(sourceType.getShape());
  SmallVector<int64_t> transposedShape(sourceRank);
  for (size_t i = 0; i < sourceRank; i++) {
    transposedShape[i] = originalShape[perm[i]];
  }

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] != srcShape[i]) {
      assert(srcShape[i] == 1 &&
             "Size of source broadcast dimension must be 1");
      broadcastDims.push_back(i);
    }
  }
  assert(!broadcastDims.empty() && "Cannot identify broadcast dimension");
  return broadcastDims;
}

SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) {
  SmallVector<int64_t> unbroadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] == srcShape[i]) {
      unbroadcastDims.emplace_back(srcShape[i]);
    }
  }
  return unbroadcastDims;
}

} // namespace GenericUtils
} // namespace TTOpConverters
