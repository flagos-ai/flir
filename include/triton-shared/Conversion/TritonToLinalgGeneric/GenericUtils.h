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

#ifndef TRITON_SHARED_TritonToLinalgGeneric_GENERIC_UTILS_H
#define TRITON_SHARED_TritonToLinalgGeneric_GENERIC_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace TTOpConverters {
namespace GenericUtils {

mlir::Value getTransposedValue(mlir::Value source, const mlir::Location loc,
                               mlir::ConversionPatternRewriter &rewriter,
                               llvm::ArrayRef<int> order);

llvm::SmallVector<mlir::utils::IteratorType>
getNParallelLoopsAttrs(unsigned n);


enum class ReduceWithIndexType { MAX, MIN, None };
enum class TieBreakType { LEFT, RIGHT, None };

struct ReduceWithIndexParams {
  ReduceWithIndexType withIndexType = ReduceWithIndexType::None;
  TieBreakType tieBreakType = TieBreakType::None;
  bool isUnsignedSrc;
};

llvm::FailureOr<ReduceWithIndexParams>
getReduceWithIndexParams(mlir::triton::ReduceOp op);

void addReduceWithIndexAttr(ReduceWithIndexParams params,
                            mlir::ConversionPatternRewriter &rewriter,
                            mlir::linalg::ReduceOp reduceOp);

llvm::SmallVector<int64_t> getBroadcastDims(mlir::RankedTensorType src,
                                            mlir::RankedTensorType dst);

llvm::SmallVector<int64_t> getUnbroadcastDims(mlir::RankedTensorType src,
                                              mlir::RankedTensorType dst);

} // namespace GenericUtils
} // namespace TTOpConverters

#endif // TRITON_SHARED_TritonToLinalgGeneric_GENERIC_UTILS_H
