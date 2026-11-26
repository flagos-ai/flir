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
#include <map>
#include <tuple>
#include <utility>
#include <vector>

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

void addReduceWithIndexAttr(ReduceWithIndexParams params,
                            ConversionPatternRewriter &rewriter,
                            linalg::ReduceOp reduceOp) {
  const StringRef reduceRef = "reduce_mode";
  const StringRef tieBreakLeftRef = "tie_break_left";
  const StringRef unsignedSrcRef = "unsigned_src";

  const StringRef tieBreakStr =
      params.tieBreakType == TieBreakType::LEFT ? "true" : "false";
  const StringRef withIndexStr =
      params.withIndexType == ReduceWithIndexType::MAX ? "max_with_index"
                                                       : "min_with_index";
  const StringRef unsignedSrcStr = params.isUnsignedSrc ? "true" : "false";

  reduceOp->setAttr(reduceRef, rewriter.getStringAttr(withIndexStr));
  reduceOp->setAttr(tieBreakLeftRef, rewriter.getStringAttr(tieBreakStr));
  reduceOp->setAttr(unsignedSrcRef, rewriter.getStringAttr(unsignedSrcStr));
}

llvm::FailureOr<ReduceWithIndexParams>
getReduceWithIndexParams(triton::ReduceOp op) {
  auto tritonReduceBlock = op.getBody();
  auto *tritonYield = tritonReduceBlock->getTerminator();
  auto yieldValues = tritonYield->getOperands();
  constexpr int yieldValuesNum = 2;
  if (yieldValues.empty()) {
    return llvm::failure();
  }
  if (yieldValues.size() != yieldValuesNum) {
    return ReduceWithIndexParams{};
  }

  // Unify signed/unsigned and int/float predicate
  enum class Predicate { Undefined = 0, lt = 1, gt = 2, eq = 3 };
  enum class Signedness { NotApplicable = 0, Signed = 1, Unsigned = 2 };
  auto unifyPredicateI =
      [](arith::CmpIPredicate p) -> std::pair<Predicate, Signedness> {
    switch (p) {
    case arith::CmpIPredicate::slt:
      return {Predicate::lt, Signedness::Signed};
    case arith::CmpIPredicate::ult:
      return {Predicate::lt, Signedness::Unsigned};
    case arith::CmpIPredicate::sgt:
      return {Predicate::gt, Signedness::Signed};
    case arith::CmpIPredicate::ugt:
      return {Predicate::gt, Signedness::Unsigned};
    case arith::CmpIPredicate::eq:
      return {Predicate::eq, Signedness::NotApplicable};
    default:
      return {Predicate::Undefined, Signedness::NotApplicable};
    }
  };
  auto unifyPredicateF =
      [](arith::CmpFPredicate p) -> std::pair<Predicate, Signedness> {
    switch (p) {
    case arith::CmpFPredicate::OLT:
      return {Predicate::lt, Signedness::Signed};
    case arith::CmpFPredicate::ULT:
      return {Predicate::lt, Signedness::Unsigned};
    case arith::CmpFPredicate::OGT:
      return {Predicate::gt, Signedness::Signed};
    case arith::CmpFPredicate::UGT:
      return {Predicate::gt, Signedness::Unsigned};
    case arith::CmpFPredicate::OEQ:
      return {Predicate::eq, Signedness::Signed};
    case arith::CmpFPredicate::UEQ:
      return {Predicate::eq, Signedness::Unsigned};
    default:
      return {Predicate::Undefined, Signedness::NotApplicable};
    }
  };

  // Composite predicate to pick index of min (or max) element have to be
  // written in following form: (v means value and i means index)
  // For leftmost element:
  //    (new_v == old_v and new_i < old_i) or new_v < old_v
  //    new_v < old_v or (new_v == old_v and new_i < old_i)
  //    new_v < old_v // python3.11 ttir
  //    (new_v == old_v and new_i < old_i) or new_v > old_v
  //    new_v > old_v or (new_v == old_v and new_i < old_i)
  //    new_v > old_v // python3.11 ttir
  // For rightmost element:
  //    (new_v == old_v and new_i > old_i) or new_v < old_v
  //    new_v < old_v or (new_v == old_v and new_i > old_i)
  //    (new_v == old_v and new_i > old_i) or new_v > old_v
  //    new_v > old_v or (new_v == old_v and new_i > old_i)

  std::map<std::vector<Predicate>, std::pair<ReduceWithIndexType, TieBreakType>>
      m{
          // leftmost
          {{Predicate::eq, Predicate::lt, Predicate::lt},
           {ReduceWithIndexType::MIN, TieBreakType::LEFT}},
          {{Predicate::lt, Predicate::eq, Predicate::lt},
           {ReduceWithIndexType::MIN, TieBreakType::LEFT}},
          {{Predicate::lt}, {ReduceWithIndexType::MIN, TieBreakType::LEFT}},
          {{Predicate::eq, Predicate::lt, Predicate::gt},
           {ReduceWithIndexType::MAX, TieBreakType::LEFT}},
          {{Predicate::gt, Predicate::eq, Predicate::lt},
           {ReduceWithIndexType::MAX, TieBreakType::LEFT}},
          {{Predicate::gt}, {ReduceWithIndexType::MAX, TieBreakType::LEFT}},
          // rightmost
          {{Predicate::eq, Predicate::gt, Predicate::lt},
           {ReduceWithIndexType::MIN, TieBreakType::RIGHT}},
          {{Predicate::lt, Predicate::eq, Predicate::gt},
           {ReduceWithIndexType::MIN, TieBreakType::RIGHT}},
          {{Predicate::eq, Predicate::gt, Predicate::gt},
           {ReduceWithIndexType::MAX, TieBreakType::RIGHT}},
          {{Predicate::gt, Predicate::eq, Predicate::gt},
           {ReduceWithIndexType::MAX, TieBreakType::RIGHT}},
      };

  std::vector<Predicate> preds;
  std::vector<Signedness> signednesses;
  // A better way is to trace the arith.select
  // Checking the operations one by one is hacky :(
  for (auto &op : tritonReduceBlock->without_terminator()) {
    Predicate pred = Predicate::Undefined;
    Signedness signedness = Signedness::NotApplicable;
    if (auto cmpiOp = dyn_cast<arith::CmpIOp>(op)) {
      auto predi = cmpiOp.getPredicate();
      std::tie(pred, signedness) = unifyPredicateI(predi);
    }
    if (auto cmpfOp = dyn_cast<arith::CmpFOp>(op)) {
      auto predf = cmpfOp.getPredicate();
      std::tie(pred, signedness) = unifyPredicateF(predf);
    }
    if (pred != Predicate::Undefined) {
      preds.push_back(pred);
      signednesses.push_back(signedness);
    }
  }

  // check if sequence of predicates matches any sequence for min/max
  // leftmost/rightmost
  if (m.find(preds) == m.end()) {
    return llvm::failure();
  }

  assert(!signednesses.empty());
  const bool isUnsignedSrc =
      signednesses[0] == Signedness::Unsigned ||
      signednesses[signednesses.size() - 1] == Signedness::Unsigned;
  return ReduceWithIndexParams{.withIndexType = m.at(preds).first,
                               .tieBreakType = m.at(preds).second,
                               .isUnsignedSrc = isUnsignedSrc};
}

} // namespace GenericUtils
} // namespace TTOpConverters
