// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "flagtree/Common/UnifiedHardware.h"

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/NoBufferize_FlagTree/NoBufferizeFlagTree.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>

#define DEBUG_TYPE "no-bufferize-flagtree"

using namespace mlir;
#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/NoBufferize_FlagTree/Passes.h.inc"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

namespace {
struct NoBufferizeConverter : public RewritePattern {
  NoBufferizeConverter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto memrefNoBufferize = [](Type t) -> bool {
      Attribute msAttr;
      if (auto memTy = dyn_cast<MemRefType>(t))
        msAttr = memTy.getMemorySpace();
      else if (auto unranked = dyn_cast<UnrankedMemRefType>(t))
        msAttr = unranked.getMemorySpace();
      else
        return false;
      if (auto intAttr = dyn_cast_or_null<IntegerAttr>(msAttr))
        return intAttr.getValue().getZExtValue() == 8;
      return false;
    };

    bool noBuf = llvm::any_of(op->getResultTypes(), memrefNoBufferize);
    if (!noBuf)
      return failure();

    op->setAttr("no_bufferize", BoolAttr::get(op->getContext(), true));
    return success();
  }
};
} // namespace

void mlir::triton::populateNoBufferizeFlagTreeConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<NoBufferizeConverter>(patterns.getContext());
}
