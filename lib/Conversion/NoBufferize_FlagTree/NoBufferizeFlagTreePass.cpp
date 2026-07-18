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

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/NoBufferize_FlagTree/NoBufferizeFlagTree.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <iostream>
#include <optional>

#define DEBUG_TYPE "no-bufferize-flagtree"
using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_NOBUFFERIZEFLAGTREE
#include "triton-shared/Conversion/NoBufferize_FlagTree/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class NoBufferizeFlagTreePass
    : public triton::impl::NoBufferizeFlagTreeBase<NoBufferizeFlagTreePass> {
  using NoBufferizeFlagTreeBase<
      NoBufferizeFlagTreePass>::NoBufferizeFlagTreeBase;

public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet localPatterns(&getContext());
    {
      TypeConverter typeConverter;
      triton::populateNoBufferizeFlagTreeConversionPatterns(localPatterns,
                                                            typeConverter);
    }
    (void)applyPatternsGreedily(module, std::move(localPatterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createNoBufferizeFlagTreePass() {
  return std::make_unique<NoBufferizeFlagTreePass>();
}
