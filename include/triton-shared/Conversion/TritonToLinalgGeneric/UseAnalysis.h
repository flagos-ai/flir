/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef TRITON_SHARED_TritonToLinalgGeneric_USE_ANALYSIS_H
#define TRITON_SHARED_TritonToLinalgGeneric_USE_ANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <optional>

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

enum class UseType {
  Undefined, // Initial state
  DataUse,   // value used for tensor computation only
  MetaUse,   // value used for metadata only
  MixUse     // value used for both tensor computation and metadata
};

struct UseInfo : public dataflow::AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UseInfo)
  using AbstractSparseLattice::AbstractSparseLattice;

  // Lattice state transfer function
  ChangeResult meetUseType(const UseType &other) {
    if (other == UseType::Undefined) {
      return ChangeResult::NoChange;
    }

    switch (type) {
    case UseType::Undefined:
      type = other;
      return ChangeResult::Change;
    case UseType::DataUse:
    case UseType::MetaUse:
      if (type == other) {
        return ChangeResult::NoChange;
      } else {
        type = UseType::MixUse;
        return ChangeResult::Change;
      }
    case UseType::MixUse:
      return ChangeResult::NoChange;
    default:
      llvm_unreachable("bad type");
    }
  }

  ChangeResult meet(const AbstractSparseLattice &other) override {
    auto rhs = reinterpret_cast<const UseInfo *>(&other);
    return meetUseType(rhs->type);
  }

  void print(raw_ostream &os) const override {
    switch (type) {
    case UseType::DataUse:
      os << "DataUse";
      break;
    case UseType::MetaUse:
      os << "MetaUse";
      break;
    case UseType::MixUse:
      os << "MixUse";
      break;
    default:
      os << "Undefined";
    }
  }

  UseType type = UseType::Undefined;
};

/// A single operand-use propagation requested by a backend hook.
struct OperandUse {
  unsigned index;
  UseType type;
};

/// Backend extension points for the hardware-independent use analysis.
///
/// Every hook defaults to null. When unset, the corresponding generic
/// behaviour applies. A backend installs hooks to teach the analysis about its
/// own dialect ops (e.g. a backend-specific indirect store) and about its own
/// marking attributes (e.g. discrete memory access markers).
struct UseAnalysisHooks {
  /// Describe the operand-use propagation for a backend-specific operation.
  /// Return std::nullopt when the op is not a backend op and the generic
  /// dispatch should run.
  std::function<std::optional<llvm::SmallVector<OperandUse>>(Operation *)>
      classifyBackendUse;

  /// Return true when a load op must be tagged as meta use even though its
  /// result is not shaped (e.g. a backend discrete memory access load).
  std::function<bool(Operation *)> forceMetaLoad;

  /// Classify a backend op as a meta user of `result`. Return std::nullopt when
  /// `user` is not a backend op and the generic classification should run.
  std::function<std::optional<bool>(Operation *, Value)> classifyMetaUser;

  /// Return true when `op` terminates an indirect load/store trace on the
  /// backend side.
  std::function<bool(Operation *)> isIndirectBoundary;

  /// Return true for backend ops that may be marked meta even though they have
  /// more than one result.
  std::function<bool(Operation *)> isMultiResultMetaOp;

  /// Backend post-processing for a single operation after the main walk.
  std::function<void(Operation *, const mlir::DataFlowSolver &)>
      postProcessOp;

  /// Backend post-processing for the whole function after the main walk.
  std::function<void(triton::FuncOp)> postProcessFunc;
};

class UseAnalysis : public dataflow::SparseBackwardDataFlowAnalysis<UseInfo> {
public:
  UseAnalysis(mlir::DataFlowSolver &solver,
              SymbolTableCollection &symbolTable,
              const UseAnalysisHooks *hooks = nullptr)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable), hooks(hooks) {}

#if LLVM_VERSION_MAJOR >= 20
  LogicalResult visitOperation(Operation *op, ArrayRef<UseInfo *> operands,
                               ArrayRef<const UseInfo *> results) override;
#else
  void visitOperation(Operation *op, ArrayRef<UseInfo *> operands,
                      ArrayRef<const UseInfo *> results) override;
#endif

  void visitBranchOperand(OpOperand &operand) override { return; }

  void visitCallOperand(OpOperand &operand) override { return; }

  void setToExitState(UseInfo *lattice) override {
    lattice->type = UseType::Undefined;
  }

private:
  const UseAnalysisHooks *hooks = nullptr;

  void propagateUse(UseInfo *lattice, const UseType &type) {
    auto changed = lattice->meetUseType(type);
    propagateIfChanged(lattice, changed);
  }

  void propagateResults(UseInfo *lattice, ArrayRef<const UseInfo *> results) {
    auto changed = ChangeResult::NoChange;
    for (auto result : results) {
      changed |= lattice->meet(*result);
    }
    propagateIfChanged(lattice, changed);
  }
};

class MetaUseEraser : public RewritePattern {
public:
  MetaUseEraser(MLIRContext *context);

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final;
};

/// Mark `rootOp` and its meta-use producers as mixed use.
void setMixUseRecursively(Operation *rootOp, bool applyRoot = true);

bool isMetaUse(Operation *op);

bool isMixUse(Operation *op);

bool opIsIndirectLoad(Operation *op);

bool opIsIndirectCalc(Operation *op);

/// Run the use analysis on `funcOp`. Backend hooks, when provided, must outlive
/// this call.
LogicalResult runUseAnalysis(triton::FuncOp &funcOp,
                             const UseAnalysisHooks *hooks = nullptr);

} // namespace TTOpConverters

#endif // TRITON_SHARED_TritonToLinalgGeneric_USE_ANALYSIS_H
