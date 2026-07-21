//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/IR/IRMapping.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton-shared/Utils/Utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <queue>
#include <string>

#define DEBUG_TYPE "triton-ptr-analysis"

namespace mlir {

// https://triton-lang.org/main/python-api/generated/triton.language.load.html#triton.language.load
// If pointer is a block pointer defined by make_block_ptr, a tensor is
// loaded. In this case: mask and other must be None, and boundary_check and
// padding_option can be specified to control the behavior of out-of-bound
// access.
// WORKAROUND: Assume the load/store ptr operand defining op is
// triton::MakeTensorPtrOp, convert the boundaryCheck to masked dimension
static void boundaryCheckToMaskDim(OpBuilder &builder, Location loc,
                                   tts::PtrState ptrState,
                                   ArrayRef<int32_t> boundaryCheck,
                                   triton::MaskState &maskState) {

  // TODO: This can be optimized based on the boundaryCheck property, which
  // specifies along which axis the mask is required.
  for (auto i : llvm::seq<uint32_t>(0, ptrState.getRank())) {
    // The shape of the tensor is used to determine the size of the data
    // being stored, so we need to convert it to index type.
    auto dim = ofrToIndexValue(ptrState.shape[i], loc, builder);
    auto stride = ofrToIndexValue(ptrState.strides[i], loc, builder);

    auto start = ofrToIndexValue(ptrState.offsets[i], loc, builder);
    start = builder.create<arith::DivSIOp>(loc, start, stride);

    auto end = ofrToIndexValue(ptrState.sizes[i], loc, builder);
    end = builder.create<arith::AddIOp>(loc, start, end);

    Value upperBound = builder.create<arith::MinSIOp>(loc, dim, end);
    upperBound = builder.create<arith::MaxSIOp>(loc, start, upperBound);
    auto maskedShape = builder.create<arith::SubIOp>(loc, upperBound, start);

    maskState.dims.push_back(maskedShape.getResult());
  }
}

namespace tts {

int32_t PtrState::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         shape.size() == offsets.size());
  return offsets.size();
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

bool PtrState::hasModulo() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimHasModulo(i)) {
      return true;
    }
  }
  return false;
}

bool PtrState::dimHasModulo(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check modulo if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(shape[dim]);
  if (!intAttr.has_value()) {
    return true;
  }

  return intAttr.value() != 0;
}

static Value applyUnstructuredMask(Operation *op, Value ptr,
                                   triton::MaskState &mstate, Location loc,
                                   OpBuilder builder) {
  // structured mask
  if (!mstate.hasNonContinuousDim())
    return ptr;

  auto gatherScatterPtr =
      ptr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>();

  // TODO: support unstructured mask for StructuredPtr
  if (!gatherScatterPtr)
    return nullptr;

  if (mstate.getRank() != 2 || mstate.getNonContinuousDim() != 0)
    return nullptr;

  auto dims = mstate.dims;
  auto nonContinuousDim = mstate.getNonContinuousDim().value();
  auto nonContinuousMask = dims[nonContinuousDim];

  assert(nonContinuousDim == gatherScatterPtr.getGatherScatterDim());

  // Clear the mask size for gather/scatter dim.
  mstate.dims[nonContinuousDim] = OpFoldResult(builder.getI32IntegerAttr(0));
  return builder
      .create<tts::MakeGatherScatterTensorPtrOp>(
          loc, gatherScatterPtr.getBase(),
          gatherScatterPtr.getGatherScatterOffset(),
          dyn_cast<Value>(nonContinuousMask),
          gatherScatterPtr.getGatherScatterDim(), gatherScatterPtr.getSizes(),
          gatherScatterPtr.getMixedStrides(),
          gatherScatterPtr.getMixedOffsets())
      .getResult();
}

bool isNotStructured(OpFoldResult offset) {
  auto value = dyn_cast<Value>(offset);
  return value && isa<ShapedType>(value.getType());
}

bool PtrState::dimIsStructured(uint32_t dim) const {
  assert(dim < getRank());

  return !isNotStructured(offsets[dim]);
}

int32_t PtrState::getNonStructuredDim() const {
  SmallVector<int32_t> dims;
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimIsStructured(i))
      continue;
    dims.emplace_back(i);
  }
  assert(dims.size() == 1 && "must have single non-continuous dimension");
  return dims.front();
}

bool PtrState::isStructured() const {
  return llvm::all_of(
      offsets, [](OpFoldResult offset) { return !isNotStructured(offset); });
}

bool isSupportedStructuredPtr(Value ptr) {

  auto defOp = ptr.getDefiningOp();
  if (isa<tts::MakeTensorPtrOp>(defOp))
    return true;

  assert(isa<tts::MakeGatherScatterTensorPtrOp>(defOp));

  auto unstructuredPtr = cast<tts::MakeGatherScatterTensorPtrOp>(defOp);

  auto ptrType = unstructuredPtr.getType();

  auto tensorType = isa<triton::PointerType>(ptrType)
                        ? cast<triton::PointerType>(ptrType).getPointeeType()
                        : ptrType;
  auto rank = cast<RankedTensorType>(tensorType).getRank();

  return rank == 2 && unstructuredPtr.getGatherScatterDim() == 0;
}

bool PtrState::isBlockPtr() const { return !order.empty(); }

LogicalResult PtrState::addState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support adding two pointer states that both "
        "have base pointers"));
    return failure();
  }

  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) {
    // both lhs and rhs are scalars. but only one of them has a scalar value
    // Usually scalar: ptr add
    // NOTE: rank > 0 && scalar: scalar has been added in offsets
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  if (!lhsState.isStructured() && !rhsState.isStructured() &&
      lhsState.getNonStructuredDim() != rhsState.getNonStructuredDim()) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: do not support adding two pointer states "
                       "that have different non-continuous dimension"));
    return failure();
  }

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    sizes.push_back(lhsState.sizes[i]);

    auto lhsStructured = lhsState.dimIsStructured(i);
    auto rhsStructured = rhsState.dimIsStructured(i);
    if (lhsStructured && rhsStructured) {
      auto newOffset =
          addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, builder);
      offsets.push_back(newOffset);
      auto newStride =
          addOFRs(lhsState.strides[i], rhsState.strides[i], loc, builder);
      strides.push_back(newStride);
      continue;
    }
    assert(!(lhsState.hasModulo() && rhsState.hasModulo()) &&
           "PtrAnalysis: do not support adding two pointer states "
           "that both have modulo");

    strides.push_back(builder.getIndexAttr(1));
    if (!lhsStructured && !rhsStructured) {
      auto newOffset =
          builder
              .create<arith::AddIOp>(loc, dyn_cast<Value>(lhsState.offsets[i]),
                                     dyn_cast<Value>(rhsState.offsets[i]))
              ->getResult(0);
      offsets.push_back(newOffset);
      continue;
    }

    auto [lhsOffset, rhsOffset] =
        lhsState.dimIsStructured(i)
            ? std::tuple<Value, Value>{ofrToIndexValue(lhsState.offsets[i], loc,
                                                       builder),
                                       dyn_cast<Value>(rhsState.offsets[i])}
            : std::tuple<Value, Value>{
                  ofrToIndexValue(rhsState.offsets[i], loc, builder),
                  dyn_cast<Value>(lhsState.offsets[i])};

    auto tensorType = cast<RankedTensorType>(rhsOffset.getType());
    lhsOffset = builder.create<arith::IndexCastOp>(
        loc, tensorType.getElementType(), lhsOffset);
    lhsOffset = builder.create<triton::SplatOp>(loc, tensorType, lhsOffset);

    auto newOffset =
        builder.create<arith::AddIOp>(loc, lhsOffset, rhsOffset)->getResult(0);
    offsets.push_back(newOffset);
  }

  // AddPtr where both lhs and rhs containing modulo operators not supported
  if (lhsState.hasModulo() && rhsState.hasModulo()) {
    // May modulo different value
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: do not support adding two pointer states "
                       "that both have modulo"));
    return failure();
  }

  if (lhsState.hasModulo() || rhsState.hasModulo()) {
    // visitOperandSplat and visitOperandExpandDims should enforce below
    assert(lhsState.getRank() <= 2);
  }

  // dealing with modulo:
  // - If lhs has no modulo, skip
  // - If rhs has zero offset on dim i, we can just use lhs's modulo
  // - If i == 0 and rhs is the result of a splat, we will allow the add. This
  // is because the user may be trying to express adding a constant offset to
  // increment dim1, but pointer analysis cannot differentiate dim1 vs dim0 in
  // this case.
  // - Else, the analysis fails

  // An example for the 3rd condition above can look like:
  // %0 = tt.splat %scalar
  // %1 = tt.splat %ptr
  // %2 = tt.arange
  // %3 = arith.remsi %2, %size
  // %4 = tt.addptr %1, %3
  // %5 = tt.addptr %4, %0
  // %5 may also occur in a loop to increment %4 every iteration.

  // Note that this is not bullet-proof. E.g., broken IR can actually increment
  // dim0 while dim0 already has modulo, since Triton offsets are element-wise
  // and not in unit of lower dimensions. However, this is highly unlikely but
  // the analysis will provide wrong result. Hence we provide a warning in this
  // case.
  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (rhs->hasModulo()) {
    std::swap(lhs, rhs);
  }

  for (uint64_t i = 0; i < lhs->getRank(); i++) {
    if (!lhs->dimHasModulo(i)) {
      shape.push_back(lhs->shape[i]);
    } else if (hasConstZero(rhs->offsets[i])) {
      shape.push_back(lhs->shape[i]);
    } else if (i == 0 && lhs->getRank() == 2 && rhs->scalar) {
      shape.push_back(lhs->shape[1]);
      shape.push_back(lhs->shape[0]);
      LLVM_DEBUG(op->emitWarning(
          "PtrAnalysis: allowing adding pointer state with modulo in dim 0 to "
          "another pointer state with offset in dim 0.\nPlease verify the "
          "operand that contains a scalar is meant to increment pointers in "
          "dim1. If that is not the case it WILL LEAD TO WRONG COMPILATION "
          "RESULTS.\n\nTo avoid this warning, use expand_dims (instead of "
          "splat) to explicitly specify which dimension contains the scalar."));
      break;
    } else {
      LLVM_DEBUG(op->emitRemark(
          "PtrAnalysis: do not support adding to operand with modulo"));
      return failure();
    }
  }

  return success();
}

void PtrState::dump() const {
  llvm::dbgs() << "PtrState: ";
  if (source) {
    llvm::dbgs() << "source: " << source << "\n";
  }
  if (scalar) {
    llvm::dbgs() << "scalar: " << scalar << "\n";
  }

  llvm::dbgs() << "offsets:\n";
  llvm::interleave(offsets, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nstrides:\n";
  llvm::interleave(strides, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nsizes:\n";
  llvm::interleave(sizes, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nshape:\n";
  llvm::interleave(shape, llvm::dbgs(), "\n");
  llvm::dbgs() << "\norder:\n";
  llvm::interleave(order, llvm::dbgs(), "\n");

  isStructured()
      ? llvm::dbgs() << "structured\n"
      : llvm::dbgs() << "dim " << getNonStructuredDim() << " not structured\n";

  llvm::dbgs() << "\n";
}

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  auto loc = op->getLoc();

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  if (lhsState.source && rhsState.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support multiplying base pointers"));
    return failure();
  }

  // currently do not support both tensors are effectively non-scalar
  if (!lhsState.scalar && !rhsState.scalar) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: only support multiplying pointer states when one of "
        "them represent a scalar"));
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  if (lhsState.scalar && rhsState.scalar) {
    scalar =
        builder.create<arith::MulIOp>(loc, lhsState.scalar, rhsState.scalar);
  }

  assert(rhs->scalar && "rhs always scalar");
  for (uint64_t i = 0; i < lhs->sizes.size(); i++) {
    OpFoldResult newShape =
        mulOFRValue(lhs->shape[i], rhs->scalar, loc, builder);
    shape.push_back(newShape);
    sizes.push_back(lhs->sizes[i]);

    if (lhs->dimIsStructured(i)) {
      auto newOffset = mulOFRValue(lhs->offsets[i], rhs->scalar, loc, builder);
      offsets.push_back(newOffset);
      auto newStride = mulOFRValue(lhs->strides[i], rhs->scalar, loc, builder);
      strides.push_back(newStride);
      continue;
    }

    assert(!lhs->hasModulo() &&
           "should not have non-structured dimension with modulo");

    // NOTE: Can also consider mul scalar on stride
    auto lhsOffset = dyn_cast<Value>(lhs->offsets[i]);

    auto tensorType = cast<RankedTensorType>(lhsOffset.getType());
    auto rhsScalar = builder.create<arith::IndexCastOp>(
        loc, tensorType.getElementType(), rhs->scalar);
    auto rhsSplatTensor =
        builder.create<triton::SplatOp>(loc, tensorType, rhsScalar);
    OpFoldResult newOffset =
        builder.create<arith::MulIOp>(loc, lhsOffset, rhsSplatTensor)
            ->getResult(0);

    offsets.push_back(newOffset);
    strides.push_back(lhs->strides[i]);
  }

  if (rhs->hasModulo()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support multiplying pointer states that has "
        "modulos"));
    return failure();
  }

  return success();
}

LogicalResult PtrState::subState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: do not support subtract two pointer states that both "
        "have base pointers"));
    return failure();
  }

  if (!lhsState.source && rhsState.source) {
    op->emitRemark("PtrAnalysis: scalar minus pointer is not meaningful");
    return failure();
  }

  // WORKAROUND: avoid other non-structured cases.
  if (!lhsState.isStructured() || !rhsState.isStructured()) {
    return failure();
  }

  source = lhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto subOp =
        builder.create<arith::SubIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = subOp.getResult();
  } else if (lhsState.getRank() == 0) { // both lhs and rhs are scalars
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    auto newOffset =
        subOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, builder);
    offsets.push_back(newOffset);

    auto newStride =
        subOFRs(lhsState.strides[i], rhsState.strides[i], loc, builder);
    strides.push_back(newStride);

    sizes.push_back(lhsState.sizes[i]);
  }

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    shape.push_back(lhsState.shape[i]);
  }

  return success();
}

Operation *PtrState::createTTSMakeTensorPtrOp(OpBuilder &builder,
                                              Location loc) {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(sizes[i]);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }

  if (!isStructured()) {
    int nonContinuousDim = getNonStructuredDim();
    Value nonContinuousOffset = cast<Value>(offsets[nonContinuousDim]);
    auto op = builder.create<tts::MakeGatherScatterTensorPtrOp>(
        loc, source, nonContinuousOffset, nonContinuousDim, staticSizes,
        strides, offsets);

    LLVM_DEBUG({
      llvm::dbgs() << "creating tts::make_gather_scatter_tensor_ptr:\n";
      op->dump();
      op->getParentOfType<triton::FuncOp>()->dump();
    });
    return op;
  }

  auto op = builder.create<tts::MakeTensorPtrOp>(
      loc, source, staticSizes, strides, offsets, shape, order);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_tensor_ptr:\n";
    op->dump();
    op->getParentOfType<triton::FuncOp>()->dump();
  });

  return op;
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed())
    return failure();

  // Checking for higher dimension is done in addState below
  if ((lhsState.getRank() == 1 && lhsState.hasModulo()) ||
      (rhsState.getRank() == 1 && rhsState.hasModulo())) {
    LLVM_DEBUG(addOp->emitRemark(
        "PtrAnalysis: do not support this pattern: a + arange(0, K) % M"));
    return failure();
  }

  return state.addState(lhsState, rhsState, addOp, builder);
}

LogicalResult PtrAnalysis::visitOperandSub(arith::SubIOp subOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(subOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(subOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (lhsState.hasModulo() || rhsState.hasModulo()) {
    LLVM_DEBUG(
        subOp->emitRemark("PtrAnalysis: do not support modulo for subi op\n"));
    return failure();
  }

  // Checking for higher dimension is done in subState below
  if ((lhsState.getRank() == 1 && lhsState.hasModulo()) ||
      (rhsState.getRank() == 1 && rhsState.hasModulo())) {
    subOp->emitRemark(
        "PtrAnalysis: do not support this pattern: a - arange(0, K) % M");
    return failure();
  }

  return state.subState(lhsState, rhsState, subOp, builder);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  return state.mulState(lhsState, rhsState, mulOp, builder);
}

LogicalResult PtrAnalysis::visitOperandDivSI(arith::DivSIOp divOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());

  // Divisor
  PtrState rhsState;
  if (visitOperand(divOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    LLVM_DEBUG(divOp->emitRemark(
        "PtrAnalysis: only support cases when divisor contains scalar"));
    return failure();
  }

  // Check if divisor is a constant value
  auto rhsInt = getConstValue(rhsState.scalar);
  if (!rhsInt.has_value()) {
    LLVM_DEBUG(divOp->emitRemark(
        "PtrAnalysis: only support cases when divisor is a constant"));
    return failure();
  }

  int64_t divisor = rhsInt.value();
  assert(divisor != 0 && "Divisor cannot be 0");

  if (visitOperand(divOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  // Check if dividend is 1D or 2D with one dimension being 1
  auto resultType = cast<ShapedType>(divOp.getType());
  if (resultType.getRank() > 2) {
    LLVM_DEBUG(divOp->emitRemark(
        "PtrAnalysis: only support 1D or 2D tensors for division"));
    return failure();
  }

  if (state.hasModulo()) {
    LLVM_DEBUG(divOp->emitRemark(
        "PtrAnalysis: do not support division on a state with modulo (e.g. "
        "(x % N) / M pattern)"));
    return failure();
  }
  // Apply division to offsets and strides
  auto constDiv =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(divisor));

  if (state.scalar) {
    state.scalar =
        builder
            .create<arith::DivSIOp>(
                loc, ofrToIndexValue(state.scalar, loc, builder), constDiv)
            .getResult();
    return failure();
  }

  auto shape = resultType.getShape();
  if (resultType.getRank() == 2 && shape[1] != 1) {
    LLVM_DEBUG(divOp->emitRemark("PtrAnalysis: only support outernest dim "
                                 "division, otherwise unstructured"));
    return failure();
  }

  for (uint32_t i = 0; i < state.getRank(); i++) {
    if (shape[i] == 1) {
      assert(hasConstZero(state.offsets[i]));
      state.offsets[i] = builder.getIndexAttr(0);
      state.strides[i] = builder.getIndexAttr(0);
      continue;
    }
    // Divide offsets by divisor
    state.offsets[i] =
        resultType.getRank() == 2
            ? builder
                  .create<tensor::CollapseShapeOp>(
                      loc, divOp, SmallVector<ReassociationIndices>{{0, 1}})
                  ->getResult(0)
            : divOp->getResult(0);
    state.strides[i] = builder.getIndexAttr(1);
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    LLVM_DEBUG(remOp->emitRemark(
        "PtrAnalysis: only support cases when rhs of remainder "
        "contains scalar"));
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  if (state.hasModulo()) {
    LLVM_DEBUG(remOp->emitRemark(
        "PtrAnalysis: do not support multiple modulo within an expression"));
    return failure();
  }

  if (state.getRank() == 1) {
    // Apply the modulo before expanding shape, the common pattern is
    // offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    // a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
    // stride_ak)
    state.shape.back() = rhsState.scalar;
  } else if (state.getRank() == 2) {
    // torch inductor expands the tensor shape before applying the modulo.
    //
    // We only support either:
    // - (tl.arange(0, end)[:, None] % mod), or
    // - (tl.arange(0, end)[None, :] % mod)
    //
    // In both cases, we apply the modulo to the non-singleton dimension.
    auto shape = cast<TensorType>(remOp.getResult().getType()).getShape();
    if (shape[0] == 1) {
      state.shape[1] = rhsState.scalar;
    } else if (shape[1] == 1) {
      state.shape[0] = rhsState.scalar;
    } else {
      LLVM_DEBUG(remOp->emitRemark(
          "PtrAnalysis: taking modulo on a 2D tensor with no singleton "
          "dimension not supported"));
      return failure();
    }
  } else {
    LLVM_DEBUG(remOp->emitRemark("PtrAnalysis: unsupported modulo pattern"));
    return failure();
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());
  return visitOperand(extOp.getIn(), state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  state.offsets.push_back(builder.getIndexAttr(start));
  state.sizes.push_back(builder.getIndexAttr(shape[0]));
  state.strides.push_back(builder.getIndexAttr(stride));
  state.shape.push_back(builder.getIndexAttr(0));
  return success();
}

LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  assert(state.isEmpty());

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  state.offsets.insert(state.offsets.begin() + axis, builder.getIndexAttr(0));
  state.sizes.insert(state.sizes.begin() + axis, builder.getIndexAttr(1));
  state.strides.insert(state.strides.begin() + axis, builder.getIndexAttr(0));
  state.shape.insert(state.shape.begin() + axis, builder.getIndexAttr(0));

  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG(expandDimsOp->emitRemark(
        "PtrAnalysis: unsupported scenario where expand_dims result "
        "has modulo and rank > 2"));
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    LLVM_DEBUG(broadcastOp->emitRemark(
        "PtrAnalysis: Unsupported broadcast source type"));
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  for (size_t i = 0; i < dstShape.size(); i++) {
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i]) {
      state.sizes[i] = builder.getIndexAttr(dstShape[i]);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
    for (auto s : dstShape) {
      state.offsets.push_back(builder.getIndexAttr(0));
      state.sizes.push_back(builder.getIndexAttr(s));
      state.strides.push_back(builder.getIndexAttr(0));
      state.shape.push_back(builder.getIndexAttr(0));
    }
  } else {
    LLVM_DEBUG(splatOp->emitRemark("PtrAnalysis: unsupported splat pattern"));
    return failure();
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension
  if (state.scalar)
    state.offsets[0] = state.scalar;

  if (state.hasModulo() && state.getRank() > 2) {
    LLVM_DEBUG(splatOp->emitRemark(
        "PtrAnalysis: unsupported scenario where splat result "
        "has modulo and rank > 2"));
    return failure();
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    return failure();
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  assert(ptrState.source && "ptr field should provide source / base pointer");

  assert(ptrState.getRank() == offsetState.getRank() &&
         "ptr and offset field should have the same rank");

  return state.addState(ptrState, offsetState, addptrOp, builder);
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  assert(state.isEmpty());
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(builder, constAttr,
                                                builder.getIndexType(), loc);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    if (i == 0) {
      state.offsets.push_back(constOp.getResult());
    } else {
      state.offsets.push_back(builder.getIndexAttr(0));
    }

    state.sizes.push_back(builder.getIndexAttr(resultType.getShape()[i]));
    state.strides.push_back(builder.getIndexAttr(0));
    state.shape.push_back(builder.getIndexAttr(0));
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {

  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();
  state.offsets = makeTPtrOp.getMixedOffsets();
  state.sizes = makeTPtrOp.getMixedSizes();
  state.strides = makeTPtrOp.getMixedStrides();
  state.shape = makeTPtrOp.getMixedShape();
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());

  return success();
}

LogicalResult
PtrAnalysis::visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();

  if (makeTPtrOp.getOrder().empty()) {
    LLVM_DEBUG(makeTPtrOp->emitRemark(
        "PtrAnalysis: expect tt.make_tensor_ptr to have order field set"));
    return failure();
  }

  auto resType = cast<triton::PointerType>(makeTPtrOp.getResult().getType());
  auto pointeeType = cast<ShapedType>(resType.getPointeeType());
  auto shape = pointeeType.getShape();

  for (int64_t i = 0; i < pointeeType.getRank(); i++) {
    state.sizes.push_back(builder.getIndexAttr(shape[i]));

    auto strideCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getStrides()[i]);
    state.strides.push_back(strideCst.getResult());

    auto offsetCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getOffsets()[i]);

    auto scaledOffset = builder.create<arith::MulIOp>(
        loc, offsetCst.getResult(), strideCst.getResult());
    state.offsets.push_back(scaledOffset.getResult());

    auto shapeCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getShape()[i]);
    state.shape.push_back(shapeCst.getResult());
  }
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());
  assert(state.isBlockPtr() &&
         "tt.make_tensor_ptr pointer state should describe a block pointer");

  return success();
}

LogicalResult PtrAnalysis::visitOperandForOp(scf::ForOp forOp, Value operand,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {

  auto it = llvm::find(forOp->getResults(), operand);
  auto index = std::distance(forOp->getResults().begin(), it);

  auto newState = getLoopResultPtrState(forOp, index);
  if (failed(newState)) {
    LLVM_DEBUG(forOp.emitError(
        "Rewrite for-op failed. Could not find PtrState returned by "
        "the loop."));
    return failure();
  }

  state = newState.value();
  return success();
}

LogicalResult PtrAnalysis::visitOperandIntToPtr(triton::IntToPtrOp op,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperandBitcast(triton::BitcastOp op,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  auto resType = op.getResult().getType();
  if (isa<ShapedType>(resType)) {
    return visitOperand(op.getSrc(), state, loc, builder);
  }
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {

  if (knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return success();
  } else if (isa<IndexType>(operand.getType())) {
    state.scalar = operand;
    return success();
  }

  if (isa<triton::PointerType>(operand.getType())) {
    // A scalar pointer can either be produced by AddPtrOp or a block
    // argument
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                  builder);
      } else if (auto castOp = dyn_cast<triton::BitcastOp>(op)) {
        return visitOperandBitcast(castOp, state, loc, builder);
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        return visitOperandIntToPtr(intToPtrOp, state, loc, builder);
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        llvm_unreachable("Unexpected operand defining operation tts.make_tptr");
      } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
        state.source = selectOp.getResult();
        return success();
      } else {
        LLVM_DEBUG(op->emitRemark("Unexpected operand defining operation"));
        return failure();
      }
    } else {
      state.source = operand;
      state.scalar = builder.create<arith::ConstantIndexOp>(loc, 0);
      return success();
    }
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::SubIOp>()) {
    return visitOperandSub(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    return visitOperandDivSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::ForOp>()) {
    return visitOperandForOp(op, operand, state, loc, builder);
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      return failure();
    }

    // This operand must be an iter-arg of an inner-loop in a multiple-level
    // nested loop, which means its PtrState must have already been populated
    // during rewriteForOp of the parent loop.
    state = knownPtrs[operand];
    return success();
  } else {
    auto resultType = operand.getType();
    if (isa<RankedTensorType>(resultType) &&
        cast<RankedTensorType>(resultType).getRank() == 1) {
      assert(state.isEmpty());
      LLVM_DEBUG(llvm::dbgs() << "\n unsupported: PtrAnalysis directly handle "
                                 "ranked-1 tensors as offset\n";);
      auto shape = cast<RankedTensorType>(resultType).getShape();
      state.offsets.push_back(operand);
      state.sizes.push_back(builder.getIndexAttr(shape.front()));
      state.strides.push_back(builder.getIndexAttr(1));
      state.shape.push_back(builder.getIndexAttr(0));
      return success();
    }

    LLVM_DEBUG(llvm::dbgs()
                   << "PtrAnalysis: encountered addptr operand produced by an "
                      "unsupported operation\n";
               operand.dump());
    return failure();
  }
}

LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  knownPtrs[op.getResult()] = state;

  if (isa<RankedTensorType>(op.getPtr().getType())) {
    auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
    ptrMap.map(op.getResult(), maketptrOp->getResult(0));
  } else {
    // record the ptr as we have visited and built up the state for this scalar
    // pointer, which may be used by rewriteForOp later.
    ptrMap.map(op.getResult(), op.getResult());
  }
  return success();
}

LogicalResult PtrAnalysis::rewriteBitcastOp(triton::BitcastOp op) {
  LLVM_DEBUG({
    llvm::dbgs() << "Rewriting bitcast op:\n";
    op->dump();
  });

  if (!triton::isPtrTypeLike(op.getOperand().getType()))
    return failure();

  OpBuilder builder(op);

  PtrState state;
  if (visitOperandBitcast(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  knownPtrs[op.getResult()] = state;

  if (isa<RankedTensorType>(op.getOperand().getType())) {
    auto resType = op.getType();
    if (auto tensorType = llvm::dyn_cast<TensorType>(resType)) {
      resType = tensorType.getElementType();
    }
    auto newBitcast =
        builder.create<triton::BitcastOp>(op->getLoc(), resType, state.source);
    state.source = newBitcast;
    auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
    ptrMap.map(op.getResult(), maketptrOp->getResult(0));
  } else {
    ptrMap.map(op.getResult(), op.getResult());
  }
  return success();
}

LogicalResult PtrAnalysis::rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandMakeTensorPtr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), maketptrOp->getResult(0));
  return success();
}

LogicalResult PtrAnalysis::rewriteAdvanceOp(triton::AdvanceOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperand(op->getOperand(0), state, loc, builder).failed()) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: Failed to analyze ptr of tt.advance"));
    return failure();
  }
  assert(state.isBlockPtr() &&
         "tt.advance pointer state should describe a block pointer");

  auto incrementOffsets = op.getOffsets();

  SmallVector<OpFoldResult> newOffsets;
  for (auto [increment, offset, stride] :
       llvm::zip(incrementOffsets, state.offsets, state.strides)) {
    Value offsetValue;
    if (auto offsetIntAttr = getIntAttr(offset)) {
      auto constOp = builder.create<arith::ConstantOp>(
          loc, builder.getIndexAttr(offsetIntAttr.value()));
      offsetValue = constOp.getResult();
    } else {
      offsetValue = cast<Value>(offset);
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), increment);
    auto mulOp = builder.create<arith::MulIOp>(loc, castOp.getResult(),
                                               cast<Value>(stride));
    auto addOp =
        builder.create<arith::AddIOp>(loc, mulOp.getResult(), offsetValue);
    newOffsets.push_back(addOp.getResult());
  }

  state.offsets = SmallVector<OpFoldResult>(newOffsets);

  auto newOp = state.createTTSMakeTensorPtrOp(builder, loc);
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), newOp->getResult(0));
  return success();
}

static bool isPointerType(Type t) {
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensor.getElementType());
  }
  return isa<triton::PointerType>(t);
}

FailureOr<PtrState> PtrAnalysis::getLoopInitArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto ptr = forOp.getInitArgs()[index];

  // If the pointer into the scf.for was defined by tts.get_structured_state,
  // we can get the pointer state from the original pointer (the op's input):
  //
  // %ptr, %offset_1, %offset_2,..., %stride_1, %stride_2,... =
  // tts.get_structured_state %original
  // scf.for ... (%ptr) {...}
  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (knownPtrs.count(originalPtr)) {
      return knownPtrs[originalPtr];
    }
  }

  // For nested loops scenarios, a pointer in init-args can be returned from
  // another loop of the same level:
  // e.g.:
  // clang-format off
  //  %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //    %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
  //    }
  //    %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
  //      %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      ...
  //    }
  //    ...
  //  }
  // clang-format on
  // Notice %arg8 = %23 comes from the return value of the first loop.
  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  // If the pointer isn't defined by tts.get_structured_state nor another loop,
  // it means the current pointer is an iterarg of the outer loop.
  // In such cases, the outer loops would have already set up the PtrState for
  // us already.
  //
  // scf.for iterargs(%ptr = %init_arg) {
  //    scf.for iterargs(%ptr1 = %ptr) {  <--- we're dealing with `%ptr1` here.
  //          ...
  //    }
  // }
  if (knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return knownPtrs[ptr];
  }

  return failure();
}

PtrState PtrAnalysis::reconcileLoopPtrState(
    scf::ForOp forOp, size_t iterArgIndex, const PtrState &state,
    llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = iterArgIndex + 1;
  if (newState.getRank() == 0) {
    // rewriteGetStructuredStateOp will return a scalar constant 0 in the
    // newState.scalar for rank 0 case. Therefore, we can always get the correct
    // newState.scalar by calling getReplacementVal with the iterArgIndex of the
    // current loop.
    // For scalar pointers, the scalar contains the offset and is
    // the only relevant newState that could be updated by the loop.
    newState.scalar = getReplacementVal(forOp, cnt);
  } else {
    int rank = newState.getRank();

    std::optional<int64_t> nonStructuredDim =
        newState.isStructured()
            ? std::nullopt
            : std::optional<int64_t>(newState.getNonStructuredDim());
    for (int i = 0; i < rank; i++) {
      // Unstructured dim will be tensor type which is not compatible with index
      // type of tts::getStructuredStateop
      if (nonStructuredDim && i == nonStructuredDim.value()) {
        forOp.getRegionIterArg(cnt).setType(
            dyn_cast<Value>(newState.offsets[i]).getType());
        forOp->getResult(cnt).setType(
            dyn_cast<Value>(newState.offsets[i]).getType());
      }
      newState.offsets[i] = getReplacementVal(forOp, cnt++);
    }

    for (auto &stride : newState.strides) {
      stride = getReplacementVal(forOp, cnt++);
    }
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getLoopIterArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  // WORKAROUND: avoid other non-structured cases. Since unstructured offset
  // need ShapeType which is not compatible with iterarg (index type)
  if (!state->isStructured() &&
      (state->getRank() != 2 || state->getNonStructuredDim() != 0)) {
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op.getRegionIterArg(index); });
}

FailureOr<PtrState> PtrAnalysis::getLoopResultPtrState(scf::ForOp forOp,
                                                       size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  // WORKAROUND: avoid other non-structured cases. Since unstructured offset
  // need ShapeType which is not compatible with result (index type)
  if (!state->isStructured() &&
      (state->getRank() != 2 || state->getNonStructuredDim() != 0)) {
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op->getResult(index); });
}

LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {
  for (auto [i, arg] : llvm::enumerate(op.getRegionIterArgs())) {
    if (!maybeStructuredArgs.contains(arg)) {
      continue;
    }

    auto state = getLoopIterArgPtrState(op, i);
    if (failed(state)) {
      // Because the maybeStructuredArgs may contain values that are not
      // considered structured by PtrAnalysis, failing to retrieve the PtrState
      // should not fail the rewrite process.
      // We emit an error for diagnostics and debugging purposes.
      LLVM_DEBUG(op->emitWarning(
          "Rewrite for-op failed. Could not find PtrState for iter-arg index " +
          std::to_string(i)));
      continue;
    }

    // Save the current init arg's PtrState
    knownPtrs[arg] = state.value();

    // For tensors of pointers, create a tts.make_tptr at the beginning of the
    // loop body that correspond to this region iter arg. In case it is used
    // by tt.load/tt.store in the loop body before pointer updates, this will
    // make sure rewriteLoadOp/rewriteStoreOp can use the analysis result.
    // E.g., given the following input (%tensor_of_ptr is a block arg):
    // scf.for (%tensor_of_ptr) {
    //   %data = tt.load %tensor_of_ptr
    //   // more operations to update %tensor_of_ptr
    // }
    // We may produce the following output:
    // scf.for (%base_ptr, %stride, %offset) {
    //   %tensor_of_ptr = tts.make_tptr(%base_ptr, %stride, %offset)
    //   %data = tts.load %tensor_of_ptr
    //   // more operations to update %offset
    // }
    // If %tensor_of_ptr is not used (i.e., %tensor_of_ptr is updated before
    // used in the original IR), it will simply be removed by
    // canonicalization.

    // For scalar pointers, there is no need to create a tts.addptr at the
    // beginning of the loop body. We don't lower tt.load and tt.store on
    // scalars in this pass; pointer arithmetics can also just use the
    // original pointer.
    // Note that there can be tensor of indices in iter-arg, so we only create
    // the make_tensor_ptr op when the arg is of pointer type.
    if (isPointerType(arg.getType()) && state->getRank() != 0) {
      OpBuilder builder(op.getRegion());
      auto maketptrOp = state->createTTSMakeTensorPtrOp(builder, op.getLoc());
      ptrMap.map(arg, maketptrOp->getResult(0));
    }
  }

  // Recursively rewrite the inner ops
  if (rewriteOp(op).failed()) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting for op"));
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::rewriteGetStructuredStateOp(tts::GetStructuredStateOp op) {
  auto tritonValue = op->getOperand(0);

  OpBuilder builder(op);

  // If this triton value isn't known, it means PtrAnalysis has failed to
  // analyze this pointer. In such cases, simply remap all uses of the
  // structured value back to its original triton value.
  if (!knownPtrs.contains(tritonValue)) {
    LLVM_DEBUG(op.emitRemark(
        "Rewrite GetStructuredStateOp failed. Could not find PtrState."));
    auto numResults = op.getNumResults();
    SmallVector<Value> replacements(
        numResults, builder.create<arith::ConstantOp>(op.getLoc(),
                                                      builder.getIndexAttr(0)));
    replacements.front() = tritonValue;
    op.getResults().replaceAllUsesWith(replacements);
    return failure();
  }

  tts::PtrState state = knownPtrs[tritonValue];

  // WORKAROUND: avoid other non-structured cases. Since unstructured offset
  // need ShapeType which is not compatible with iterarg (index type)
  if (!state.isStructured() &&
      (state.getRank() != 2 || state.getNonStructuredDim() != 0)) {
    auto numResults = op.getNumResults();
    SmallVector<Value> replacements(
        numResults, builder.create<arith::ConstantOp>(op.getLoc(),
                                                      builder.getIndexAttr(0)));
    replacements.front() = tritonValue;
    op.getResults().replaceAllUsesWith(replacements);
    return failure();
  }

  Value remappedValue =
      ptrMap.contains(tritonValue) ? ptrMap.lookup(tritonValue) : tritonValue;

  SmallVector<Value> replacements{remappedValue};

  if (state.getRank() == 0) {
    // For scalar pointers, the scalar contains the offset and is the only
    // relevant state that could be updated by the loop.
    if (state.scalar) {
      replacements.push_back(state.scalar);
    } else {
      // This operand is a pointer directly from the kernel arguments.
      // Use offset 0.
      replacements.push_back(builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getIndexAttr(0)));
    }
  } else {
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }
  }

  op->replaceAllUsesWith(replacements);
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op,
                                         bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "loadOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: scalar loadOp will not be rewritten"));
    return failure();
  }

  if (!isSupportedStructuredPtr(ptr)) {
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);
  Value scalarOther;

  OpBuilder builder(op);
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      LLVM_DEBUG(op->emitRemark("MaskAnalysis failed"));
      return failure();
    }

    ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
    if (!ptr)
      return failure();

    dims = mstate.dims;
  }

  auto boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck.empty()) {
    assert(dims.empty() && "Mask and boundary check cannot be used together");

    boundaryCheckToMaskDim(builder, loc, knownPtrs.at(op.getPtr()),
                           boundaryCheck, mstate);
    dims = mstate.dims;
  }

  if (other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = triton::getScalarValue(other, loc, builder);
    if (!scalarOther) {
      LLVM_DEBUG(op->emitRemark("other value used in masked load produced by "
                                "unsupported instruction"));
      return failure();
    }
  }

  auto loadOp = builder.create<tts::LoadOp>(loc, ptr, dims, scalarOther);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::load:\n";
    loadOp->dump();
    loadOp->getParentOfType<triton::FuncOp>()->dump();
  });

  op.replaceAllUsesWith(loadOp.getResult());
  op->erase();
  return success();
}

// Structured values from the TritonStructuredDialect have offsets and strides
// that might change in each loop iteration and hence will appear in an scf.for
// iter-args like so:
//
// %structured, %offsets, %strides  = tts.get_structured_state
// scf.for (%arg0 = %structured, %arg1 = %offsets, %arg2 = %strides) {
//   %a = %arg0 + 1
//   %b = %b + 2
//   scf.for (%arg1 = %b) {
//      ...
//   }
// }
//
// In `rewriteForOp`, we have to recognize such structured values in order to
// rewrite their PtrState accordingly. Previously, only values of Pointer-like
// type (e.g.: tensor<tt.ptr<>> or tt.ptr<tensor<>>), so detecting these values
// is as easy as checking the type.
//
// Now, tensor of indices could also appear in a loop's iter-arg. To reliably
// detect all such cases, we perform a BFS-like traversal of the IR where the
// sources are the results of `tts.get_structured_state`. All values that
// originate from the results of `tts.get_structured_state` are consider
// "maybeStructured". If a loop's iter-arg is considered "maybeStructured", we
// must set up their PtrState during `rewriteForOp`.
void PtrAnalysis::initializeMaybeStructuredArgs(Operation *op) {
  std::queue<Value> q;
  DenseSet<Value> visited;

  op->walk([&q, &visited](tts::GetStructuredStateOp getStateOp) {
    Value value = getStateOp->getResult(0);
    visited.insert(value);
    q.push(value);
  });

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto user : v.getUsers()) {
      // scf.for is a special case. We have 2 set of values to consider:
      // - iter-args
      // - loop results
      // for every init arg that originates from a `tts.get_structured_state`
      // op, its corresponding iter-arg and loop result will also be considered
      // "maybeStructured".
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        auto it = llvm::find(forOp.getInitArgs(), v);

        if (it == forOp.getInitArgs().end()) {
          continue;
        }

        auto argIndex = std::distance(forOp.getInitArgs().begin(), it);
        auto iterArg = forOp.getRegionIterArg(argIndex);
        auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);

        SmallVector<Value> neighbors{iterArg, tiedLoopRes};
        for (auto neighbor : neighbors) {
          maybeStructuredArgs.insert(neighbor);
          if (!visited.contains(neighbor)) {
            visited.insert(neighbor);
            q.push(neighbor);
          }
        }

      } else {
        for (auto res : user->getResults()) {
          if (res.getType() != v.getType()) {
            continue;
          }
          maybeStructuredArgs.insert(res);
          if (!visited.contains(res)) {
            visited.insert(res);
            q.push(res);
          }
        }
      }
    }
  }
}

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op,
                                          bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getValue();
  auto mask = op.getMask();
  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "storeOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(
        op->emitRemark("PtrAnalysis: scalar storeOp will not be rewritten"));
    return failure();
  }

  if (!isSupportedStructuredPtr(ptr)) {
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);

  OpBuilder builder(op);

  // Analyze the mask operand to determine at runtime the size of the data
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      LLVM_DEBUG(op->emitRemark("MaskAnalysis failed"));
      return failure();
    }

    ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
    if (!ptr)
      return failure();

    dims = mstate.dims;
  }

  auto boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck.empty()) {
    assert(dims.empty() && "Mask and boundary check cannot be used together");

    boundaryCheckToMaskDim(builder, loc, knownPtrs.at(op.getPtr()),
                           boundaryCheck, mstate);
    dims = mstate.dims;
  }

  auto storeOp = builder.create<tts::StoreOp>(loc, ptr, val, dims);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::store:\n";
    storeOp->dump();
    storeOp->getParentOfType<triton::FuncOp>()->dump();
  });

  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteAtomicRMWOp(triton::AtomicRMWOp op,
                                              bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getVal();
  auto mask = op.getMask();
  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "AtomicCASOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: scalar AtomicCASOp will not be rewritten"));
    return failure();
  }

  if (!isSupportedStructuredPtr(ptr)) {
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);

  OpBuilder builder(op);

  // Analyze the mask operand to determine at runtime the size of the data
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      LLVM_DEBUG(op->emitRemark("MaskAnalysis failed"));
      return failure();
    }

    ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
    if (!ptr)
      return failure();

    dims = mstate.dims;
  }

  auto atomicRMWOp = builder.create<tts::AtomicRMWOp>(
      loc, op.getType(), ptr, val, dims, op.getAtomicRmwOpAttr(),
      op.getSemAttr(), op.getScopeAttr());

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::atomic_rmw:\n";
    atomicRMWOp->dump();
  });

  op.replaceAllUsesWith(atomicRMWOp.getResult());
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteAtomicCASOp(triton::AtomicCASOp op) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto cmp = op.getCmp();
  auto val = op.getVal();

  auto loc = op.getLoc();

  if (!ptr) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: pointer is not replace with tts.make_tptr so "
        "AtomicCASOp cannot be rewritten"));
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    LLVM_DEBUG(op->emitRemark(
        "PtrAnalysis: scalar AtomicCASOp will not be rewritten"));
    return failure();
  }

  if (!isSupportedStructuredPtr(ptr)) {
    return failure();
  }

  OpBuilder builder(op);

  auto atomicCASOp = builder.create<tts::AtomicCASOp>(
      loc, op.getType(), ptr, cmp, val, nullptr, op.getSemAttr(),
      op.getScopeAttr());

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::atomic_cas:\n";
    atomicCASOp->dump();
  });

  op.replaceAllUsesWith(atomicCASOp.getResult());
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp, bool useUnsafeMask) {
  LLVM_DEBUG({
    llvm::dbgs() << "rewriting rootOp\n";
    rootOp->dump();
  });

  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == rootOp) {
      return WalkResult::advance();
    }
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<triton::AddPtrOp>([&](auto addptr) {
          if (rewriteAddptrOp(addptr).failed()) {
            LLVM_DEBUG(
                addptr->emitRemark("PtrAnalysis: Failed to rewrite AddPtrOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::BitcastOp>([&](auto bitcast) {
          if (rewriteBitcastOp(bitcast).failed()) {
            LLVM_DEBUG(bitcast->emitRemark(
                "PtrAnalysis: Failed to rewrite BitcastOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::MakeTensorPtrOp>([&](auto maketptr) {
          if (rewriteMakeTensorPtrOp(maketptr).failed()) {
            LLVM_DEBUG(maketptr->emitRemark(
                "PtrAnalysis: Failed to rewrite MakeTensorPtrOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::AdvanceOp>([&](auto advance) {
          if (rewriteAdvanceOp(advance).failed()) {
            LLVM_DEBUG(advance->emitRemark(
                "PtrAnalysis: Failed to rewrite AdvanceOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::LoadOp>([&](auto load) {
          if (rewriteLoadOp(load, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::StoreOp>([&](auto store) {
          if (rewriteStoreOp(store, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::AtomicRMWOp>([&](auto atomicRMW) {
          if (rewriteAtomicRMWOp(atomicRMW, useUnsafeMask).failed()) {
            LLVM_DEBUG(atomicRMW->emitRemark(
                "PtrAnalysis: Failed to rewrite AtomicRMWOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::AtomicCASOp>([&](auto atomicCAS) {
          if (rewriteAtomicCASOp(atomicCAS).failed()) {
            LLVM_DEBUG(atomicCAS->emitRemark(
                "PtrAnalysis: Failed to rewrite AtomicCASOp"));
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<scf::ForOp>([&](auto forOp) {
          // `rewriteForOp` recursively visits its children, so regardless
          // whether the rewrite succeeds or not, we need to return "skip" so
          // that the the walk does not visit the for-op's child operations
          // the second time.
          if (rewriteForOp(forOp).failed()) {
            LLVM_DEBUG(
                forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp"));
          }
          return WalkResult::skip();
        })
        .Case<tts::GetStructuredStateOp>(
            [&](tts::GetStructuredStateOp getStateOp) {
              // For tensor of indices potentially being used in pointer
              // arithmetic sequence, we need to manually populate the state of
              // none already exists.
              // This process is necessary because unlike triton pointers in a
              // loop which always have a `tt.addptr` that triggers the rewrite
              // process which includes generating the ops for updating offsets
              // and strides, tensor of indices only have a simple `arith.addi`
              // (or other arith ops).
              // Without visiting these ops manually, the ops to update the
              // offsets and strides would not be generated.
              auto tritonValue = getStateOp->getOperand(0);
              if (!knownPtrs.contains(tritonValue)) {
                PtrState state;
                OpBuilder b(getStateOp);
                if (succeeded(visitOperand(tritonValue, state,
                                           getStateOp->getLoc(), b)) &&
                    state.isStructured()) {
                  knownPtrs[tritonValue] = state;
                } else {
                  LLVM_DEBUG(getStateOp->emitRemark(
                      "PtrAnalysis: Failed to populate ptr "
                      "state for tensor of indices"));
                }
              }

              return WalkResult::skip();
            })
        .Default([&](auto) { return WalkResult::advance(); });
  });

  return success();
}

} // namespace tts
} // namespace mlir
