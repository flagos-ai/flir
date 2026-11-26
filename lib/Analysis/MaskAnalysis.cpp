//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
#include "flagtree/Common/UnifiedHardware.h"

#include "triton-shared/Analysis/MaskAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#define DEBUG_TYPE "MaskAnalysis"
#include "llvm/Support/Debug.h"
namespace mlir {

namespace triton {

// Return the offset (interval lower bound) of dim i, defaulting to a static 0
// when the state does not carry offsets for that dim (prefix interval).
static OpFoldResult getDimOffset(const MaskState &state, int64_t i,
                                 OpBuilder &builder) {
  if (i < static_cast<int64_t>(state.offsets.size()) && state.offsets[i])
    return state.offsets[i];
  return builder.getIndexAttr(0);
}

// An empty interval must be represented as offset 0: a subview [lo, 0) with
// lo == tensor dim is rejected by the memref.subview verifier.
static OpFoldResult canonicalizeEmptyInterval(OpFoldResult dim,
                                              OpFoldResult lo,
                                              OpBuilder &builder) {
  auto dimAttr = getIntAttr(dim);
  if (dimAttr && dimAttr.value() == 0)
    return builder.getIndexAttr(0);
  return lo;
}
///////////ascend
void dimInfo::dump() const {
    LLVM_DEBUG({
       llvm::dbgs() << "MaskDimInfo: \n" ;
       llvm::dbgs() << "dim = " << dim << "\n";
       llvm::dbgs() << "shape = " << shape << "\n";
       llvm::dbgs() << "div = " << div << "\n";
       llvm::dbgs() << "isSlt = " << isSlt << "\n";
       llvm::dbgs() << "rhs = " << rhs << "\n";
       llvm::dbgs() << "isRealDim = " << isRealDim << "\n";
    });
};

static Value structuredMaskToUnstructuredMask(OpBuilder &builder, Location loc,
                                              uint32_t size, uint32_t maskDim) {

  if (maskDim == size) {
    // Full mask.
    return Value();
  }

  auto i32TensorType = RankedTensorType::get({size}, builder.getI32Type());
  Value range =
      triton::MakeRangeOp::create(builder, loc, i32TensorType, 0, size);
  Value v = arith::ConstantIndexOp::create(builder, loc, maskDim);
  v = arith::IndexCastUIOp::create(builder, loc, builder.getI32Type(), v);
  v = triton::SplatOp::create(builder, loc, i32TensorType, v).getResult();
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, range,
                                       v);
};

/////////////ascend
LogicalResult MaskState::parse(Value operand, const Location loc,
                               OpBuilder &builder) {
  auto hardwareManager = mlir::flagtree::createUnifiedHardwareManager();
  auto incubatedTag = hardwareManager -> getIncubatedTag();
  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return this->parseConstant(op, loc, builder);
  } else if (isa<IntegerType>(operand.getType())) {
    return this->parseIntScalar(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return this->parseAdd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return this->parseAnd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    return this->parseCmp(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, builder);
  } else if (!operand.getDefiningOp()) {
    return this->parseLoopIterArg(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return this->parseExtSI(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::SubIOp>()) {
    return this->parseSub(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    if (incubatedTag)
      return this->parseRemsi(op, loc, builder);
    else
      return failure();
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    if (incubatedTag)
      return this->parseDivsi(op, loc, builder);
    else
      return failure();
  } else {
    auto resultType = operand.getType();
    if (isa<RankedTensorType>(resultType) &&
        cast<RankedTensorType>(resultType).getRank() == 1) {
      assert(isEmpty());
      dims.push_back(operand);
      offsets.push_back(builder.getIndexAttr(0));
      nonContinuousDim.push_back(0);
      return success();
    }
    return failure();
  }
}

tensor::ExtractSliceOp MaskState::getExtractSlice(Value source,
                                                  const Location loc,
                                                  OpBuilder &builder) const {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> sliceOffsets;
  for (size_t i = 0; i < getRank(); ++i)
    sliceOffsets.push_back(getDimOffset(*this, i, builder));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  // LLVM 23 dropped the offsets/strides arguments of inferResultType.
#if LLVM_VERSION_MAJOR >= 23
  auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, dims);
#else
  auto dstType =
      tensor::ExtractSliceOp::inferResultType(sourceType, sliceOffsets, dims,
                                              strides);
#endif

  return tensor::ExtractSliceOp::create(builder, loc, dstType, source,
                                                sliceOffsets, dims, strides);
}

memref::SubViewOp MaskState::getSubview(Value source, const Location loc,
                                        OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> subOffsets;
  for (size_t i = 0; i < getRank(); ++i)
    subOffsets.push_back(getDimOffset(*this, i, builder));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, subOffsets, dims, strides);

  return memref::SubViewOp::create(builder, loc, cast<MemRefType>(dstType),
                                           source, subOffsets, dims, strides);
}

static memref::SubViewOp createSubview(Value src, Location loc, OpBuilder &b,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return memref::SubViewOp::create(b, loc, cast<MemRefType>(dstType), src,
                                     offsets, sizes, strides);
}

// Assume block1 wraps around and the remainder is block2.
//
// |----------------------|
// |         |            |
// | block2  |  block1    |
// |         |            |
// |----------------------|
//
// Once we copy the chunks in order, the end result is block1 followed by
// block2.
//
//   buffer_tmp:
//
// |----------------------|
// |             |        |
// | block1      | block2 |
// |             |        |
// |----------------------|
//
// Assume we have the following subview:
//
// +++++++++++++++++-------
// +               +      |
// + subview       +      |
// +               +      |
// +++++++++++++++++-------
//
// If we simply take the subview of `buffer_tmp`, this requires an extra
// buffer to just hold the temporary result.
//
// So we can subview into block1 and block2 directly. There are 2 cases:
//   + subview only spans block1
//   + subview spans both block1 and block2, creating sv1 and sv2 (illustrated
//     below for case when we wrap around side-by-side)
//
// |----------------------------------------|
// |                                        |
// |    col2                       col1     |
// |++++++--------|          |+++++++++++++++
// | sv2 + block2 |          | block1 & sv1 +
// |++++++--------|          |+++++++++++++++
// |                                        |
// |----------------------------------------|
//
// For simplicity, assume we only wrap around side-by-side.
//
// Let (row, col1) and (row, col2) be the dimensions of block1 and block2,
// respectively.
//
// Let (rowFull, colFull), (rowView1, colView1) and (rowView2, colView2) be
// the dimensions of the full subview, sv1, and sv2, respectively.
//
// + colView1 = min(colFull, col1)
// + colView2 = colFull - colView1
// + rowView1 = rowView2 = row = rowFull
std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getSideBySideSubviews(Value block1, Value block2, const Location loc,
                                 OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult col1 = memref::DimOp::create(builder, loc, block1, 1).getResult();
  OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, builder);
  OpFoldResult subviewCol2 = subOFRs(subviewColFull, subviewCol1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRowFull, subviewCol1}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRowFull, subviewCol2}, strides);

  return {sv1, sv2};
}

std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getStackedSubviews(Value block1, Value block2, const Location loc,
                              OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult row1 = memref::DimOp::create(builder, loc, block1, 0).getResult();
  OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, builder);
  OpFoldResult subviewRow2 = subOFRs(subviewRowFull, subviewRow1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRow1, subviewColFull}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRow2, subviewColFull}, strides);
  return {sv1, sv2};
}

LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  start = addOFRs(state.start, scalar, loc, builder);
  end = addOFRs(state.end, scalar, loc, builder);
  dims = state.dims;
  offsets = state.offsets;
  // Adding `scalar` to every lane value shifts a valid interval [lo, hi) in
  // value space to [lo - scalar, hi - scalar). Clamp the lower bound at 0.
  for (auto &off : offsets)
    off = maxOFRs(subOFRs(off, scalar, loc, builder),
                  builder.getIndexAttr(0), loc, builder);
  return success();
}

LogicalResult MaskState::subStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  start = subOFRs(state.start, scalar, loc, builder);
  end = subOFRs(state.end, scalar, loc, builder);
  dims = state.dims;
  offsets = state.offsets;
  // Subtracting `scalar` shifts the interval to [lo + scalar, hi + scalar).
  for (auto &off : offsets)
    off = addOFRs(off, scalar, loc, builder);
  return success();
}

LogicalResult MaskState::subStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitWarning(loc) << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar)
    return subStateScalar(rhsState, lhsState.scalar, loc, builder);
  else
    return subStateScalar(lhsState, rhsState.scalar, loc, builder);
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitWarning(loc) << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar)
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  else
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
}

LogicalResult MaskState::minStateScalar(const MaskState &lhsState,
                                        const MaskState &rhsState, Location loc,
                                        OpBuilder &builder) {
  auto intersect = [&](OpFoldResult lhsOff, OpFoldResult lhsDim,
                       OpFoldResult rhsOff, OpFoldResult rhsDim) {
    auto lo = maxOFRs(lhsOff, rhsOff, loc, builder);
    auto hi = minOFRs(addOFRs(lhsOff, lhsDim, loc, builder),
                      addOFRs(rhsOff, rhsDim, loc, builder), loc, builder);
    // Clamp lo to hi so the subview stays valid for an empty interval; skip
    // when lo is statically 0 (0 <= hi always holds, keeps lo static).
    auto loAttr = getIntAttr(lo);
    if (!loAttr || loAttr.value() != 0)
      lo = minOFRs(lo, hi, loc, builder);
    auto dim = subOFRs(hi, lo, loc, builder);
    lo = canonicalizeEmptyInterval(dim, lo, builder);
    offsets.push_back(lo);
    dims.push_back(dim);
  };

  if (lhsState.scalar && rhsState.scalar) {
    auto zero = builder.getIndexAttr(0);
    intersect(zero, lhsState.dims[0], zero, rhsState.dims[0]);
  } else if (lhsState.scalar) {
    for (uint32_t i = 0; i < rhsState.getRank(); i++) {
      auto lhsDim = lhsState.dims[0];
      auto rhsDim = rhsState.dims[i];
      auto rhsOff = getDimOffset(rhsState, i, builder);
      intersect(builder.getIndexAttr(0), lhsDim, rhsOff, rhsDim);
    }
  } else if (rhsState.scalar) {
    for (uint32_t i = 0; i < lhsState.getRank(); i++) {
      auto lhsDim = lhsState.dims[i];
      auto rhsDim = rhsState.dims[0];
      auto lhsOff = getDimOffset(lhsState, i, builder);
      intersect(lhsOff, lhsDim, builder.getIndexAttr(0), rhsDim);
    }
  } else {
    InFlightDiagnostic diag =
        emitWarning(loc) << "Unexpected case where both lhs and rhs are not scalars";
    return failure();
  }
  return success();
}

LogicalResult MaskState::minStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.getRank() != rhsState.getRank()) {
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "Unexpected case where lhs and rhs have different ranks";
    return failure();
  }

  // Interval intersection per dim: lo = max(lo1, lo2),
  // hi = min(lo1+d1, lo2+d2), dim = clamp(hi - lo, 0).
  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    auto lhsOff = getDimOffset(lhsState, i, builder);
    auto rhsOff = getDimOffset(rhsState, i, builder);
    auto lo = maxOFRs(lhsOff, rhsOff, loc, builder);
    auto hi = minOFRs(addOFRs(lhsOff, lhsDim, loc, builder),
                      addOFRs(rhsOff, rhsDim, loc, builder), loc, builder);
    // Clamp lo to hi so the subview stays valid for an empty interval; skip
    // when lo is statically 0 (0 <= hi always holds, keeps lo static).
    auto loAttr = getIntAttr(lo);
    if (!loAttr || loAttr.value() != 0)
      lo = minOFRs(lo, hi, loc, builder);
    auto dim = subOFRs(hi, lo, loc, builder);
    lo = canonicalizeEmptyInterval(dim, lo, builder);
    offsets.push_back(lo);
    dims.push_back(dim);
  }
  return success();
}

LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location loc, OpBuilder &builder) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && isa<IntegerType>(elementType) &&
           "All elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = builder.getIndexAttr(value.getSExtValue());
    auto op = arith::ConstantOp::materialize(builder, constAttr,
                                             builder.getIndexType(), loc);
    this->scalar = op.getValue();
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    this->scalar = builder.getIndexAttr(value);
  }

  return success();
}

LogicalResult MaskState::parseIntScalar(Value scalar, const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto castOp =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(), scalar);
  this->scalar = castOp.getResult();
  return success();
}

void MaskState::dump() const {
  llvm::dbgs() << "start: " << start << "\n";
  llvm::dbgs() << "end: " << end << "\n";
  llvm::dbgs() << "scalar: " << scalar << "\n";
  llvm::dbgs() << "useUnsafeMask: " << useUnsafeMask << "\n";
  llvm::dbgs() << "dims: ";
  for (auto dim : dims)
    llvm::dbgs() << "\t" << dim << "\n";
  llvm::dbgs() << "offsets: ";
  for (auto off : offsets)
    llvm::dbgs() << "\t" << off << "\n";
  llvm::dbgs() << "\n";
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder)))
    return failure();

  // WORKAROUND: avoid other non-structured cases. Since mask analysis is not
  // consistent with noncontinuous dims now
  if (lhsState.hasNonContinuousDim() || rhsState.hasNonContinuousDim()) {
    return failure();
  }

  return this->addStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseSub(arith::SubIOp subOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(subOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(subOp.getRhs(), loc, builder)))
    return failure();

  // WORKAROUND: avoid other non-structured cases. Since mask analysis is not
  // consistent with noncontinuous dims now
  if (lhsState.hasNonContinuousDim() || rhsState.hasNonContinuousDim()) {
    return failure();
  }

  return this->subStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)))
    return failure();

  if (lhsState.hasNonContinuousDim() || rhsState.hasNonContinuousDim()) {

    auto lhsNonContinuousDim = lhsState.getNonContinuousDim();
    auto rhsNonContinuousDim = rhsState.getNonContinuousDim();

    // assert(
    //     !(lhsNonContinuousDim.has_value() && rhsNonContinuousDim.has_value()) &&
    //     "Cannot have both non continuous dims");

    // If both sides have non-continuous dims at the same dimension:
    // directly AND the two boolean tensors element-wise.
    if (lhsNonContinuousDim.has_value() && rhsNonContinuousDim.has_value()) {
      if (lhsNonContinuousDim.value() != rhsNonContinuousDim.value()) {
        InFlightDiagnostic diag = andOp->emitWarning(
          "Cannot have non-continuous dims at different positions");
        return failure();
      }

      auto nonContinuousDim = lhsNonContinuousDim.value();
      auto tensorType = cast<RankedTensorType>(andOp.getType());
      assert(tensorType.getElementTypeBitWidth() == 1 &&
             "Unexpected case where type is not i1");

      for (uint32_t i = 0; i < lhsState.getRank(); i++) {
        if (i == nonContinuousDim) {
          auto lhsVal = dyn_cast<Value>(lhsState.dims[nonContinuousDim]);
          auto rhsVal = dyn_cast<Value>(rhsState.dims[nonContinuousDim]);
          assert(lhsVal && rhsVal &&
                 "Expected boolean tensor Values for non-continuous dims");
          dims.push_back(
              arith::AndIOp::create(builder, loc, lhsVal, rhsVal).getResult());
          offsets.push_back(builder.getIndexAttr(0));
          continue;
        }
        auto lhsDim = lhsState.dims[i];
        auto rhsDim = rhsState.dims[i];
        auto lhsOff = getDimOffset(lhsState, i, builder);
        auto rhsOff = getDimOffset(rhsState, i, builder);
        auto lo = maxOFRs(lhsOff, rhsOff, loc, builder);
        auto hi = minOFRs(addOFRs(lhsOff, lhsDim, loc, builder),
                          addOFRs(rhsOff, rhsDim, loc, builder), loc, builder);
        auto loAttr = getIntAttr(lo);
        if (!loAttr || loAttr.value() != 0)
          lo = minOFRs(lo, hi, loc, builder);
        auto dim = subOFRs(hi, lo, loc, builder);
        lo = canonicalizeEmptyInterval(dim, lo, builder);
        offsets.push_back(lo);
        dims.push_back(dim);
      }
      this->nonContinuousDim = lhsState.nonContinuousDim;
      return success();
    }

    auto [lhs, rhs] =
        lhsNonContinuousDim.has_value()
            ? std::tuple<MaskState *, MaskState *>(&lhsState, &rhsState)
            : std::tuple<MaskState *, MaskState *>(&rhsState, &lhsState);

    auto nonContinuousDim = lhs->getNonContinuousDim().value();
    auto rhsDim = rhs->dims[nonContinuousDim];
    auto rhsDimInt = getIntAttr(rhsDim);

    auto tensorType = cast<RankedTensorType>(andOp.getType());
    assert(tensorType.getElementTypeBitWidth() == 1 &&
           "Unexpected case where type is not i1");

    for (uint32_t i = 0; i < lhsState.getRank(); i++) {
      if (i == nonContinuousDim) {
        Value rhsDimValue;
        if (rhsDimInt.has_value()) {
          rhsDimValue = structuredMaskToUnstructuredMask(
              builder, loc, tensorType.getShape()[nonContinuousDim],
              rhsDimInt.value());
        } else {
          // dynamic rhsDim: create range < splat(rhsDim) directly
          uint32_t dimSize = tensorType.getShape()[nonContinuousDim];
          auto i32TensorType =
              RankedTensorType::get({dimSize}, builder.getI32Type());
          Value range = triton::MakeRangeOp::create(builder, loc, i32TensorType, 0,
                                                            dimSize);
          Value rhsVal = ofrToIndexValue(rhsDim, loc, builder);
          Value rhsValI32 = arith::IndexCastOp::create(builder,
              loc, builder.getI32Type(), rhsVal);
          Value rhsValSplat =
              triton::SplatOp::create(builder, loc, i32TensorType, rhsValI32);
          rhsDimValue = arith::CmpIOp::create(builder,
              loc, arith::CmpIPredicate::ult, range, rhsValSplat);
        }

        dims.push_back(
            rhsDimValue
                ? arith::AndIOp::create(builder,
                          loc, dyn_cast<Value>(lhs->dims[nonContinuousDim]),
                          rhsDimValue)
                      .getResult()
                : lhs->dims[nonContinuousDim]);
        offsets.push_back(builder.getIndexAttr(0));
        continue;
      }

      auto lhsDim = lhsState.dims[i];
      auto rhsDim = rhsState.dims[i];
      auto lhsOff = getDimOffset(lhsState, i, builder);
      auto rhsOff = getDimOffset(rhsState, i, builder);
      auto lo = maxOFRs(lhsOff, rhsOff, loc, builder);
      auto hi = minOFRs(addOFRs(lhsOff, lhsDim, loc, builder),
                        addOFRs(rhsOff, rhsDim, loc, builder), loc, builder);
      auto loAttr = getIntAttr(lo);
      if (!loAttr || loAttr.value() != 0)
        lo = minOFRs(lo, hi, loc, builder);
      auto dim = subOFRs(hi, lo, loc, builder);
      lo = canonicalizeEmptyInterval(dim, lo, builder);
      offsets.push_back(lo);
      dims.push_back(dim);
    }
    this->nonContinuousDim = lhs->nonContinuousDim;
    return success();
  }
  // TODO(FLIR): should be isMask()
  if(!lhsState.isMaskWithoutScalar() || !rhsState.isMaskWithoutScalar()) {
    return this->minStateScalar(lhsState, rhsState, loc, builder);
  }
  return this->minStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseExtSI(arith::ExtSIOp op, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());
  return parse(op.getIn(), loc, builder);
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
      cmpOp.getPredicate() != arith::CmpIPredicate::ult &&
      cmpOp.getPredicate() != arith::CmpIPredicate::sge) {
    InFlightDiagnostic diag = cmpOp->emitWarning("Unsupported cmpi");
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(cmpOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder)))
    return failure();

  // sge requires a scalar rhs: lane i is valid when start + i >= rhs.
  // The rhs need not be zero; the interval lower bound is computed at runtime
  // for dynamic rhs.
  if (cmpOp.getPredicate() == arith::CmpIPredicate::sge &&
      !rhsState.scalar) {
    InFlightDiagnostic diag =
        cmpOp->emitWarning("Unsupported sge cmpi with non-scalar rhs");
    return failure();
  }

  // This cmpDim is used to determine which dimension contains the actual
  // data range (i.e., the dimension with a size greater than 1)
  // Because mask analysis assumes that only one dimension can have an actual
  // range
  int32_t cmpDim = lhsState.scalar && rhsState.scalar ? 0 : -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      if (cmpDim != -1) {
        InFlightDiagnostic diag = cmpOp->emitWarning(
          "Unsupported cmpi with more than one "
          "dimension with size larger than 1");
        return failure();
      }
      cmpDim = i;
    }
  }
  assert(cmpDim != -1 &&
         "Unexpected case where no dimension has size larger than 1");

  OpFoldResult newDim;
  OpFoldResult newOffset = builder.getIndexAttr(0);
  if (lhsState.hasNonContinuousDim()) {
    assert(rhsState.scalar && "Unexpected case where rhs is not a scalar");
    assert(cmpDim == lhsState.getNonContinuousDim());
    auto lhsValue = dyn_cast<Value>(lhsState.dims[cmpDim]);
    auto tensorType = cast<RankedTensorType>(lhsValue.getType());

    auto rhsValue = arith::IndexCastOp::create(builder,
        loc, tensorType.getElementType(),
        ofrToIndexValue(rhsState.scalar, loc, builder));
    auto rhsTensorValue =
        triton::SplatOp::create(builder, loc, tensorType, rhsValue);

    newDim = arith::CmpIOp::create(builder, loc, cmpOp.getPredicate(), lhsValue,
                                        rhsTensorValue)
                 ->getResult(0);

    nonContinuousDim = lhsState.nonContinuousDim;
  } else if (lhsState.scalar) {
    assert(rhsState.scalar && "Unexpected case where rhs is not a scalar");
    // If both lhs and rhs are scalars, we can't just derive the dimension of
    // the mask as the minimum value: lhs/rhs could be 0 and then we don't
    // load/store anything.
    //
    // Instead treat the comparison as a scalar that determines if anything
    // should be loaded/stored by inserting a comparison + select:
    //    dim = lhs < rhs ? lhs.dim : 0
    newDim = compareOFRs(lhsState.scalar, rhsState.scalar, cmpOp.getPredicate(),
                  lhsState.dims[cmpDim], builder.getIndexAttr(0),
                  loc, builder);
  } else if (cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
    cmpOp.getPredicate() == arith::CmpIPredicate::ult) {
    // Important:
    // In the case where the values we are loading are entirely masked off like
    // the following:
    //
    // ---|-------|-----------|
    //    ^       ^           ^
    //   scalar  start       end
    //
    // newEnd = min(end, scalar) = scalar
    // Now scalar < start, so simply doing dim = newEnd - start is incorrect.
    //
    // The correct formula is to optionally move `newDim` back to `start` using
    // max(newEnd, start).
    auto newEnd = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
    // Avoid minus dim. ([5, 10] < 3: 3 - 5 = -2��
    newEnd = maxOFRs(newEnd, lhsState.start, loc, builder);
    newDim = subOFRs(newEnd, lhsState.start, loc, builder);
  } else {
    assert(cmpOp.getPredicate() == arith::CmpIPredicate::sge && rhsState.scalar);
    if (!lhsState.start || !lhsState.end) {
      // sge against a mask state (e.g. a comparison result) has no value
      // range to derive the lower bound from.
      cmpOp->emitWarning("Unsupported sge cmpi without a value range");
      return failure();
    }
    // Lane i is valid when start + i >= rhs, i.e. the interval
    // [clamp(rhs - start, 0), dims). Clamp the offset to the tensor size so
    // the empty case (lo == dims) still yields a valid subview.
    auto fullDim = lhsState.dims[cmpDim];
    newOffset = subOFRs(rhsState.scalar, lhsState.start, loc, builder);
    newOffset = maxOFRs(newOffset, builder.getIndexAttr(0), loc, builder);
    newOffset = minOFRs(newOffset, fullDim, loc, builder);
    newDim = subOFRs(fullDim, newOffset, loc, builder);
    newOffset = canonicalizeEmptyInterval(newDim, newOffset, builder);
  }

  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    if (i == cmpDim) {
      this->dims.push_back(newDim);
      this->offsets.push_back(newOffset);
    } else {
      this->dims.push_back(lhsState.dims[i]);
      this->offsets.push_back(getDimOffset(lhsState, i, builder));
    }
  }

  return success();
}

LogicalResult MaskState::parseLoopIterArg(Value v, const Location loc,
                                          OpBuilder &builder) {
  assert(!v.getDefiningOp());

  auto forOp = llvm::dyn_cast<scf::ForOp>(v.getParentRegion()->getParentOp());

  if (!forOp) {
    return failure();
  }

  // TODO: This implementation does not work with nested loops
  if (forOp->getParentOfType<scf::ForOp>()) {
    return failure();
  }

  auto it = llvm::find(forOp.getRegionIterArgs(), v);
  if (it == forOp.getRegionIterArgs().end()) {
    return failure();
  }

  auto argIndex = std::distance(forOp.getRegionIterArgs().begin(), it);
  auto initArg = forOp.getInitArgs()[argIndex];
  if (auto getStateOp = initArg.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto tritonValue = getStateOp->getOperand(0);
    MaskState lhsState;
    if (failed(lhsState.parse(tritonValue, loc, builder))) {
      return failure();
    }

#ifdef FLAGTREE_BACKEND_TSINGMICRO
    if (llvm::isa_and_nonnull<arith::ConstantOp>(tritonValue.getDefiningOp())) {
      // It's accurately a size 1 tl.arange op
      auto constOp = tritonValue.getDefiningOp<arith::ConstantOp>();
      if (llvm::succeeded(this->parseConstant(constOp, loc, builder))) {
        this->dims.push_back(builder.getIndexAttr(1));
        this->offsets.push_back(builder.getIndexAttr(0));
        return success();
      }
      return failure();
    }
#endif
    // This is a bit of a hack!!
    //
    // The offsets and dimensions of a MaskState can now depend on a loop's
    // iter-arg.
    //
    // Because the PtrAnalysis's pre-pass already sets up the offsets,
    // we can create a new MaskState for each loop iteration by adding the
    // original MaskState with the current iter-arg, which is at `argIndex +
    // 1`.
    //
    // This will not work for nested loop scenarios, which would need a
    // more robust implementation.
    if (failed(this->addStateScalar(
            lhsState, forOp.getRegionIterArgs()[argIndex + 1], loc, builder))) {
      return failure();
    }

    return success();
  }

  return failure();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    InFlightDiagnostic diag =
        rangeOp->emitWarning(
          "stride must be 1 for make_range whose result is used "
          "as load or store masks");
    return failure();
  }

  this->start = builder.getIndexAttr(start);
  this->end = builder.getIndexAttr(end);
  this->dims.push_back(builder.getIndexAttr(shape[0]));
  // Mask intervals are relative to the lane index (lane 0 is the interval
  // start), so a plain range contributes a prefix interval.
  this->offsets.push_back(builder.getIndexAttr(0));

  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, builder)))
    return failure();

  // A broadcast dim covers its whole extent: the interval on that dim starts
  // at 0. Align the offsets vector with dims (missing entries default to 0).
  this->offsets.resize(this->dims.size(), builder.getIndexAttr(0));
  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i]) {
      this->dims[i] = builder.getIndexAttr(dstShape[i]);
      this->offsets[i] = builder.getIndexAttr(0);
    } else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }

  return success();
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (!isa<IntegerType>(src.getType())) {
    InFlightDiagnostic diag =
        splatOp->emitWarning(
          "splat source must be an integer scalar for load/store masks");
    return failure();
  }

  // A boolean splat is a lane predicate, which the interval model cannot
  // express: `minStates` only intersects dims and ignores `scalar`, so the
  // predicate would be silently dropped and the whole tile accessed anyway.
  // Reject it so PtrAnalysis falls back to gather/scatter. Only a literal
  // `true` splat really means "all lanes valid".
  if (src.getType().isInteger(1)) {
    bool isConstTrue = false;
    if (auto constOp = src.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        isConstTrue = !intAttr.getValue().isZero();
    if (!isConstTrue) {
      InFlightDiagnostic diag =
          splatOp->emitWarning("scalar boolean mask is not supported by the "
                               "structured interval model; using gather/scatter");
      return failure();
    }
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  for (auto s : dstShape) {
    this->dims.push_back(builder.getIndexAttr(s));
    this->offsets.push_back(builder.getIndexAttr(0));
  }

  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  if (failed(this->parse(expandDimsOp.getSrc(), loc, builder)))
    return failure();

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");
  this->dims.insert(this->dims.begin() + axis, builder.getIndexAttr(1));
  // Offsets stay parallel to dims: the expanded axis is a prefix interval.
  this->offsets.insert(this->offsets.begin() + axis, builder.getIndexAttr(0));

  if (hasNonContinuousDim()) {
    assert(dstShape.size() == 2);
    this->nonContinuousDim[0] = 1 - axis;
  }
  return success();
}
////////ASCEND
LogicalResult MaskState::parseRemsi(arith::RemSIOp remsiOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());
  auto defaultAttr = builder.getIndexAttr(0);

  MaskState lhsState;
  if (failed(lhsState.parse(remsiOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(remsiOp.getRhs(), loc, builder)))
    return failure();

  if(lhsState.scalar || !rhsState.scalar){
    remsiOp->emitRemark("Unsupported remsi scenario");
    return failure();
  }

  int64_t staticShape;

  if(auto value = rhsState.scalar.dyn_cast<Value>()){
    auto constop = value.getDefiningOp<arith::ConstantOp>();
    staticShape = cast<IntegerAttr>(constop.getValue()).getInt();
  }else if(auto rhsIntAttr = getIntAttr(rhsState.scalar)){
    assert(rhsIntAttr.has_value());
    staticShape = rhsIntAttr.value();
  }else{
    remsiOp->emitWarning("MaskAnalysis: Static compilation cannot determine the value of this parameter");
    return failure();
  }

  start = lhsState.start;
  dims = lhsState.dims;
  offsets = lhsState.offsets;
  end = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
  stateInfo = lhsState.stateInfo;
  for(auto &info: stateInfo){
    if(info.isRealDim){
      auto staticDim = getIntAttr(rhsState.scalar);
      if(!staticDim.has_value() || (staticDim.value() % staticShape != 0 && staticShape % staticDim.value() != 0)){
        remsiOp->emitWarning("MaskAnalysis: The shape of the mask is not divisible by the shape of the block");
        return failure();
      }
      // if(getIntAttr(info.div).has_value() && getIntAttr(info.div).value() != 0){
      //   remsiOp->emitWarning("MaskAnalysis: do not support remsi after div");
      //   return failure();
      // }
      info.shape = builder.getIndexAttr(staticShape);
    }
  }

  return success();
}

LogicalResult MaskState::parseDivsi(arith::DivSIOp divsiOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  auto defaultAttr = builder.getIndexAttr(0);

  MaskState lhsState;
  if (failed(lhsState.parse(divsiOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(divsiOp.getRhs(), loc, builder)))
    return failure();

  if(lhsState.scalar || !rhsState.scalar){
    divsiOp->emitRemark("Unsupported divsi scenario");
    return failure();
  }

  int64_t staticDiv;

  if(auto value = rhsState.scalar.dyn_cast<Value>()){
    auto constop = value.getDefiningOp<arith::ConstantOp>();
    staticDiv = cast<IntegerAttr>(constop.getValue()).getInt();
  }else if(auto rhsIntAttr = getIntAttr(rhsState.scalar)){
    assert(rhsIntAttr.has_value());
    staticDiv = rhsIntAttr.value();
  }else{
    divsiOp->emitWarning("MaskAnalysis: Static compilation cannot determine the value of this parameter");
    return failure();
  }

  start = divOFRs(lhsState.start, rhsState.scalar, loc, builder);
  dims = lhsState.dims;
  offsets = lhsState.offsets;
  auto minEnd = addOFRs(start, builder.getIndexAttr(1), loc, builder);
  end = subOFRs(lhsState.end, builder.getIndexAttr(1), loc, builder);
  end = divOFRs(end, rhsState.scalar, loc, builder);
  end = addOFRs(end, builder.getIndexAttr(1), loc, builder);
  end = maxOFRs(end, minEnd, loc, builder);
  stateInfo = lhsState.stateInfo;
  for(auto &info: stateInfo){
    if(info.isRealDim){
      auto staticDim = getIntAttr(rhsState.scalar);
      if(!staticDim.has_value() || (staticDim.value() % staticDiv != 0 && staticDiv % staticDim.value() != 0)){
        divsiOp->emitWarning("MaskAnalysis: The shape of the mask is not divisible by the shape of the block");
        return failure();
      }
      if(getIntAttr(info.shape).has_value() && getIntAttr(info.shape).value() != 0){
        divsiOp->emitWarning("MaskAnalysis: do not support div after remsi");
        return failure();
      }
      info.div = builder.getIndexAttr(staticDiv);
    }
  }

  return success();
}

void MaskState::eraseInsertedOps(Operation *rawOp, PatternRewriter &rewriter) {
  auto moduleOp = rawOp->getParentOfType<ModuleOp>();
  SmallVector<Operation *> worklist;
  moduleOp->walk([&](Operation *op) {
    if (isOpTriviallyDead(op))
      worklist.push_back(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    for (Value value : op->getOperands()) {
      if (auto defOp = value.getDefiningOp())
        worklist.push_back(defOp);
    }
    LLVM_DEBUG({
      llvm::dbgs() << "[MaskState]==> inserted op: \n"
                   << *op << "\n[MaskState]<== is removed\n";
    });
    rewriter.eraseOp(op);
  }
}

tensor::InsertSliceOp MaskState::getInsertSlice(Value source, Value dest,
                                                const Location &loc,
                                                OpBuilder &builder) const {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> insertOffsets;
  for (size_t i = 0; i < getRank(); ++i)
    insertOffsets.push_back(getDimOffset(*this, i, builder));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  return tensor::InsertSliceOp::create(builder, loc, source, dest, insertOffsets,
                                               dims, strides);
}

} // namespace triton
} // namespace mlir
