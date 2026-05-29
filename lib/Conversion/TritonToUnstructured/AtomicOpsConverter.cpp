//===- AtomicOpsConverter.cpp ---------------------------------------------===//
//
// See AtomicOpsConverter.h for design notes and pipeline placement.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToUnstructured/AtomicOpsConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "atomic-ops-converter"

using namespace mlir;
using namespace mlir::triton::shared;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

namespace {

/// Return true iff `v` is an arith.constant with a DenseElementsAttr that is a
/// all-`expected` boolean splat.
static bool isBoolSplat(Value v, bool expected) {
  if (!v)
    return false;
  auto *def = v.getDefiningOp();
  if (!def)
    return false;
  auto constOp = dyn_cast<arith::ConstantOp>(def);
  if (!constOp)
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!dense || !dense.isSplat())
    return false;
  // Accept both i1 tensors and 0-d i1.
  auto iTy = dyn_cast<IntegerType>(dense.getType().getElementType());
  if (!iTy || iTy.getWidth() != 1)
    return false;
  return dense.getSplatValue<bool>() == expected;
}

/// Extract the pointee type from a tt.ptr or tensor<N x tt.ptr<T>>.
static Type getPointeeType(Type ptrOrTensorOfPtr) {
  if (auto ptrTy = dyn_cast<triton::PointerType>(ptrOrTensorOfPtr))
    return ptrTy.getPointeeType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(ptrOrTensorOfPtr))
    if (auto ptrElem = dyn_cast<triton::PointerType>(tensorTy.getElementType()))
      return ptrElem.getPointeeType();
  return {};
}

/// Ensure `val` is a memref with `shape` and `elemTy`.
/// If it is still a ranked tensor, emit a bufferization.to_memref.
static Value ensureMemRef(OpBuilder &b, Location loc, Value val,
                          ArrayRef<int64_t> shape, Type elemTy) {
  if (isa<MemRefType>(val.getType()))
    return val;
  auto mrt = MemRefType::get(shape, elemTy);
#if LLVM_VERSION_MAJOR < 22
  return b.create<bufferization::ToMemrefOp>(loc, mrt, val);
#else
  return b.create<bufferization::ToBufferOp>(loc, mrt, val);
#endif
}

} // namespace

//===----------------------------------------------------------------------===//
// AtomicRMWConverter – helpers
//===----------------------------------------------------------------------===//

bool AtomicRMWConverter::isSplatTrue(Value mask) {
  return isBoolSplat(mask, true);
}
bool AtomicRMWConverter::isSplatFalse(Value mask) {
  return isBoolSplat(mask, false);
}

Value AtomicRMWConverter::buildBinaryOp(OpBuilder &b, Location loc,
                                        triton::RMWOp kind, Type elemTy,
                                        Value lhs, Value rhs) const {
  switch (kind) {
  case triton::RMWOp::FADD:
    return b.create<arith::AddFOp>(loc, lhs, rhs);
  case triton::RMWOp::ADD:
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  case triton::RMWOp::XOR:
    return b.create<arith::XOrIOp>(loc, lhs, rhs);
  case triton::RMWOp::OR:
    return b.create<arith::OrIOp>(loc, lhs, rhs);
  case triton::RMWOp::AND:
    return b.create<arith::AndIOp>(loc, lhs, rhs);
  case triton::RMWOp::MAX:
    return isa<FloatType>(elemTy)
               ? b.create<arith::MaxNumFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MaxSIOp>(loc, lhs, rhs).getResult();
  case triton::RMWOp::MIN:
    return isa<FloatType>(elemTy)
               ? b.create<arith::MinNumFOp>(loc, lhs, rhs).getResult()
               : b.create<arith::MinSIOp>(loc, lhs, rhs).getResult();
  case triton::RMWOp::UMAX:
    return b.create<arith::MaxUIOp>(loc, lhs, rhs);
  case triton::RMWOp::UMIN:
    return b.create<arith::MinUIOp>(loc, lhs, rhs);
  case triton::RMWOp::XCHG:
    return rhs; // exchange: new value is simply rhs
  default:
    break;
  }
  llvm_unreachable("unhandled RMWOp in buildBinaryOp");
}

//===----------------------------------------------------------------------===//
// AtomicRMWConverter::matchAndRewrite
//===----------------------------------------------------------------------===//

LogicalResult
AtomicRMWConverter::matchAndRewrite(triton::AtomicRMWOp op,
                                    PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  // ── 直接从原始 op 拿 ptr，自己做类型转换 ─────────────────────────────────
  // 不使用 adaptor.getPtr()，因为框架的 TypeConverter 可能还没把函数参数
  // 类型转换好（tt.func 签名未变）。我们手动把 !tt.ptr<T> → memref<?xT>。
  Value rawPtr = op.getPtr();
  Value val = op.getVal();
  Value mask = op.getMask();
  auto rmwKind = op.getAtomicRmwOp();
  Type resultTy = op.getResult().getType();
  Value ptr;
  Type elemTy;

  // ── Fast-path: mask 全 false → op 是 no-op ───────────────────────────────
  if (mask && isSplatFalse(mask)) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, elemTy, rewriter.getZeroAttr(elemTy));
    rewriter.replaceOp(op, zero);
    return success();
  }

  bool isTensor = isa<RankedTensorType>(resultTy);

  // =========================================================================
  // (A) Scalar path
  // =========================================================================
  if (!isTensor) {
    if (auto ptrTy = dyn_cast<triton::PointerType>(rawPtr.getType())) {
      elemTy = ptrTy.getPointeeType();
      auto memrefTy = MemRefType::get({ShapedType::kDynamic}, elemTy);
      ptr = rewriter
                .create<UnrealizedConversionCastOp>(loc, TypeRange{memrefTy},
                                                    ValueRange{rawPtr})
                .getResult(0);
    } else if (auto memrefTy = dyn_cast<MemRefType>(rawPtr.getType())) {
      ptr = rawPtr;
      elemTy = memrefTy.getElementType();
    } else {
      return rewriter.notifyMatchFailure(op, "scalar: unexpected ptr type");
    }

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value old = rewriter.create<memref::LoadOp>(loc, ptr, ValueRange{c0});

    bool unconditional = (!mask || isSplatTrue(mask));

    auto emitStore = [&](OpBuilder &b, Location l) {
      Value newVal = buildBinaryOp(b, l, rmwKind, elemTy, old, val);
      b.create<memref::StoreOp>(l, newVal, ptr, ValueRange{c0});
    };

    if (unconditional) {
      emitStore(rewriter, loc);
    } else {
      Value maskScalar = mask;
      // mask 如果还是 tt.ptr 转来的 memref 的话，load 出来
      if (isa<MemRefType>(mask.getType())) {
        Value mc0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        maskScalar =
            rewriter.create<memref::LoadOp>(loc, mask, ValueRange{mc0});
      }
      rewriter.create<scf::IfOp>(loc, maskScalar,
                                 [&](OpBuilder &b, Location l) {
                                   emitStore(b, l);
                                   b.create<scf::YieldOp>(l);
                                 });
    }

    rewriter.replaceOp(op, old);
    return success();
  }

  // =========================================================================
  // (B) Tensor path
  // =========================================================================
  auto tensorResultTy = cast<RankedTensorType>(resultTy);
  ArrayRef<int64_t> shape = tensorResultTy.getShape();
  unsigned rank = shape.size();

  // tensor ptr: tensor<Nx!tt.ptr<T>> → 每个元素都是 ptr，需要逐元素处理
  // 在 linalg.generic 里处理，ptr memref 通过 cast 获得
  Value ptrMemRef;
  if (auto tensorPtrTy = dyn_cast<RankedTensorType>(rawPtr.getType())) {
    auto ptrElemTy =
        dyn_cast<triton::PointerType>(tensorPtrTy.getElementType());
    if (!ptrElemTy)
      return rewriter.notifyMatchFailure(op,
                                         "tensor ptr element is not tt.ptr");
    elemTy = ptrElemTy.getPointeeType();
    auto memrefTy = MemRefType::get(shape, elemTy);
    ptrMemRef = rewriter
                    .create<UnrealizedConversionCastOp>(
                        loc, TypeRange{memrefTy}, ValueRange{rawPtr})
                    .getResult(0);
  } else if (auto memrefTy = dyn_cast<MemRefType>(rawPtr.getType())) {
    ptrMemRef = rawPtr;
    elemTy = memrefTy.getElementType();
  } else {
    return rewriter.notifyMatchFailure(op, "unexpected tensor ptr type");
  }

  // val memref
  Value valMR;
  if (isa<RankedTensorType>(val.getType())) {
    auto valMemRefTy = MemRefType::get(shape, elemTy);
    valMR = rewriter
                .create<UnrealizedConversionCastOp>(loc, TypeRange{valMemRefTy},
                                                    ValueRange{val})
                .getResult(0);
  } else {
    valMR = val;
  }

  // result buffer（存 old values）
  Value resultBuf =
      rewriter.create<memref::AllocOp>(loc, MemRefType::get(shape, elemTy));

  auto idMap = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);

  bool needMask = mask && !isSplatTrue(mask);
  Value maskMR;
  if (needMask) {
    auto maskMemRefTy = MemRefType::get(shape, rewriter.getI1Type());
    if (isa<RankedTensorType>(mask.getType())) {
      maskMR = rewriter
                   .create<UnrealizedConversionCastOp>(
                       loc, TypeRange{maskMemRefTy}, ValueRange{mask})
                   .getResult(0);
    } else {
      maskMR = mask;
    }
  }

  SmallVector<Value> inputs = {ptrMemRef, valMR};
  SmallVector<Value> outputs = {ptrMemRef, resultBuf};
  SmallVector<AffineMap> maps = {idMap, idMap, idMap, idMap};
  if (needMask) {
    inputs.push_back(maskMR);
    maps.push_back(idMap);
  }

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{}, inputs, outputs, maps, iters,
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value ptrElem = args[0];
        Value valElem = args[1];
        Value computed = buildBinaryOp(b, l, rmwKind, elemTy, ptrElem, valElem);
        Value writeBack = computed;
        if (needMask) {
          Value maskElem = args[2];
          writeBack = b.create<arith::SelectOp>(l, maskElem, computed, ptrElem);
        }
        b.create<linalg::YieldOp>(l, ValueRange{writeBack, ptrElem});
      });

  MLIRContext *context = rewriter.getContext();
  const StringRef genericAtomicRMW = "GenericAtomicRMW";
  const StringRef memSemantic      = "MemSemantic";
  const StringRef memSyncScope     = "MemSyncScope";

  auto rmwKindStr = [](triton::RMWOp kind) -> StringRef {
    switch (kind) {
      case triton::RMWOp::FADD: return "fadd";
      case triton::RMWOp::ADD:  return "add";
      case triton::RMWOp::XCHG: return "xchg";
      case triton::RMWOp::AND:  return "and";
      case triton::RMWOp::OR:   return "or";
      case triton::RMWOp::XOR:  return "xor";
      case triton::RMWOp::MAX:  return "max";
      case triton::RMWOp::MIN:  return "min";
      case triton::RMWOp::UMAX: return "umax";
      case triton::RMWOp::UMIN: return "umin";
      default:                  return "unknown";
    }
  };

  genericOp->setAttr(genericAtomicRMW,
      mlir::StringAttr::get(context, rmwKindStr(rmwKind)));
  genericOp->setAttr(memSemantic,
      rewriter.getStringAttr(stringifyEnum(op.getSem())));
  genericOp->setAttr(memSyncScope,
      rewriter.getStringAttr(stringifyEnum(op.getScope())));
  genericOp->setAttr("Software", rewriter.getUnitAttr());

  Value resultTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, tensorResultTy, resultBuf);
  rewriter.replaceOp(op, resultTensor);
  return success();
}

//===----------------------------------------------------------------------===//
// AtomicCASConverter::matchAndRewrite
//===----------------------------------------------------------------------===//

LogicalResult
AtomicCASConverter::matchAndRewrite(triton::AtomicCASOp op,
                                    PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value ptr = op.getPtr();
  Value cmp = op.getCmp();
  Value val = op.getVal();

  auto ptrMRT = dyn_cast<MemRefType>(ptr.getType());
  if (!ptrMRT)
    return op.emitOpError(
        "[AtomicCASConverter] ptr must be MemRefType after type conversion");

  Type elemTy = ptrMRT.getElementType();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Load old value.
  Value old = rewriter.create<memref::LoadOp>(loc, ptr, ValueRange{c0});

  // Compare old == cmp.
  Value eq;
  if (isa<FloatType>(elemTy)) {
    eq = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, old,
                                        cmp);
  } else {
    eq =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, old, cmp);
  }

  // Conditionally store new value.
  rewriter.create<scf::IfOp>(loc, eq, [&](OpBuilder &b, Location l) {
    b.create<memref::StoreOp>(l, val, ptr, ValueRange{c0});
    b.create<scf::YieldOp>(l);
  });

  // Result is the old value before the potential swap.
  rewriter.replaceOp(op, old);
  return success();
}

//===----------------------------------------------------------------------===//
// Phase-1 canonicalizers
//===----------------------------------------------------------------------===//

// ── ScalarAtomicRMWCanonicalizer ──────────────────────────────────────────

LogicalResult
ScalarAtomicRMWCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                              PatternRewriter &rewriter) const {
  // Only touch scalar (non-tensor) ptr ops.
  if (isa<RankedTensorType>(op.getPtr().getType()))
    return failure();

  Value mask = op.getMask();
  if (!mask)
    return failure();

  // Only rewrite if mask is a rank-1 tensor<1xi1>.
  auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskTy || maskTy.getRank() != 1 || maskTy.getDimSize(0) != 1)
    return failure();

  Location loc = op.getLoc();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value scalarMask =
      rewriter.create<tensor::ExtractOp>(loc, mask, ValueRange{c0});

  rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
      op, op.getResult().getType(), op.getAtomicRmwOpAttr(), op.getPtr(),
      op.getVal(), scalarMask, op.getSemAttr(), op.getScopeAttr());
  return success();
}

// ── ScalarAtomicCASCanonicalizer ─────────────────────────────────────────

LogicalResult
ScalarAtomicCASCanonicalizer::matchAndRewrite(triton::AtomicCASOp op,
                                              PatternRewriter &rewriter) const {
  // AtomicCASOp does not have a mask operand in current Triton; nothing to do.
  // This canonicalizer is kept as a placeholder for future extension.
  return failure();
}

// ── AtomicMaxMinCanonicalizer ─────────────────────────────────────────────

LogicalResult
AtomicMaxMinCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                           PatternRewriter &rewriter) const {
  auto kind = op.getAtomicRmwOp();
  bool isMaxMin = (kind == triton::RMWOp::MAX || kind == triton::RMWOp::MIN ||
                   kind == triton::RMWOp::UMAX || kind == triton::RMWOp::UMIN);
  if (!isMaxMin)
    return failure();

  // Determine the pointee type from the (still-triton) ptr operand.
  Type pointeeTy = getPointeeType(op.getPtr().getType());
  if (!pointeeTy)
    return failure();

  Value val = op.getVal();
  Type valTy = getElementTypeOrSelf(val.getType());

  if (valTy == pointeeTy)
    return failure(); // already matching – nothing to do

  Location loc = op.getLoc();

  // Compute the destination type (preserve tensor wrapper if present).
  Type dstTy;
  if (auto valTensorTy = dyn_cast<RankedTensorType>(val.getType())) {
    dstTy = RankedTensorType::get(valTensorTy.getShape(), pointeeTy);
  } else {
    dstTy = pointeeTy;
  }

  Value casted;
  if (isa<FloatType>(pointeeTy) && isa<FloatType>(valTy)) {
    unsigned dstW = cast<FloatType>(pointeeTy).getWidth();
    unsigned srcW = cast<FloatType>(valTy).getWidth();
    if (dstW > srcW)
      casted = rewriter.create<arith::ExtFOp>(loc, dstTy, val);
    else
      casted = rewriter.create<arith::TruncFOp>(loc, dstTy, val);
  } else if (isa<IntegerType>(pointeeTy) && isa<IntegerType>(valTy)) {
    unsigned dstW = cast<IntegerType>(pointeeTy).getWidth();
    unsigned srcW = cast<IntegerType>(valTy).getWidth();
    if (dstW > srcW)
      casted = rewriter.create<arith::ExtSIOp>(loc, dstTy, val);
    else
      casted = rewriter.create<arith::TruncIOp>(loc, dstTy, val);
  } else {
    // Mixed float/int – unsupported here.
    return failure();
  }

  rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
      op, op.getResult().getType(), op.getAtomicRmwOpAttr(), op.getPtr(),
      casted, op.getMask(), op.getSemAttr(), op.getScopeAttr());
  return success();
}
