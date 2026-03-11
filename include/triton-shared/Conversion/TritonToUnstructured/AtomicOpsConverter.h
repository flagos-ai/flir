//===- AtomicOpsConverter.h -----------------------------------------------===//
//
// Atomic op lowering patterns for the triton-shared pipeline.
//
// Pipeline placement
// ------------------
// Canonicalizers (Phase-1):
//   Registered in TritonToStructured or as a standalone pre-pass.
//   These rewrites run BEFORE type conversion so tt.ptr types are still intact.
//
//     ScalarAtomicRMWCanonicalizer  – normalise single-element tensor masks
//     ScalarAtomicCASCanonicalizer  – same for CAS
//     AtomicMaxMinCanonicalizer     – insert type-promotion casts for MAX/MIN
//
// Converters (Phase-2):
//   Registered in TritonToUnstructured (or UnstructuredToMemref).
//   These run AFTER TritonTypeConverter has rewritten tt.ptr → memref.
//
//     AtomicRMWConverter  –  tt.atomic_rmw  →  load + arith-op + store
//     AtomicCASConverter  –  tt.atomic_cas  →  load + cmpi/cmpf + scf.if +
//     store
//
// Both converters emit *software* (non-hardware) atomic sequences that are
// correct for single-core or UB-local execution.  AND/OR/XOR are always
// software; FADD/ADD/XCHG/MAX/MIN/UMAX/UMIN follow the same path.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace shared {

//===----------------------------------------------------------------------===//
// Phase-1 canonicalizers  (run before type conversion, on tt.ptr IR)
//===----------------------------------------------------------------------===//

/// Normalise a scalar AtomicRMWOp whose mask is a rank-1 tensor<1xi1> into an
/// op carrying a scalar i1 mask, so the Phase-2 converter only sees i1 masks.
///
///   tt.atomic_rmw ..., %ptr, %val, %mask_tensor  (mask : tensor<1xi1>)
/// →
///   %idx = arith.constant 0
///   %m   = tensor.extract %mask_tensor[%idx]
///   tt.atomic_rmw ..., %ptr, %val, %m            (mask : i1)
class ScalarAtomicRMWCanonicalizer
    : public OpRewritePattern<triton::AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

/// Same normalisation for AtomicCASOp.
class ScalarAtomicCASCanonicalizer
    : public OpRewritePattern<triton::AtomicCASOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override;
};

/// Insert arith extension/truncation casts so that the `val` operand of a
/// MAX/MIN AtomicRMWOp has the same type as the pointee type.
/// Example: fmax on f32-ptr with f16 val → insert arith.extf before the op.
class AtomicMaxMinCanonicalizer : public OpRewritePattern<triton::AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Phase-2 converters  (run after TritonTypeConverter: tt.ptr → memref)
//===----------------------------------------------------------------------===//

/// Lower triton::AtomicRMWOp to a software read-modify-write sequence.
///
/// ── Scalar case  (ptr already converted to memref<?xT>) ──────────────────
///
///   %c0  = arith.constant 0 : index
///   %old = memref.load %ptr[%c0]
///   %new = arith.{op}  %old, %val
///   memref.store %new, %ptr[%c0]         ← may be inside scf.if when masked
///   // replace op result with %old
///
/// ── Tensor case  (ptr already converted to memref<N0×…xT>) ───────────────
///
///   %result_buf = memref.alloc() : memref<N0×…xT>
///   linalg.generic ins(%ptr_memref, %val_memref [, %mask_memref])
///                  outs(%ptr_memref, %result_buf)
///   {
///     ^bb0(%ptr_elem, %val_elem, [%mask_elem,] %ptr_out, %res_init):
///       %new = arith.{op} %ptr_elem, %val_elem
///       // if mask: yield (%new, %ptr_elem) else yield (%ptr_elem, %ptr_elem)
///       linalg.yield %selected_new, %ptr_elem
///   }
///   %result = bufferization.to_tensor %result_buf
///
class AtomicRMWConverter : public OpRewritePattern<triton::AtomicRMWOp> {
public:
  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Build the scalar arith binary op for the given rmwOp.
  /// lhs = old value loaded from memory; rhs = atomic operand.
  Value buildBinaryOp(OpBuilder &b, Location loc, triton::RMWOp rmwOp,
                      Type elemTy, Value lhs, Value rhs) const;

  static bool isSplatTrue(Value mask);
  static bool isSplatFalse(Value mask);
};

/// Lower triton::AtomicCASOp to a software compare-and-swap.
///
/// ── Scalar case ──────────────────────────────────────────────────────────
///
///   %c0  = arith.constant 0 : index
///   %old = memref.load %ptr[%c0]
///   %eq  = arith.cmpi eq, %old, %cmp     (arith.cmpf oeq for float)
///   scf.if %eq {
///     memref.store %val, %ptr[%c0]
///   }
///   // replace op result with %old
///
class AtomicCASConverter : public OpRewritePattern<triton::AtomicCASOp> {
public:
  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override;
};
//  explicit AtomicCASConverter(MLIRContext *ctx)
//      : OpConversionPattern<triton::AtomicCASOp>(ctx) {}
//
//  LogicalResult matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
//                                ConversionPatternRewriter &rewriter) const
//                                override;
//};

//===----------------------------------------------------------------------===//
// Registration helpers
//===----------------------------------------------------------------------===//

/// Populate Phase-1 (pre-conversion) canonicalization patterns.
/// Intended to be called from populateCanonicalizationPatterns() in the
/// TritonToStructured pass or a dedicated pre-pass.
inline void
populateAtomicCanonicalizationPatterns(RewritePatternSet &patterns) {
  patterns.add<ScalarAtomicRMWCanonicalizer, ScalarAtomicCASCanonicalizer,
               AtomicMaxMinCanonicalizer>(patterns.getContext());
}

/// Populate Phase-2 (post-conversion) conversion patterns.
/// Intended to be called from populateConversionPatterns() in the
/// TritonToUnstructured or UnstructuredToMemref pass.
inline void populateAtomicConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AtomicRMWConverter, AtomicCASConverter>(patterns.getContext());
}

} // namespace shared
} // namespace triton
} // namespace mlir
