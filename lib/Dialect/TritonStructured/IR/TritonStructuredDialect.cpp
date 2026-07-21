#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

using namespace mlir;
using namespace mlir::tts;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonStructuredDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"
      >();
}

struct TTSLoadContiguousPattern : public OpRewritePattern<tts::LoadOp> {
  using OpRewritePattern<tts::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tts::LoadOp op,
                                PatternRewriter &rewriter) const override {

    auto makeTensorPtrOp =
        op->getOperand(0).getDefiningOp<tts::MakeTensorPtrOp>();
    if (!makeTensorPtrOp)
      return failure();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto rank = tensorType.getRank();

    if (rank != 2) {
      // TODO: Make it work for higher rank. Here simplify for calculate
      // strides.
      return failure();
    }

    auto mixedStrides = makeTensorPtrOp.getMixedStrides();
    assert(mixedStrides.size() == 2);

    // Check if first dimension (dim0) stride is 1
    auto dim0Stride = mixedStrides[0];
    auto dim0StrideVal = getConstValue(dim0Stride);
    if (!dim0StrideVal || dim0StrideVal.value() != 1) {
      return failure();
    }

    // Prevent infinite loop: check that dim1 stride is not also 1
    auto dim1Stride = mixedStrides[1];
    auto dim1StrideVal = getConstValue(dim1Stride);
    if (dim1StrideVal && dim1StrideVal.value() == 1) {
      return failure();
    }

    // NOTE: In FlagGems outer op, MakeTensorPtrOp may has stride = 0. So we
    // need to check the order to decide whether to transpose
    // FIXME: How to handle the stride = 0 case?
    auto order = makeTensorPtrOp.getOrder();
    if (!order.empty() &&
        llvm::is_sorted(makeTensorPtrOp.getOrder(), std::greater<>())) {
      return failure();
    }

    SmallVector<OpFoldResult> newStrides;
    // New dim0 stride = original dim0 size
    // newStrides.push_back(rewriter.getIndexAttr(sizes[0]));
    newStrides.push_back(mixedStrides[1]);
    // New dim1 stride = 1
    newStrides.push_back(rewriter.getIndexAttr(1));

    // Create new strides array with dim1 stride set to 1 and dim0 stride set to
    // size[0]
    auto sizes = makeTensorPtrOp.getSizes();
    // Create new sizes array (swapped)
    SmallVector<int64_t> newSizes;
    newSizes.push_back(sizes[1]); // Original dim1 size becomes new dim0 size
    newSizes.push_back(sizes[0]); // Original dim0 size becomes new dim1 size

    auto offset = makeTensorPtrOp.getMixedOffsets();
    // Shape is original source tensor shape, no need to swap
    SmallVector<OpFoldResult> mixedShape = makeTensorPtrOp.getMixedShape();
    SmallVector<OpFoldResult> newShape =
        makeTensorPtrOp.isSplitPtr()
            ? SmallVector<OpFoldResult>{mixedShape[1], mixedShape[0]}
            : mixedShape;

    // Create new offsets array (swapped)
    auto mixedOffsets = makeTensorPtrOp.getMixedOffsets();
    assert(mixedOffsets.size() == 2);
    // dim0 stride is 1
    SmallVector<OpFoldResult> newOffsets = {mixedOffsets[1], mixedOffsets[0]};

    // TODO: Now not used order
    // Create the new MakeTensorPtrOp with swapped dimensions
    auto newMakeTensorPtrOp = rewriter.create<tts::MakeTensorPtrOp>(
        op->getLoc(), makeTensorPtrOp.getBase(), newSizes, newStrides,
        newOffsets, newShape, order);

    auto dims = op.getMixedMaskDims();
    assert(!op.hasMask() || dims.size() == 2);
    SmallVector<OpFoldResult> newDims =
        op.hasMask() ? SmallVector<OpFoldResult>{dims[1], dims[0]} : dims;

    auto newLoadOp = rewriter.create<tts::LoadOp>(
        op->getLoc(), newMakeTensorPtrOp, newDims, op.getOther());

    Value init = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), tensorType.getShape(), tensorType.getElementType());

    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        op->getLoc(), newLoadOp, init,
        rewriter.getDenseI64ArrayAttr(
            {1, 0})); // Permutation to swap dimensions
    rewriter.replaceOp(op, transposeOp);

    return success();
  }
};

void TritonStructuredDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<TTSLoadContiguousPattern>(getContext());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp.inc"
