#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::triton::tile {

static LogicalResult mergeExplicitMemoryEncoding(Operation *op, Value value,
                                                 Attribute candidate,
                                                 Attribute &encoding) {
  if (!candidate)
    return success();

  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (tensorType.getEncoding() != candidate)
      return op->emitOpError("has explicit CommonIR encoding that does not match "
                             "the tensor type encoding");
  }

  if (!encoding) {
    encoding = candidate;
    return success();
  }

  if (encoding == candidate)
    return success();

  op->emitOpError("has conflicting explicit CommonIR memory encodings:\n  ")
      << encoding << "\nand\n  " << candidate;
  return failure();
}

LogicalResult inferExplicitMemoryEncoding(Operation *op, Attribute &encoding) {
  encoding = getExplicitMemoryEncoding(op);

  for (OpResult result : op->getResults()) {
    if (failed(mergeExplicitMemoryEncoding(
            op, result,
            getExplicitResultEncoding(op, result.getResultNumber()), encoding)))
      return failure();
  }

  for (OpOperand &operand : op->getOpOperands()) {
    if (failed(mergeExplicitMemoryEncoding(
            op, operand.get(), getExplicitValueEncoding(operand.get()), encoding)))
      return failure();
  }

  return success();
}

void clearLayoutHints(ModuleOp module) {
  module.walk([](Operation *op) {
    op->removeAttr(getExplicitMemoryEncodingAttrName());
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      op->removeAttr(getExplicitEncodingAttrName(i));
  });
}

} // namespace mlir::triton::tile
