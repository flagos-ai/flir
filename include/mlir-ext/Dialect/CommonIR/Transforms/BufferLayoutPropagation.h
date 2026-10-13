#ifndef COMMONIR_TRANSFORMS_BUFFERLAYOUTPROPAGATION_H
#define COMMONIR_TRANSFORMS_BUFFERLAYOUTPROPAGATION_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::tile {

// Buffer views share storage, so their layouts must be planned together.
// Operation/type knowledge belongs to the adapter; encodings stay opaque here.
struct BufferLayoutAdapter {
  virtual ~BufferLayoutAdapter() = default;
  virtual bool isAllocation(Value value) const = 0;
  virtual Value getViewSource(Value value) const = 0;
  virtual Attribute toSourceEncoding(Value view, Attribute encoding) const = 0;
  virtual Attribute toViewEncoding(Value view, Attribute encoding) const = 0;
  virtual Type withEncoding(Value value, Attribute encoding) const = 0;
};

struct BufferLayoutRequirement {
  Value value;
  Attribute encoding;
};

struct BufferLayoutPlan {
  llvm::MapVector<Value, Type> types;
  SmallVector<OpOperand *> accesses;

  Type getType(Value value) const;
  void apply() const;
};

Value getBufferLayoutRoot(Value value, const BufferLayoutAdapter &adapter);
// Read-only: reject conflicting root requirements or unsupported projections.
FailureOr<BufferLayoutPlan>
planBufferLayouts(ArrayRef<BufferLayoutRequirement> requirements,
                  const BufferLayoutAdapter &adapter);

} // namespace mlir::triton::tile
#endif
