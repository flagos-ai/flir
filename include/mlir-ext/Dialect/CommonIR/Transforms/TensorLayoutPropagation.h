#ifndef COMMONIR_TRANSFORMS_TENSOR_LAYOUT_PROPAGATION_H
#define COMMONIR_TRANSFORMS_TENSOR_LAYOUT_PROPAGATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::tile {

// Concrete encodings and dialect-specific rewrites belong to the adapter.
// A null operand encoding makes an operation a propagation boundary.
struct TensorLayoutAdapter {
  virtual ~TensorLayoutAdapter() = default;
  // Explicit requirements remain boundaries for automatic propagation.
  virtual bool canChangeEncoding(Value value, Attribute encoding) const {
    return true;
  }
  virtual Value getConversionSource(Value value) const = 0;
  virtual Attribute inferOperandEncoding(Operation *op,
                                         Attribute resultEncoding) const = 0;
  virtual Value cloneWithLayout(Operation *op, RankedTensorType resultType,
                                ValueRange operands) const = 0;
  // Materialize at the consumer, preserving intervening synchronization.
  virtual Value convertLayout(Value value, RankedTensorType type,
                              Operation *before) const = 0;
};

struct TensorLayoutRequirement {
  OpOperand *use;
  Attribute encoding;
  // Hardware ports may require a conversion without rematerializing inputs.
  bool propagate = true;
};

struct TensorLayoutRequirements {
  SmallVector<TensorLayoutRequirement> uses;
  // Rules must validate in-place producer changes before adding assignments.
  llvm::MapVector<Value, Attribute> assignments;
};

// Collect values tied by layout conversions and scf.for slots. Operation rules
// validate the producers and boundary users of this shared, read-only graph.
SmallVector<Value>
collectTensorLayoutComponent(Value seed, const TensorLayoutAdapter &adapter);

// Analyze requirements before applying any IR changes. Unknown consumers keep
// their original layouts through boundary conversions.
void applyTensorLayoutRequirements(const TensorLayoutRequirements &requirements,
                                   const TensorLayoutAdapter &adapter);

} // namespace mlir::triton::tile

#endif
