#ifndef COMMONIR_IR_LAYOUTHINTATTRS_H
#define COMMONIR_IR_LAYOUTHINTATTRS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace mlir::triton::tile {

static constexpr const char *kExplicitEncodingAttrPrefix =
    "tile.explicit_encoding.";
static constexpr const char *kExplicitMemoryEncodingAttrName =
    "tile.explicit_memory_encoding";

inline std::string getExplicitEncodingAttrName(unsigned resultNumber) {
  return (llvm::Twine(kExplicitEncodingAttrPrefix) + llvm::Twine(resultNumber))
      .str();
}

inline const char *getExplicitMemoryEncodingAttrName() {
  return kExplicitMemoryEncodingAttrName;
}

inline Attribute getExplicitResultEncoding(Operation *op,
                                           unsigned resultNumber) {
  return op->getAttr(getExplicitEncodingAttrName(resultNumber));
}

inline void setExplicitResultEncoding(Operation *op, unsigned resultNumber,
                                      Attribute encoding) {
  op->setAttr(getExplicitEncodingAttrName(resultNumber), encoding);
}

inline void setExplicitResultEncoding(OpResult result, Attribute encoding) {
  setExplicitResultEncoding(result.getOwner(), result.getResultNumber(),
                            encoding);
}

inline Attribute getExplicitMemoryEncoding(Operation *op) {
  return op->getAttr(kExplicitMemoryEncodingAttrName);
}

inline void setExplicitMemoryEncoding(Operation *op, Attribute encoding) {
  op->setAttr(kExplicitMemoryEncodingAttrName, encoding);
}

inline Attribute getExplicitValueEncoding(Value value) {
  if (auto result = dyn_cast<OpResult>(value))
    return getExplicitResultEncoding(result.getOwner(),
                                     result.getResultNumber());
  return nullptr;
}

// Use convert_layout to carry hints on block arguments and region results,
// where indexed attributes are absent or unsafe across control-flow rewrites.
inline LogicalResult trySetExplicitValueEncoding(Value value,
                                                 Attribute encoding) {
  auto result = dyn_cast<OpResult>(value);
  if (!result || result.getOwner()->getNumRegions())
    return failure();
  if (Attribute existing = getExplicitValueEncoding(value))
    if (existing != encoding)
      return failure();
  setExplicitResultEncoding(result, encoding);
  return success();
}

// Check agreement between result/operand constraints on a memory operation.
// The caller decides which operations require a common memory encoding.
LogicalResult inferExplicitMemoryEncoding(Operation *op, Attribute &encoding);

} // namespace mlir::triton::tile

#endif
